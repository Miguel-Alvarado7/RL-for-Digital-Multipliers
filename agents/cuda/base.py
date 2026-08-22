"""Base tabular VECTORIZADA para agentes que corren sobre entornos CUDA.

Un "batch" = env.n_envs episodios en paralelo; todos avanzan en lockstep
(cada paso rellena una celda en todos), así que el cursor es uniforme entre
entornos en cada paso: s = cursor_pos[0] (0..CC-1).

Optimización clave (del motor del profesor): el estado es únicamente el
cursor y la política no depende del contenido de la rejilla, así que un
episodio entero se muestrea por adelantado en tres kernels
(_sample_episode_actions) y se evalúa de una sola vez con
env.commit_episodes, en lugar de CC iteraciones de Python con ~4
sincronizaciones CPU<->GPU cada una.

`collect_batch` devuelve estados, acciones y **retorno terminal** (n,).
Cada algoritmo deriva su propio objetivo:
  - MC        : retorno G directamente.
  - Q-learning: r_t + gamma * max_a' Q[s_{t+1}, a'].
  - SARSA     : r_t + gamma * Q[s_{t+1}, a'] (a' = acción realmente tomada).

En no-incremental r_t es 0 salvo el último paso (per_step_rewards lo
reconstruye poniendo el retorno G en el paso terminal), y Q[CC] (estado
terminal) es una fila de ceros, lo que cierra el bootstrap.

Update con índices duplicados: `Q[states, actions] += ...` con indexación
avanzada hace read-modify-write y ante índices repetidos -inevitables,
porque n_envs episodios comparten los mismos CC estados- solo sobrevive la
escritura de UN entorno. Por eso todos los algoritmos usan
index_put_(accumulate=True) y dividen por el conteo (media explícita).
"""

import numpy as np
import torch

from .._base import epsilon_at


class TabularAgentCUDA:
    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        device="cuda",
        seed=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.Q = torch.zeros(n_states, n_actions, device=self.device)
        self.N = torch.zeros(n_states, n_actions, dtype=torch.int64, device=self.device)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        if seed is not None:
            torch.manual_seed(seed)

    # =========================================================================
    # Política
    # =========================================================================

    def epsilon_at(self, episode, max_episodes):
        return epsilon_at(self.epsilon_start, self.epsilon_end, episode, max_episodes)

    def _sample_episode_actions(self, T, n, epsilon):
        """Muestrea el episodio COMPLETO de golpe: (T, n) acciones epsilon-greedy.

        Cada entorno decide por separado en cada paso:
          - con prob epsilon: acción aleatoria uniforme (exploración);
          - con prob 1-eps  : argmax(Q[t]) (explotación).

        Clave del rendimiento: el estado es únicamente el cursor, y el cursor
        en el paso t vale exactamente t en todos los entornos. La política por
        tanto NO depende de lo que se haya escrito en la rejilla, y Q no cambia
        dentro de un batch. Así que las CC decisiones de un episodio se pueden
        tomar por adelantado, en tres kernels, en lugar de CC iteraciones de
        Python con sus sincronizaciones CPU<->GPU.

        Returns:
            (T, n) LongTensor en self.device.
        """
        greedy = self.Q[:T].argmax(dim=1)                     # (T,) sin .item()
        explore = torch.rand(T, n, device=self.device) < epsilon
        random_a = torch.randint(0, self.Q.shape[1], (T, n), device=self.device)
        return torch.where(explore, random_a, greedy[:, None])

    # =========================================================================
    # Recogida y actualización de episodios (batched)
    # =========================================================================

    def collect_batch(self, env, epsilon):
        """Ejecuta un batch: n_envs episodios en paralelo con política epsilon-greedy.

        Todos los entornos parten del mismo estado (tabla vacía, cursor 0) y
        rellenan una celda por paso. Tras CC pasos todos terminan a la vez.

        Args:
            env:     BinaryMathEnvCUDA o BinaryMathEnvCUDAOptimized.
            epsilon: Probabilidad de exploración (0 => greedy puro).

        Returns:
            states:  (CC, n_envs) int64 — estado s por (paso, entorno).
            actions: (CC, n_envs) int64 — acción a por (paso, entorno).
            returns: (n_envs,) float32 — retorno terminal G por entorno.
        """
        n, T = env.n_envs, env.CC

        # Episodio completo sin loop de Python: ver _sample_episode_actions.
        actions = self._sample_episode_actions(T, n, epsilon)
        returns = env.commit_episodes(actions.T.contiguous())

        # El estado en el paso t es el cursor, que vale t en todos los entornos.
        states = torch.arange(T, device=self.device).unsqueeze(1).expand(T, n)

        return states, actions, returns

    def per_step_rewards(self, actions, returns):
        """Reconstruye la recompensa por paso (T, n) a partir del retorno terminal.

        En modo no-incremental r_t = 0 salvo el paso terminal (t = T-1), donde
        llega el retorno del episodio. Necesario para objetivos TD
        (Q-learning, SARSA); MC usa el retorno directamente.

        Args:
            actions: (T, n) int64 — acciones ejecutadas.
            returns: (n,) float32 — retorno terminal por entorno.

        Returns:
            (T, n) float32 — recompensa por paso.
        """
        T, n = actions.shape
        rewards = torch.zeros(T, n, device=self.device)
        rewards[-1] = returns
        return rewards

    def update(self, states, actions, returns):
        raise NotImplementedError

    # =========================================================================
    # Entrenamiento y evaluación
    # =========================================================================

    #: Columnas de la matriz de estadisticas por batch que devuelve train().
    BATCH_STAT_COLUMNS = ("batch", "episode_start", "epsilon", "mean", "std",
                          "min", "max", "p25", "p50", "p75")

    def train(self, env, n_batches, on_batch=None, early_stop_return=0.0,
              greedy_eval_every=10, checkpoint_every=0, on_checkpoint=None):
        """Entrena por `n_batches` batches. Cada batch = n_envs episodios.

        Criterio de parada HÍBRIDO (política greedy, no episodio suelto):
          1. Candidato rápido: si un episodio del behavior policy (ε-greedy)
             alcanza `early_stop_return`, NO es victoria aún — la exploración
             puede acertar un circuito por suerte sin que Q generalice.
          2. Confirmación: se ejecuta una corrida greedy (ε=0). Al ser el
             estado solo el cursor, la política greedy es determinista: una
             corrida caracteriza por completo a la política.
          3. Solo se detiene si la corrida greedy cumple el umbral en TODOS
             los entornos ((g_returns >= umbral).all()). Si el candidato no se
             confirma, se continúa entrenando (se reporta vía callback).
          Cada `greedy_eval_every` batches se ejecuta la corrida greedy aunque
          no haya candidato, para monitorear la convergencia real de la política.

        Args:
            env:               Entorno CUDA batched.
            n_batches:         Número de batches (cada uno evalúa n_envs episodios).
            on_batch:          Callback opcional on_batch(batch_idx, epsilon,
                               returns_tensor, greedy_info, batch_grids) para
                               logging. greedy_info es dict (o None) con claves
                               'mean', 'candidate', 'confirmed'.
            early_stop_return: Umbral de retorno. None desactiva la parada.
                               En CUDA el óptimo es 0 (circuito perfecto).
            greedy_eval_every: Cada cuántos batches se evalúa la política greedy
                               (0 desactiva el monitoreo periódico).
            checkpoint_every:  Cada cuántos batches se llama a `on_checkpoint`
                               (0 lo desactiva).
            on_checkpoint:     Callback opcional on_checkpoint(n_batches_hechos,
                               stats_hasta_ahora) para volcar estado parcial.

        Returns:
            stats:   (batches_ejecutados, 10) float64 — una fila POR BATCH, con
                     las columnas de BATCH_STAT_COLUMNS.
            stopped: True si se detuvo porque la política greedy quedó confirmada.
        """
        stopped = False
        threshold = early_stop_return if early_stop_return is not None else 0.0

        # Estadisticas POR BATCH, calculadas donde nacen los datos. Antes se
        # acumulaba un retorno por episodio y se agregaba al guardar: con 50M
        # episodios eso eran ~50M tuplas de Python vivas a la vez (medido:
        # ~162 MB de RSS por millon de episodios, o sea ~9 GB al final de una
        # corrida de 50M). Asi el coste es O(n_batches), no O(episodios).
        #
        # En float64 a proposito: la ruta antigua pasaba por float() de Python
        # y np.array(dtype=float), de modo que las estadisticas salian en
        # float64. Calcularlas en el float32 del tensor daria numeros distintos.
        n_envs = env.n_envs
        stats = np.empty((n_batches, len(self.BATCH_STAT_COLUMNS)), dtype=np.float64)
        n_done = 0

        for b in range(n_batches):
            epsilon = self.epsilon_at(b, n_batches)
            states, actions, returns = self.collect_batch(env, epsilon)
            self.update(states, actions, returns)

            # Las acciones del batch de comportamiento SON la rejilla; usarlas
            # directamente evita depender de env.suma_grid, que la corrida
            # greedy posterior sobrescribiría.
            batch_grids = actions.T

            r = returns.double().cpu().numpy()
            stats[b] = (b, b * n_envs, epsilon, r.mean(), r.std(),
                        r.min(), r.max(), *np.percentile(r, [25, 50, 75]))
            n_done = b + 1

            # ¿Hay candidato (un episodio alcanzó el umbral) o toca monitoreo?
            candidate = early_stop_return is not None and \
                bool((returns >= early_stop_return).any())
            periodic = greedy_eval_every > 0 and b % greedy_eval_every == 0

            greedy_info = None
            if candidate or periodic:
                g_return = float(self.greedy_return(env))
                confirmed = g_return >= threshold
                greedy_info = {
                    'mean': g_return,
                    'candidate': bool(candidate),
                    'confirmed': bool(confirmed),
                }
                # Parar solo si el candidato se confirma con la política greedy.
                if candidate and confirmed:
                    stopped = True

            if on_batch is not None:
                on_batch(b, epsilon, returns, greedy_info, batch_grids)

            if (checkpoint_every and on_checkpoint is not None
                    and n_done % checkpoint_every == 0):
                on_checkpoint(n_done, stats[:n_done])

            if stopped:
                break

        return stats[:n_done], stopped

    def greedy_return(self, env):
        """Retorno de la política greedy (argmax Q), evaluando UNA trayectoria.

        La política greedy es determinista y el estado es solo el cursor, así
        que los n_envs episodios de una corrida con epsilon=0 son idénticos
        entre sí: basta uno. Medido, un batch greedy de 256 entornos producía
        1 único retorno distinto — el 99.6% del cómputo era redundante.

        Returns:
            float — retorno de la única trayectoria greedy.
        """
        greedy = self.Q[:env.CC].argmax(dim=1)          # (CC,)
        return env.evaluate_grids(greedy.unsqueeze(0))[0].item()

    def evaluate(self, env):
        """Evalúa la política greedy (argmax Q).

        Como no hay last_metrics en los entornos CUDA, se reporta el retorno
        (proxy directo del error: reward = 0 es un circuito perfecto).

        Returns:
            (1,) float — retorno greedy. Es un array de un elemento a propósito:
            la política es determinista, repetirla N veces daría N copias del
            mismo número (ver greedy_return).
        """
        return np.asarray([self.greedy_return(env)])

    def greedy_action(self, s):
        """Acción greedy (argmax Q) para un estado escalar s."""
        return int(self.Q[s].argmax().item())
