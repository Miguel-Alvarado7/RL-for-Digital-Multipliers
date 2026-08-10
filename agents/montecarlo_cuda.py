"""Agente de Monte Carlo on-policy (first-visit) VECTORIZADO para entornos CUDA.

Funciona sobre BinaryMathEnvCUDA / BinaryMathEnvCUDAOptimized (API por-batch).

Diferencia clave vs. MonteCarloAgent (agents/montecarlo.py):
  - MonteCarloAgent  : un episodio = un entorno, evaluado con iverilog/vvp.
  - MonteCarloAgentCUDA: un "batch" = n_envs episodios en paralelo, evaluados
    con operaciones vectorizadas de torch (sin iverilog).

Estado = cursor_pos. Como todos los entornos del batch avanzan en lockstep
(cada paso rellena exactamente una celda en todos), el cursor es uniforme
entre entornos en cada paso y vale exactamente t: s = t (0..CC-1).

De ahí sale la optimización principal: la política NO depende del contenido de
la rejilla, solo del paso. Un episodio entero se muestrea por adelantado en
tres kernels y se evalúa de una sola vez (env.commit_episodes), en lugar de CC
iteraciones de Python con ~4 sincronizaciones CPU<->GPU cada una.

Recompensa: en modo no-incremental solo llega al terminar el episodio
(env.rewards). Rango CUDA:
  - BinaryMathEnvCUDA          : [-100, 0]  (óptimo = 0)
  - BinaryMathEnvCUDAOptimized : [ -10, 0]  (óptimo = 0)
Al ser recompensas terminales y el cursor estrictamente creciente, ningún
par (s, a) se repite dentro de un episodio: first-visit == every-visit.

Update vectorizado:  Q[s, a] += alpha * (media de los G_i de los episodios que
visitaron (s, a) - Q[s, a]).  Ver MonteCarloAgentCUDA.update sobre por qué el
promedio tiene que ser explícito.
"""

import numpy as np
import torch


class MonteCarloAgentCUDA:
    """Monte Carlo on-policy batched para entornos CUDA.

    Un "batch" de entrenamiento ejecuta `env.n_envs` episodios en paralelo;
    todos los entornos se reinician juntos y avanzan CC pasos en lockstep.
    Al final del batch se actualiza Q con los retornos de TODOS los episodios
    a la vez (indexing vectorizado), reemplazando el loop por-episodio del
    agente CPU.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        epsilon_start=1.0,
        epsilon_end=0.01,
        device="cuda",
        seed=None,
    ):
        """
        Args:
            n_states:       Número de estados. En este entorno el estado es el
                            cursor (0..CC-1), así que n_states >= CC. Se pasa
                            CC+1 para reservar el índice CC por seguridad.
            n_actions:      env.n_actions.
            alpha:          Tasa de aprendizaje MC.
            epsilon_start:  Epsilon inicial (exploración pura).
            epsilon_end:    Epsilon final (tras decaimiento lineal).
            device:         'cuda' o 'cpu'. Si CUDA no está disponible, cae a CPU.
            seed:           Semilla para torch.manual_seed (reproducibilidad).
        """
        # Fallback automático a CPU si se pide cuda sin GPU disponible.
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.Q = torch.zeros(n_states, n_actions, device=self.device)
        self.N = torch.zeros(n_states, n_actions, dtype=torch.int64, device=self.device)
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        if seed is not None:
            torch.manual_seed(seed)

    # =========================================================================
    # Política
    # =========================================================================

    def epsilon_at(self, episode, max_episodes):
        """Decaimiento lineal de epsilon a lo largo del entrenamiento."""
        if max_episodes <= 1:
            return self.epsilon_end
        frac = episode / (max_episodes - 1)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

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

    def update(self, states, actions, returns):
        """Actualización MC vectorizada: Q[s,a] += alpha*(mean(G) - Q[s,a]).

        Promedia sobre TODOS los episodios del batch que visitaron (s, a).
        Esto importa: `Q[states, actions] += ...` con indexación avanzada hace
        un read-modify-write, y ante índices duplicados —inevitables, porque
        n_envs episodios comparten los mismos CC estados— solo sobrevive la
        escritura de UN entorno, elegido de forma indefinida. Con n_envs=64 eso
        descartaba ~63 de cada 64 retornos, y subir n_envs no mejoraba el
        aprendizaje en absoluto. index_put_(accumulate=True) sí acumula.

        Args:
            states:  (T, n_envs) int64 — estados visitados.
            actions: (T, n_envs) int64 — acciones ejecutadas.
            returns: (n_envs,) float32 — retornos terminales por entorno.
        """
        s_flat = states.reshape(-1)
        a_flat = actions.reshape(-1)
        g_flat = returns[None, :].expand_as(states).reshape(-1).float()

        sums = torch.zeros_like(self.Q)
        counts = torch.zeros_like(self.Q)
        sums.index_put_((s_flat, a_flat), g_flat, accumulate=True)
        counts.index_put_((s_flat, a_flat), torch.ones_like(g_flat),
                          accumulate=True)

        visited = counts > 0
        mean_G = torch.where(visited, sums / counts.clamp(min=1.0), self.Q)
        self.Q += self.alpha * (mean_G - self.Q) * visited
        self.N += counts.to(self.N.dtype)

    # =========================================================================
    # Entrenamiento y evaluación
    # =========================================================================

    def train(self, env, n_batches, on_batch=None, early_stop_return=0.0,
              greedy_eval_every=10):
        """Entrena por `n_batches` batches. Cada batch = n_envs episodios.

        Criterio de parada HÍBRIDO (política greedy, no episodio suelto):
          1. Candidato rápido: si un episodio del behavior policy (ε-greedy)
             alcanza `early_stop_return`, NO es victoria aún — la exploración
             puede acertar un circuito por suerte sin que Q generalice.
          2. Confirmación: se ejecuta una corrida greedy (ε=0). Como el estado
             es solo el cursor, la política greedy es determinista: una corrida
             caracteriza por completo a la política.
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

        Returns:
            records: lista de (episodio_global, epsilon, retorno) por episodio real.
            stopped: True si se detuvo porque la política greedy quedó confirmada.
        """
        stopped = False
        threshold = early_stop_return if early_stop_return is not None else 0.0
        # Los retornos se acumulan en GPU y se bajan UNA vez al final: un
        # .cpu() por batch sincroniza y serializa el pipeline sin necesidad.
        all_returns, all_eps = [], []

        for b in range(n_batches):
            epsilon = self.epsilon_at(b, n_batches)
            states, actions, returns = self.collect_batch(env, epsilon)
            self.update(states, actions, returns)

            # Las acciones del batch de comportamiento SON la rejilla; usarlas
            # directamente evita depender de env.suma_grid, que la corrida
            # greedy posterior sobrescribiría.
            batch_grids = actions.T

            all_returns.append(returns)
            all_eps.append(epsilon)

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

            if stopped:
                break

        n = env.n_envs
        flat = torch.cat(all_returns).cpu().numpy()
        records = [(i, all_eps[i // n], float(r)) for i, r in enumerate(flat)]
        return records, stopped

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
