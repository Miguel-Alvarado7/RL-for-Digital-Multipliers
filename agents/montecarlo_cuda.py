"""Agente de Monte Carlo on-policy (first-visit) VECTORIZADO para entornos CUDA.

Funciona sobre BinaryMathEnvCUDA / BinaryMathEnvCUDAOptimized (API por-batch).

Diferencia clave vs. MonteCarloAgent (agents/montecarlo.py):
  - MonteCarloAgent  : un episodio = un entorno, evaluado con iverilog/vvp.
  - MonteCarloAgentCUDA: un "batch" = n_envs episodios en paralelo, evaluados
    con operaciones vectorizadas de torch (sin iverilog).

Estado = cursor_pos. Como todos los entornos del batch avanzan en lockstep
(cada paso rellena exactamente una celda en todos), el cursor es uniforme
entre entornos en cada paso: s = cursor_pos[0] (0..CC-1).

Recompensa: en modo no-incremental solo llega al terminar el episodio
(env.rewards). Rango CUDA:
  - BinaryMathEnvCUDA          : [-100, 0]  (óptimo = 0)
  - BinaryMathEnvCUDAOptimized : [ -10, 0]  (óptimo = 0)
Al ser recompensas terminales y el cursor estrictamente creciente, ningún
par (s, a) se repite dentro de un episodio: first-visit == every-visit.

Update vectorizado:  Q[s, a] += alpha * (G_i - Q[s, a])   para cada episodio i.
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

    def _sample_actions(self, s, epsilon, n):
        """Muestrea n acciones epsilon-greedy para el estado s.

        En un batch, todos los entornos comparten el mismo cursor s, así que
        se usa una sola fila Q[s]. Cada entorno decide por separado:
          - con prob epsilon: acción aleatoria uniforme (exploración);
          - con prob 1-eps  : argmax(Q[s]) (explotación).

        Returns:
            (n,) LongTensor en self.device.
        """
        q = self.Q[s]  # (n_actions,)
        greedy = int(q.argmax().item())
        explore = torch.rand(n, device=self.device) < epsilon
        random_a = torch.randint(0, q.shape[0], (n,), device=self.device)
        return torch.where(explore, random_a, greedy)

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
        env.reset()
        n = env.n_envs
        T = env.CC
        states = torch.empty(T, n, dtype=torch.int64, device=self.device)
        actions = torch.empty(T, n, dtype=torch.int64, device=self.device)
        returns = torch.zeros(n, device=self.device)

        for t in range(T):
            s = int(env.cursor_pos[0].item())  # cursor uniforme en el batch
            a = self._sample_actions(s, epsilon, n)
            states[t] = s
            actions[t] = a
            r, _ = env.step(a)                 # (n_envs,) recompensa del paso
            returns += r                       # en no-incremental solo importa la final

        return states, actions, returns

    def update(self, states, actions, returns):
        """Actualización MC vectorizada: Q[s,a] += alpha*(G - Q[s,a]).

        Args:
            states:  (T, n_envs) int64 — estados visitados.
            actions: (T, n_envs) int64 — acciones ejecutadas.
            returns: (n_envs,) float32 — retornos terminales por entorno.

        Nota: Q[states, actions] con estados/actions 2D devuelve un tensor
        (T, n_envs) de los valores actuales; el incremento se aplica in-place.
        """
        target = self.Q[states, actions]
        self.Q[states, actions] += self.alpha * (returns[None, :] - target)
        self.N[states, actions] += 1

    # =========================================================================
    # Entrenamiento y evaluación
    # =========================================================================

    def train(self, env, n_batches, on_batch=None, early_stop_return=0.0):
        """Entrena por `n_batches` batches. Cada batch = n_envs episodios.

        Args:
            env:               Entorno CUDA batched.
            n_batches:         Número de batches (cada uno evalúa n_envs episodios).
            on_batch:          Callback opcional on_batch(batch_idx, epsilon,
                               returns_tensor, returns_list) para logging.
            early_stop_return: Detiene en cuanto algún episodio alcanza este
                               retorno. En CUDA el óptimo es 0 (circuito
                               perfecto), a diferencia del agente CPU donde es 100.

        Returns:
            records: lista de (episodio_global, epsilon, retorno) por episodio real.
            stopped: True si se detuvo por early_stop.
        """
        records = []
        stopped = False
        ep_counter = 0
        for b in range(n_batches):
            epsilon = self.epsilon_at(b, n_batches)
            states, actions, returns = self.collect_batch(env, epsilon)
            self.update(states, actions, returns)

            ret_list = returns.cpu().tolist()
            for r in ret_list:
                records.append((ep_counter, epsilon, r))
                ep_counter += 1

            if on_batch is not None:
                on_batch(b, epsilon, returns, ret_list)

            if early_stop_return is not None and (returns >= early_stop_return).any():
                stopped = True
                break

        return records, stopped

    def evaluate(self, env, n_batches):
        """Evalúa la política greedy (argmax Q) sobre n_batches de episodios.

        Cada batch produce n_envs episodios greedy. Como no hay last_metrics
        en los entornos CUDA, se reporta el retorno (proxy directo del error:
        reward = 0 es un circuito perfecto).

        Returns:
            (n_batches*n_envs,) float — retornos de cada episodio greedy.
        """
        returns_all = []
        for _ in range(n_batches):
            _, _, returns = self.collect_batch(env, epsilon=0.0)
            returns_all.extend(returns.cpu().tolist())
        return np.asarray(returns_all)

    def greedy_action(self, s):
        """Acción greedy (argmax Q) para un estado escalar s."""
        return int(self.Q[s].argmax().item())
