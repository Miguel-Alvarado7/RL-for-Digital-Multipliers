"""Base tabular para agentes CPU (un episodio = un entorno, gym API).

La observación del entorno es [cursor_position, casillas_rellenas]; como cada
paso rellena una celda, el estado efectivo es el cursor (0..CC). Las
recompensas intermedias son 0 y la terminal llega al completar la tabla.

`collect_episode` devuelve la trayectoria con la información mínima para
cualquier update tabular: (s, a, r, s_next, a_next). MC ignora la parte TD;
Q-learning y SARSA usan r y el siguiente valor.
"""

import numpy as np

from .._base import choose_epsilon_greedy, epsilon_at


class TabularAgentCPU:
    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        rng_seed=None,
    ):
        self.Q = np.zeros((n_states, n_actions))
        self.N = np.zeros((n_states, n_actions), dtype=np.int64)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.rng = np.random.default_rng(rng_seed)

    def epsilon_at(self, episode, max_episodes):
        return epsilon_at(self.epsilon_start, self.epsilon_end, episode, max_episodes)

    # =========================================================================
    # Recogida de episodios
    # =========================================================================

    def collect_episode(self, env, epsilon, seed):
        """Ejecuta un episodio completo con política epsilon-greedy.

        Returns:
            trajectory: lista de (s, a, r, s_next, a_next). En el último paso
                        a_next es None (estado terminal, Q[s_next]=0).
        """
        obs, _ = env.reset(seed=seed)
        trajectory = []
        terminated = truncated = False
        s = int(obs[0])
        while not (terminated or truncated):
            a = choose_epsilon_greedy(self.Q[s], epsilon, self.rng)
            obs, reward, terminated, truncated, _ = env.step(a)
            s_next = int(obs[0])
            if not (terminated or truncated):
                a_next = choose_epsilon_greedy(self.Q[s_next], epsilon, self.rng)
            else:
                a_next = None
            trajectory.append((s, a, float(reward), s_next, a_next))
            s = s_next
        return trajectory

    @staticmethod
    def episode_return(trajectory):
        return sum(r for *_, r, _, _ in trajectory)

    # =========================================================================
    # Actualización (implementada por cada algoritmo)
    # =========================================================================

    def update(self, trajectory):
        raise NotImplementedError

    # =========================================================================
    # Entrenamiento y evaluación
    # =========================================================================

    def train(self, env, episodes, base_seed=42, on_episode=None,
              early_stop_return=None):
        """Entrena el agente por `episodes` episodios.

        Si `early_stop_return` no es None, detiene el entrenamiento en cuanto
        un episodio alcanza ese retorno (p. ej. el óptimo de 100.0).

        Nota: la secuencia de casos de prueba del entorno (np.random global)
        debe estar sembrada por el llamador para reproducibilidad.
        """
        returns = np.empty(episodes)
        epsilons = np.empty(episodes)
        stopped = False
        for ep in range(episodes):
            epsilon = self.epsilon_at(ep, episodes)
            trajectory = self.collect_episode(env, epsilon, seed=base_seed + ep)
            self.update(trajectory)
            return_ = self.episode_return(trajectory)
            returns[ep] = return_
            epsilons[ep] = epsilon
            if on_episode is not None:
                on_episode(ep, return_, epsilon)
            if early_stop_return is not None and return_ >= early_stop_return:
                stopped = True
                break
        if stopped:
            returns = returns[: ep + 1]
            epsilons = epsilons[: ep + 1]
        return returns, epsilons

    def greedy_action(self, s):
        return int(np.argmax(self.Q[s]))

    def evaluate(self, env, episodes, base_seed=1000):
        """Evalúa la política greedy (argmax Q) sobre episodios con semillas frescas."""
        returns = np.empty(episodes)
        errors = np.empty(episodes)
        for i in range(episodes):
            trajectory = self.collect_episode(env, epsilon=0.0, seed=base_seed + i)
            returns[i] = self.episode_return(trajectory)
            metrics = getattr(env, "last_metrics", None)
            if metrics and "error_mean" in metrics:
                errors[i] = metrics["error_mean"]
            else:
                errors[i] = float("nan")
        return returns, errors
