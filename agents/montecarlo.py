"""Agente de Monte Carlo on-policy (first-visit) para BinaryMathEnv.

Estado = observación del entorno: [cursor_position, casillas_rellenas].
Como ambas componentes siempre son iguales (cada paso llena una celda),
el estado efectivo es el cursor (0..CC).

Puesto que las recompensas intermedias son 0 y la recompensa terminal llega
solo al final del episodio, el retorno G_t es el mismo para todos los pasos
del episodio. Además el cursor crece estrictamente, así que ningún par (s, a)
se repite dentro de un episodio: first-visit == every-visit.

Update:  Q[s, a] += alpha * (G - Q[s, a])   para todo (s, a) visitado.
"""

import numpy as np

from .epsilon_greedy import choose_epsilon_greedy


class MonteCarloAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        epsilon_start=1.0,
        epsilon_end=0.01,
        rng_seed=None,
    ):
        self.Q = np.zeros((n_states, n_actions))
        self.N = np.zeros((n_states, n_actions), dtype=np.int64)
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.rng = np.random.default_rng(rng_seed)

    def epsilon_at(self, episode, max_episodes):
        """Decaimiento lineal de epsilon a lo largo del entrenamiento."""
        if max_episodes <= 1:
            return self.epsilon_end
        frac = episode / (max_episodes - 1)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def collect_episode(self, env, epsilon, seed):
        """Ejecuta un episodio completo con política epsilon-greedy.

        Returns:
            episode: lista de (s, a).
            total_return: recompensa terminal (las intermedias son 0).
        """
        obs, _ = env.reset(seed=seed)
        episode = []
        terminated = truncated = False
        total_return = 0.0
        while not (terminated or truncated):
            s = int(obs[0])
            a = choose_epsilon_greedy(self.Q[s], epsilon, self.rng)
            obs, reward, terminated, truncated, _ = env.step(a)
            episode.append((s, a))
            total_return += float(reward)
        return episode, total_return

    def update(self, episode, return_):
        for s, a in episode:
            self.Q[s, a] += self.alpha * (return_ - self.Q[s, a])
            self.N[s, a] += 1

    def train(self, env, episodes, base_seed=42, on_episode=None):
        """Entrena el agente por `episodes` episodios.

        Nota: la secuencia de casos de prueba del entorno (np.random global)
        debe estar sembrada por el llamador para reproducibilidad.
        """
        returns = np.empty(episodes)
        epsilons = np.empty(episodes)
        for ep in range(episodes):
            epsilon = self.epsilon_at(ep, episodes)
            episode, return_ = self.collect_episode(env, epsilon, seed=base_seed + ep)
            self.update(episode, return_)
            returns[ep] = return_
            epsilons[ep] = epsilon
            if on_episode is not None:
                on_episode(ep, return_, epsilon)
        return returns, epsilons

    def greedy_action(self, s):
        return int(np.argmax(self.Q[s]))

    def evaluate(self, env, episodes, base_seed=1000):
        """Evalúa la política greedy (argmax Q) sobre episodios con semillas frescas."""
        returns = np.empty(episodes)
        errors = np.empty(episodes)
        for i in range(episodes):
            episode, return_ = self.collect_episode(env, epsilon=0.0, seed=base_seed + i)
            returns[i] = return_
            metrics = getattr(env, "last_metrics", None)
            if metrics and "error_mean" in metrics:
                errors[i] = metrics["error_mean"]
            else:
                errors[i] = float("nan")
        return returns, errors
