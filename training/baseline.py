"""Baselines aleatorios (referencia) para CPU y CUDA."""

import numpy as np


def random_baseline_cpu(env, n_episodes, base_seed):
    """Política aleatoria uniforme como referencia (CPU, un episodio a la vez).

    Returns:
        (n_episodes,) float — retornos de cada episodio aleatorio.
    """
    returns = np.empty(n_episodes)
    for i in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + i)
        terminated = truncated = False
        total = 0.0
        while not (terminated or truncated):
            action = np.random.randint(env.action_space.n)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
        returns[i] = total
    return returns


def random_baseline_cuda(agent_cls, env, n_batches, seed):
    """Política aleatoria uniforme como referencia (batched).

    Reutiliza el sampler del agente con epsilon=1.0 (exploración pura).
    `agent_cls` es cualquier TabularAgentCUDA; con epsilon=1.0 los valores de
    Q no influyen.

    Returns:
        (n_batches*n_envs,) float — retornos de cada episodio aleatorio.
    """
    agent = agent_cls(
        n_states=env.CC + 1,
        n_actions=env.n_actions,
        device=env.device.type,
        seed=seed,
    )
    returns_all = []
    for _ in range(n_batches):
        _, _, returns = agent.collect_batch(env, epsilon=1.0)
        returns_all.append(returns.cpu().numpy())
    return np.concatenate(returns_all)
