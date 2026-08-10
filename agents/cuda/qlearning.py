"""Agente de Q-learning (off-policy TD) VECTORIZADO para entornos CUDA.

Update:  Q[s, a] += alpha * (r_t + gamma * max_a' Q[s_{t+1}, a'] - Q[s, a]).

El último paso bootstrapa contra Q[CC] (fila de ceros), cerrando el episodio.
La recompensa r_t se reconstruye con per_step_rewards (0 en todos los pasos
salvo el terminal, donde llega el retorno G del episodio).

Al igual que MC, el update usa index_put_(accumulate=True) y divide por el
conteo: n_envs episodios comparten los mismos CC estados, así que los pares
(s, a) se repiten en el batch y la indexación avanzada (read-modify-write)
descartaría todos los targets menos uno. La media de los deltas es correcta
porque Q_old(s, a) es el mismo para todos los duplicados del batch.
"""

import torch

from .base import TabularAgentCUDA


class QLearningAgentCUDA(TabularAgentCUDA):
    def update(self, states, actions, returns):
        T, n = states.shape
        rewards = self.per_step_rewards(actions, returns)

        next_states = torch.cat([
            states[1:],
            torch.full((1, n), self.Q.shape[0] - 1,
                       dtype=torch.int64, device=self.device),
        ])
        max_next = self.Q[next_states].max(dim=2).values  # (T, n)
        target = rewards + self.gamma * max_next

        s_flat = states.reshape(-1)
        a_flat = actions.reshape(-1)
        delta = target.reshape(-1) - self.Q[s_flat, a_flat]

        sums = torch.zeros_like(self.Q)
        counts = torch.zeros_like(self.Q)
        sums.index_put_((s_flat, a_flat), delta, accumulate=True)
        counts.index_put_((s_flat, a_flat), torch.ones_like(delta),
                          accumulate=True)

        visited = counts > 0
        self.Q += self.alpha * (sums / counts.clamp(min=1.0)) * visited
        self.N += counts.to(self.N.dtype)
