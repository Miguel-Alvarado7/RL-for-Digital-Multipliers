"""Agente de SARSA (on-policy TD) VECTORIZADO para entornos CUDA.

Update:  Q[s, a] += alpha * (r_t + gamma * Q[s_{t+1}, a'] - Q[s, a]), con a'
la acción realmente tomada en el siguiente paso por la política ε-greedy.

En el último paso se bootstrapa contra Q[CC, ·] (fila de ceros). La
recompensa r_t se reconstruye con per_step_rewards (0 salvo el paso terminal).

El update usa index_put_(accumulate=True) y divide por el conteo por la misma
razón que MC y Q-learning: los pares (s, a) se repiten en el batch y la
indexación avanzada descartaría todos los deltas menos uno.
"""

import torch

from .base import TabularAgentCUDA


class SarsaAgentCUDA(TabularAgentCUDA):
    def update(self, states, actions, returns):
        T, n = states.shape
        rewards = self.per_step_rewards(actions, returns)

        next_states = torch.cat([
            states[1:],
            torch.full((1, n), self.Q.shape[0] - 1,
                       dtype=torch.int64, device=self.device),
        ])
        next_actions = torch.cat([
            actions[1:],
            torch.zeros(1, n, dtype=torch.int64, device=self.device),
        ])
        next_val = self.Q[next_states, next_actions]
        target = rewards + self.gamma * next_val

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
