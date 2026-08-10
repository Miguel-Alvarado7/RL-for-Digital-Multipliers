"""Agente de Monte Carlo on-policy (first-visit) VECTORIZADO para entornos CUDA.

Recompensa CUDA en modo no-incremental: solo llega al terminar el episodio
([-10, 0] con 0 = circuito perfecto en CUDAOptimized). Como el cursor crece
estrictamente, ningún par (s, a) se repite dentro de un episodio: first-visit
== every-visit.

Update vectorizado con media explícita:  Q[s, a] += alpha*(mean(G) - Q[s, a]).
El promedio es obligatorio: `Q[states, actions] += ...` con indexación
avanzada hace read-modify-write y ante índices duplicados —inevitables, porque
n_envs episodios comparten los mismos CC estados— solo sobrevive la escritura
de UN entorno. Con n_envs=64 eso descartaba ~63 de cada 64 retornos.
"""

from .base import TabularAgentCUDA
import torch


class MonteCarloAgentCUDA(TabularAgentCUDA):
    def update(self, states, actions, returns):
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
