"""Agente de Monte Carlo on-policy (first-visit) para BinaryMathEnv.

Update:  Q[s, a] += alpha * (G - Q[s, a])   para todo (s, a) visitado.

Puesto que las recompensas intermedias son 0 y la terminal llega solo al final,
el retorno G_t es el mismo para todos los pasos del episodio. El cursor crece
estrictamente, así que ningún par (s, a) se repite dentro de un episodio:
first-visit == every-visit.
"""

from .base import TabularAgentCPU


class MonteCarloAgent(TabularAgentCPU):
    def update(self, trajectory):
        return_ = self.episode_return(trajectory)
        for s, a, _r, _sn, _an in trajectory:
            self.Q[s, a] += self.alpha * (return_ - self.Q[s, a])
            self.N[s, a] += 1
