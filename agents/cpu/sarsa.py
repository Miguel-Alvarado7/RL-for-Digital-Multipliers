"""Agente de SARSA (on-policy TD) para BinaryMathEnv.

Update:  Q[s, a] += alpha * (r + gamma * Q[s', a'] - Q[s, a]), donde a' es la
acción realmente tomada en s' por la política de comportamiento.

Estado terminal (a_next = None): el valor de Q[CC] es 0.
"""

from .base import TabularAgentCPU


class SarsaAgentCPU(TabularAgentCPU):
    def update(self, trajectory):
        for s, a, r, s_next, a_next in trajectory:
            if a_next is None:
                next_val = 0.0
            else:
                next_val = float(self.Q[s_next, a_next])
            self.Q[s, a] += self.alpha * (r + self.gamma * next_val - self.Q[s, a])
            self.N[s, a] += 1
