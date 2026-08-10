"""Agente de Q-learning (off-policy TD) para BinaryMathEnv.

Update:  Q[s, a] += alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a]).

Estado terminal (s_next = CC, a_next = None): el valor de Q[CC] es 0, por lo
que el bootstrap en el último paso reduce a r.
"""

from .base import TabularAgentCPU


class QLearningAgentCPU(TabularAgentCPU):
    def update(self, trajectory):
        for s, a, r, s_next, a_next in trajectory:
            max_next = 0.0 if a_next is None else float(self.Q[s_next].max())
            self.Q[s, a] += self.alpha * (r + self.gamma * max_next - self.Q[s, a])
            self.N[s, a] += 1
