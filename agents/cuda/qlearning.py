"""Agente de Q-learning (off-policy TD) VECTORIZADO para entornos CUDA.

Update:  Q[s, a] += alpha * (r_t + gamma * max_a' Q[s_{t+1}, a'] - Q[s, a]).

Recompensa POR PASO con shaping potencial de la rejilla:
    r_t = R(partial_t) - R(partial_{t-1}),   R(partial_{-1}) = R(vacío),
donde R(g) es el reward del entorno para la rejilla g y partial_t es el
episodio con solo las primeras t+1 celdas colocadas (-1 = celda vacía, que
`evaluate_grids` ignora). La suma de los r_t telescopea al retorno terminal G.

POR QUÉ ES NECESARIO: con la reconstrucción antigua (r_t = 0 salvo el último
paso) el target TD era idéntico para TODAS las acciones de un estado no
terminal (r_t = 0, y el max sobre la fila del siguiente estado no depende de
la acción actual), así que las filas de Q convergían planas y la política
greedy elegía entre valores indistinguibles. Con el shaping, r_t depende de
la acción colocada en t y el target diferencia acciones en cada paso.

El último paso bootstrapa contra Q[CC] (fila de ceros), cerrando el episodio.

Al igual que MC, el update usa index_put_(accumulate=True) y divide por el
conteo: n_envs episodios comparten los mismos CC estados, así que los pares
(s, a) se repiten en el batch y la indexación avanzada (read-modify-write)
descartaría todos los targets menos uno. La media de los deltas es correcta
porque Q_old(s, a) es el mismo para todos los duplicados del batch.
"""

import torch

from .base import TabularAgentCUDA


class QLearningAgentCUDA(TabularAgentCUDA):
    def collect_batch(self, env, epsilon):
        """Como la base, pero además materializa la recompensa por paso.

        `collect_batch` de la base entrega solo el retorno terminal; aquí se
        evalúan las T rejillas parciales de una sola vez (un único call a
        `env.evaluate_grids`) y se guarda r_t = R(partial_t) - R(partial_{t-1})
        para que `update` use un target dependiente de la acción.
        """
        states, actions, returns = super().collect_batch(env, epsilon)
        self._shaped_rewards = self._shaped_step_rewards(env, actions)
        return states, actions, returns

    def _shaped_step_rewards(self, env, actions):
        """Recompensa por paso (T, n) con shaping potencial de la rejilla.

        partial_t[e, c] = actions[c, e] si c <= t, si no -1 (celda vacía).
        Todas las T rejillas parciales se evalúan en UNA llamada a
        `evaluate_grids` con (T·n, CC) filas.
        """
        n, T = env.n_envs, env.CC
        at = actions.T  # (n, T)

        cols = torch.arange(T, device=self.device)
        t_idx = torch.arange(T, device=self.device)[:, None, None]  # (T, 1, 1)
        col_idx = cols[None, None, :]                               # (1, 1, T)
        filled = at.unsqueeze(0).expand(T, n, T)                    # (T, n, T)
        partial = torch.where(
            col_idx <= t_idx,
            filled,
            torch.full((T, n, T), -1, dtype=filled.dtype,
                       device=self.device),
        )                                                           # (T, n, T)

        R_all = env.evaluate_grids(partial.reshape(T * n, T)).view(T, n)
        R_empty = env.evaluate_grids(
            torch.full((1, T), -1, dtype=filled.dtype, device=self.device)
        ).expand(n)                                                 # (n,)

        R_prev = torch.cat([R_empty.unsqueeze(0), R_all[:-1]], dim=0)
        return R_all - R_prev

    def update(self, states, actions, returns):
        rewards = getattr(self, "_shaped_rewards", None)
        if rewards is None:
            rewards = self.per_step_rewards(actions, returns)

        T, n = states.shape

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
