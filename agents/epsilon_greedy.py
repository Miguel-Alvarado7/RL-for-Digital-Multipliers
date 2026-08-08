"""Política epsilon-greedy compartida por los agentes tabulares."""

import numpy as np


def choose_epsilon_greedy(q_row, epsilon, rng=None):
    """Selecciona una acción epsilon-greedy dada la fila de Q(s, ·).

    Args:
        q_row:  array (n_actions,) con los valores Q(s, ·).
        epsilon: probabilidad de explorar uniformemente (0 => greedy puro).
        rng:    np.random.Generator o None (usa np.random global).

    Returns:
        int: índice de acción.
    """
    if rng is None:
        rng = np.random
    if epsilon > 0 and rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    return int(np.argmax(q_row))
