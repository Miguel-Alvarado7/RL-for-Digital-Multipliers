"""Utilidades compartidas por los agentes tabulares (CPU y CUDA)."""

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


def epsilon_at(epsilon_start, epsilon_end, index, max_index):
    """Decaimiento lineal de epsilon a lo largo del entrenamiento.

    `index` avanza 0..max_index-1; en max_index<=1 se usa epsilon_end.
    """
    if max_index <= 1:
        return epsilon_end
    frac = index / (max_index - 1)
    return epsilon_start + (epsilon_end - epsilon_start) * frac
