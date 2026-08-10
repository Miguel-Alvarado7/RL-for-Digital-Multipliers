"""Agentes tabulares para BinaryMathEnv (CPU) y variantes CUDA.

Los agentes CPU solo dependen de numpy; los CUDA arrastran torch y se cargan
bajo demanda (PEP 562) para no exigir PyTorch en experimentos de CPU.
"""

import importlib

from .cpu import MonteCarloAgent
from .cpu import QLearningAgentCPU
from .cpu import SarsaAgentCPU

__all__ = [
    "MonteCarloAgent",
    "QLearningAgentCPU",
    "SarsaAgentCPU",
    "MonteCarloAgentCUDA",
    "QLearningAgentCUDA",
    "SarsaAgentCUDA",
]

_LAZY_MODULES = {
    "MonteCarloAgentCUDA": ".cuda.montecarlo",
    "QLearningAgentCUDA": ".cuda.qlearning",
    "SarsaAgentCUDA": ".cuda.sarsa",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
