"""Agentes tabulares para BinaryMathEnv (CPU) y variantes CUDA.

Los agentes CPU solo dependen de numpy; los CUDA arrastran torch y se cargan
bajo demanda para no exigir PyTorch en experimentos de CPU. Se usa __getattr__
a nivel de clase de módulo (compatible con Python 3.6, donde el __getattr__ de
módulo de PEP 562 aún no existe).
"""

import importlib
import sys
from types import ModuleType

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


class _AgentsModule(ModuleType):
    def __getattr__(self, name):
        module_name = _LAZY_MODULES.get(name)
        if module_name is not None:
            module = importlib.import_module(module_name, __name__)
            value = getattr(module, name)
            setattr(self, name, value)
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _AgentsModule
