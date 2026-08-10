"""
Entorno de Aprendizaje por Refuerzo para Multiplicadores Binarios
"""

import importlib

from .base import BinaryMathEnv


__all__ = [
    'BinaryMathEnv',
    'BinaryMathEnvSecuencial',
    'BinaryMathEnvCUDA',
    'BinaryMathEnvCUDAOptimized',
]

# Lazy imports (PEP 562): BinaryMathEnv only needs numpy+gymnasium.
# Las clases con dependencias pesadas (pygame, torch) se importan bajo demanda,
# evitando instalarlas para experimentos en CPU.
_LAZY_MODULES = {
    'BinaryMathEnvSecuencial': '.pygame_env',
    'BinaryMathEnvCUDA': '.cuda',
    'BinaryMathEnvCUDAOptimized': '.cuda_optimized',
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
