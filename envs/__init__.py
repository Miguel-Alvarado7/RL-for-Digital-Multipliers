"""
Entorno de Aprendizaje por Refuerzo para Multiplicadores Binarios
"""

import importlib
import sys
from types import ModuleType

from .base import BinaryMathEnv


__all__ = [
    'BinaryMathEnv',
    'BinaryMathEnvSecuencial',
    'BinaryMathEnvCUDA',
    'BinaryMathEnvCUDAOptimized',
]

# Lazy imports: BinaryMathEnv solo necesita numpy+gymnasium. Las clases con
# dependencias pesadas (pygame, torch) se cargan bajo demanda, evitando
# instalarlas para experimentos en CPU. Se usa __getattr__ a nivel de clase de
# módulo (compatible con Python 3.6, donde el __getattr__ de módulo de PEP 562
# aún no existe).
_LAZY_MODULES = {
    'BinaryMathEnvSecuencial': '.pygame_env',
    'BinaryMathEnvCUDA': '.cuda',
    'BinaryMathEnvCUDAOptimized': '.cuda_optimized',
}


class _EnvModule(ModuleType):
    def __getattr__(self, name):
        module_name = _LAZY_MODULES.get(name)
        if module_name is not None:
            module = importlib.import_module(module_name, __name__)
            value = getattr(module, name)
            setattr(self, name, value)
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _EnvModule
