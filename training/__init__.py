"""Helpers compartidos para los scripts de entrenamiento.

- baseline: política aleatoria de referencia (CPU y CUDA).
- plot: curva de aprendizaje.
- artifacts: TopK de circuitos + export Verilog + guardado de Q/returns.
"""

from . import artifacts
from . import baseline
from . import plot

__all__ = ["artifacts", "baseline", "plot"]
