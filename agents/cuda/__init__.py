"""Agentes tabulares VECTORIZADOS para entornos CUDA."""

from .montecarlo import MonteCarloAgentCUDA
from .qlearning import QLearningAgentCUDA
from .sarsa import SarsaAgentCUDA

__all__ = ["MonteCarloAgentCUDA", "QLearningAgentCUDA", "SarsaAgentCUDA"]
