"""Agentes tabulares CPU para BinaryMathEnv."""

from .montecarlo import MonteCarloAgent
from .qlearning import QLearningAgentCPU
from .sarsa import SarsaAgentCPU

__all__ = ["MonteCarloAgent", "QLearningAgentCPU", "SarsaAgentCPU"]
