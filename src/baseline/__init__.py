"""
Baseline Optimization Algorithms for Hospital Layout Optimization

This module provides traditional optimization algorithms as baselines
for comparison with reinforcement learning approaches:

- Genetic Algorithm (GA): Population-based evolutionary optimization
- Simulated Annealing (SA): Temperature-based probabilistic search

Both algorithms optimize the department-to-slot assignment to minimize
total patient travel cost (Quadratic Assignment Problem).
"""

from src.baseline.base import BaseOptimizer, OptimizationResult
from src.baseline.genetic import GAConfig, GeneticAlgorithm
from src.baseline.runner import BaselineRunner
from src.baseline.simulated_annealing import SAConfig, SimulatedAnnealing

__all__ = [
    'BaseOptimizer',
    'OptimizationResult',
    'GAConfig',
    'GeneticAlgorithm',
    'SAConfig',
    'SimulatedAnnealing',
    'BaselineRunner',
]
