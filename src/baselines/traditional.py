"""Traditional optimization algorithms for layout optimization.

This module implements baseline algorithms for comparison with RL approaches:
- Simulated Annealing (SA)
- Greedy Local Search
"""

from typing import Callable

import numpy as np


class SimulatedAnnealing:
    """Simulated Annealing baseline algorithm.

    A meta-heuristic optimization algorithm that uses probabilistic acceptance
    of worse solutions to escape local optima.

    Args:
        cost_fn: Function that evaluates layout cost given a permutation
        n_items: Number of items to optimize
        T_init: Initial temperature
        T_min: Minimum temperature (stopping criterion)
        alpha: Temperature decay rate (0 < alpha < 1)
    """

    def __init__(
        self,
        cost_fn: Callable[[list[int]], float],
        n_items: int,
        T_init: float = 1000.0,
        T_min: float = 1.0,
        alpha: float = 0.995,
    ):
        self.cost_fn = cost_fn
        self.n_items = n_items
        self.T_init = T_init
        self.T_min = T_min
        self.alpha = alpha

    def solve(
        self, max_steps: int = 10000, initial_solution: list[int] | None = None
    ) -> tuple[list[int], float, list[float]]:
        """Run simulated annealing optimization.

        Args:
            max_steps: Maximum number of iterations
            initial_solution: Optional initial permutation (default: identity)

        Returns:
            Tuple of (best_solution, best_cost, cost_history)
        """
        # Initialize
        if initial_solution is not None:
            current = initial_solution.copy()
        else:
            current = list(range(self.n_items))

        current_cost = self.cost_fn(current)
        best, best_cost = current.copy(), current_cost

        T = self.T_init
        history = [current_cost]

        for step in range(max_steps):
            # Generate neighbor by random swap
            i, j = np.random.choice(self.n_items, 2, replace=False)
            neighbor = current.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_cost = self.cost_fn(neighbor)

            # Metropolis acceptance criterion
            delta = neighbor_cost - current_cost
            if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
                current, current_cost = neighbor, neighbor_cost
                if current_cost < best_cost:
                    best, best_cost = current.copy(), current_cost

            # Cool down temperature
            T = max(self.T_min, T * self.alpha)
            history.append(best_cost)

            # Early stopping if temperature too low
            if T <= self.T_min:
                break

        return best, best_cost, history


class GreedySwap:
    """Greedy local search baseline.

    Iteratively performs the swap that yields the largest cost reduction.
    Terminates when no improving swap can be found (local optimum).

    Args:
        cost_fn: Function that evaluates layout cost given a permutation
        n_items: Number of items to optimize
    """

    def __init__(self, cost_fn: Callable[[list[int]], float], n_items: int):
        self.cost_fn = cost_fn
        self.n_items = n_items

    def solve(
        self, max_steps: int = 1000, initial_solution: list[int] | None = None
    ) -> tuple[list[int], float, list[float]]:
        """Run greedy local search optimization.

        Args:
            max_steps: Maximum number of iterations
            initial_solution: Optional initial permutation (default: identity)

        Returns:
            Tuple of (best_solution, best_cost, cost_history)
        """
        # Initialize
        if initial_solution is not None:
            current = initial_solution.copy()
        else:
            current = list(range(self.n_items))

        current_cost = self.cost_fn(current)
        history = [current_cost]

        for _ in range(max_steps):
            best_swap = None
            best_delta = 0

            # Evaluate all possible swaps
            for i in range(self.n_items):
                for j in range(i + 1, self.n_items):
                    neighbor = current.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    delta = self.cost_fn(neighbor) - current_cost
                    if delta < best_delta:
                        best_swap = (i, j)
                        best_delta = delta

            # If no improving swap found, we're at local optimum
            if best_swap is None:
                break

            # Apply best swap
            i, j = best_swap
            current[i], current[j] = current[j], current[i]
            current_cost += best_delta
            history.append(current_cost)

        return current, current_cost, history


class RandomSearch:
    """Random search baseline for sanity check.

    Randomly samples permutations and keeps the best one found.

    Args:
        cost_fn: Function that evaluates layout cost given a permutation
        n_items: Number of items to optimize
    """

    def __init__(self, cost_fn: Callable[[list[int]], float], n_items: int):
        self.cost_fn = cost_fn
        self.n_items = n_items

    def solve(
        self, max_steps: int = 1000, initial_solution: list[int] | None = None
    ) -> tuple[list[int], float, list[float]]:
        """Run random search.

        Args:
            max_steps: Number of random permutations to evaluate
            initial_solution: Optional initial solution (used as baseline)

        Returns:
            Tuple of (best_solution, best_cost, cost_history)
        """
        if initial_solution is not None:
            best = initial_solution.copy()
        else:
            best = list(range(self.n_items))

        best_cost = self.cost_fn(best)
        history = [best_cost]

        for _ in range(max_steps):
            # Random permutation
            candidate = list(range(self.n_items))
            np.random.shuffle(candidate)
            candidate_cost = self.cost_fn(candidate)

            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost

            history.append(best_cost)

        return best, best_cost, history
