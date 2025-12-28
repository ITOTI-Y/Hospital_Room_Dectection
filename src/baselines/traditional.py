"""
Traditional optimization baselines for layout optimization.

These serve as comparison baselines for the RL approach in terms of:
1. Sample efficiency (steps to reach good solution)
2. Final solution quality
3. Computation time
"""

import numpy as np
from typing import Callable, List, Tuple
from loguru import logger


class SimulatedAnnealing:
    """
    Simulated Annealing baseline algorithm.

    Classic metaheuristic that allows uphill moves with decreasing probability,
    enabling escape from local optima.
    """

    def __init__(
        self,
        cost_fn: Callable[[List[int]], float],
        n_items: int,
        T_init: float = 1000.0,
        T_min: float = 1.0,
        alpha: float = 0.995,
        seed: int | None = None,
    ):
        """
        Args:
            cost_fn: Function that takes a permutation and returns cost
            n_items: Number of items to arrange
            T_init: Initial temperature
            T_min: Minimum temperature (stopping criterion)
            alpha: Cooling rate (T = T * alpha each step)
            seed: Random seed for reproducibility
        """
        self.cost_fn = cost_fn
        self.n_items = n_items
        self.T_init = T_init
        self.T_min = T_min
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def solve(self, max_steps: int = 10000) -> Tuple[List[int], float, List[float]]:
        """
        Run simulated annealing optimization.

        Returns:
            best: Best permutation found
            best_cost: Cost of best permutation
            history: List of best costs at each step
        """
        # Initialize with identity permutation
        current = list(range(self.n_items))
        current_cost = self.cost_fn(current)
        best, best_cost = current.copy(), current_cost

        T = self.T_init
        history = [current_cost]

        for step in range(max_steps):
            # Random swap
            i, j = self.rng.choice(self.n_items, 2, replace=False)
            neighbor = current.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_cost = self.cost_fn(neighbor)

            # Metropolis criterion
            delta = neighbor_cost - current_cost
            if delta < 0 or self.rng.random() < np.exp(-delta / max(T, 1e-10)):
                current, current_cost = neighbor, neighbor_cost
                if current_cost < best_cost:
                    best, best_cost = current.copy(), current_cost

            # Cooling
            T = max(self.T_min, T * self.alpha)
            history.append(best_cost)

            # Early stopping if temperature is too low
            if T <= self.T_min:
                # Continue with greedy moves only
                pass

        logger.debug(f"SA completed: best_cost={best_cost:.2f}, steps={len(history)}")
        return best, best_cost, history


class GreedySwap:
    """
    Greedy Swap baseline algorithm.

    At each step, evaluates all possible swaps and takes the best one.
    Terminates when no improving swap exists (local optimum).
    """

    def __init__(
        self,
        cost_fn: Callable[[List[int]], float],
        n_items: int,
        seed: int | None = None,
    ):
        """
        Args:
            cost_fn: Function that takes a permutation and returns cost
            n_items: Number of items to arrange
            seed: Random seed for tie-breaking
        """
        self.cost_fn = cost_fn
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def solve(self, max_steps: int = 1000) -> Tuple[List[int], float, List[float]]:
        """
        Run greedy swap optimization.

        Returns:
            best: Best permutation found (local optimum)
            best_cost: Cost of best permutation
            history: List of costs after each improving swap
        """
        current = list(range(self.n_items))
        current_cost = self.cost_fn(current)
        history = [current_cost]

        for step in range(max_steps):
            best_swap, best_delta = None, 0

            # Evaluate all possible swaps
            for i in range(self.n_items):
                for j in range(i + 1, self.n_items):
                    neighbor = current.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbor_cost = self.cost_fn(neighbor)
                    delta = neighbor_cost - current_cost

                    if delta < best_delta:
                        best_swap, best_delta = (i, j), delta

            if best_swap is None:
                # Local optimum reached
                logger.debug(f"Greedy reached local optimum at step {step}")
                break

            # Apply best swap
            i, j = best_swap
            current[i], current[j] = current[j], current[i]
            current_cost += best_delta
            history.append(current_cost)

        logger.debug(f"Greedy completed: cost={current_cost:.2f}, steps={len(history)}")
        return current, current_cost, history


class RandomSearch:
    """
    Random Search baseline.

    Simply tries random swaps and keeps track of the best solution found.
    Useful as a lower bound for comparison.
    """

    def __init__(
        self,
        cost_fn: Callable[[List[int]], float],
        n_items: int,
        seed: int | None = None,
    ):
        self.cost_fn = cost_fn
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def solve(self, max_steps: int = 10000) -> Tuple[List[int], float, List[float]]:
        """
        Run random search.

        Returns:
            best: Best permutation found
            best_cost: Cost of best permutation
            history: List of best costs at each step
        """
        current = list(range(self.n_items))
        best_cost = self.cost_fn(current)
        best = current.copy()
        history = [best_cost]

        for step in range(max_steps):
            # Random swap
            i, j = self.rng.choice(self.n_items, 2, replace=False)
            current[i], current[j] = current[j], current[i]

            current_cost = self.cost_fn(current)
            if current_cost < best_cost:
                best_cost = current_cost
                best = current.copy()

            history.append(best_cost)

        logger.debug(f"Random completed: best_cost={best_cost:.2f}, steps={len(history)}")
        return best, best_cost, history


class TabuSearch:
    """
    Tabu Search baseline.

    Maintains a tabu list of recently visited swaps to avoid cycling.
    """

    def __init__(
        self,
        cost_fn: Callable[[List[int]], float],
        n_items: int,
        tabu_tenure: int = 10,
        seed: int | None = None,
    ):
        self.cost_fn = cost_fn
        self.n_items = n_items
        self.tabu_tenure = tabu_tenure
        self.rng = np.random.default_rng(seed)

    def solve(self, max_steps: int = 1000) -> Tuple[List[int], float, List[float]]:
        """
        Run tabu search.
        """
        current = list(range(self.n_items))
        current_cost = self.cost_fn(current)
        best, best_cost = current.copy(), current_cost
        history = [current_cost]

        # Tabu list: maps (i, j) -> step when it becomes non-tabu
        tabu_list = {}

        for step in range(max_steps):
            best_move, best_move_cost = None, float('inf')

            # Find best non-tabu move (or aspiration)
            for i in range(self.n_items):
                for j in range(i + 1, self.n_items):
                    # Check if move is tabu
                    is_tabu = tabu_list.get((i, j), 0) > step

                    neighbor = current.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbor_cost = self.cost_fn(neighbor)

                    # Aspiration criterion: accept if better than best known
                    if neighbor_cost < best_cost:
                        is_tabu = False

                    if not is_tabu and neighbor_cost < best_move_cost:
                        best_move = (i, j)
                        best_move_cost = neighbor_cost

            if best_move is None:
                break

            # Apply move
            i, j = best_move
            current[i], current[j] = current[j], current[i]
            current_cost = best_move_cost

            # Add to tabu list
            tabu_list[(i, j)] = step + self.tabu_tenure
            tabu_list[(j, i)] = step + self.tabu_tenure

            if current_cost < best_cost:
                best, best_cost = current.copy(), current_cost

            history.append(best_cost)

        logger.debug(f"Tabu completed: best_cost={best_cost:.2f}, steps={len(history)}")
        return best, best_cost, history
