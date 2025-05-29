import numpy as np
import multiprocessing as mp
from collections import deque
from typing import Callable, Tuple, List, Optional, Dict, Any, Union

from .base import Optimizer


class CrestPorcupineOptimizer(Optimizer):
    """
    Crested Porcupine Optimizer (CPO) for optimization problems.
    
    This class implements the CPO algorithm, a nature-inspired metaheuristic that mimics
    the defensive behaviors of crested porcupines (sight, sound, odor, physical attack) to
    balance exploration and exploitation, with cyclic population reduction for convergence.
    
    Parameters
    ----------
    dimensions : int
        Number of dimensions in the search space.
    bounds : tuple of numpy.ndarray
        A tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        (dimensions,).
    pop_size : int, optional
        Number of search agents (porcupines) in the initial population (default: 30).
    min_pop_size : int, optional
        Minimum population size during reduction cycles (default: pop_size // 2).
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    cycles : int, optional
        Number of cycles for population reduction (default: 2).
    alpha : float, optional
        Convergence rate for fourth defense mechanism (default: 0.2).
    tf : float, optional
        Tradeoff threshold between third and fourth mechanisms (default: 0.8).
    ftol : float, optional
        Relative error in objective_func(best_pos) acceptable for
        convergence (default: -np.inf).
    ftol_iter : int, optional
        Number of iterations over which the relative error in
        objective_func(best_pos) is acceptable for convergence (default: 1).
    
    Attributes
    ----------
    min_pop_size : int
        Minimum population size during reduction cycles.
    cycles : int
        Number of cycles for population reduction.
    alpha : float
        Convergence rate for fourth defense mechanism.
    tf : float
        Tradeoff threshold between third and fourth mechanisms.
    positions : ndarray
        Current positions of all porcupines in the population.
    fitness : ndarray
        Current fitness values of all porcupines.
    personal_best_pos : ndarray
        Personal best positions found by each porcupine.
    """
    
    def __init__(
        self,
        dimensions: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 30,
        min_pop_size: Optional[int] = None,
        max_iter: int = 100,
        cycles: int = 2,
        alpha: float = 0.2,
        tf: float = 0.8,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
    ):
        # Initialize base class
        super(CrestPorcupineOptimizer, self).__init__(
            dimensions=dimensions,
            bounds=bounds,
            pop_size=pop_size,
            max_iter=max_iter,
            options={"alpha": alpha, "tf": tf, "cycles": cycles},
            ftol=ftol,
            ftol_iter=ftol_iter,
        )
        
        # Set CPO-specific parameters
        self.min_pop_size = min_pop_size if min_pop_size is not None else max(10, pop_size // 2)
        self.cycles = cycles
        self.alpha = alpha
        self.tf = tf
        
        # Initialize population attributes
        self.positions = None
        self.fitness = None
        self.personal_best_pos = None
        
        # Initialize the population
        self._init_population()
    
    def _init_population(self):
        """Initialize the population of porcupines"""
        lb, ub = self.bounds
        
        # Initialize positions within bounds
        self.positions = np.zeros((self.pop_size, self.dimensions))
        for i in range(self.dimensions):
            self.positions[:, i] = np.random.uniform(lb[i], ub[i], self.pop_size)
        
        # Initialize fitness and personal best
        self.fitness = np.full(self.pop_size, np.inf)
        self.personal_best_pos = self.positions.copy()
    
    def _evaluate_population(self, objective_func, f_ieqcons=None, pool=None, **kwargs):
        """Evaluate the fitness of all porcupines in the population
        
        Parameters
        ----------
        objective_func : callable
            The objective function to minimize
        f_ieqcons : callable, optional
            Constraint function
        pool : multiprocessing.Pool, optional
            Pool for parallel evaluation
        **kwargs
            Additional arguments for the objective function
        """
        if pool is None:
            # Serial evaluation
            for i in range(len(self.positions)):
                if f_ieqcons is None or np.all(np.array(f_ieqcons(self.positions[i])) >= 0):
                    self.fitness[i] = objective_func(self.positions[i], **kwargs)
        else:
            # Parallel evaluation
            results = []
            for i in range(len(self.positions)):
                if f_ieqcons is None or np.all(np.array(f_ieqcons(self.positions[i])) >= 0):
                    results.append(pool.apply_async(objective_func, (self.positions[i],), kwargs))
                else:
                    results.append(None)
            
            # Collect results
            for i, result in enumerate(results):
                if result is not None:
                    self.fitness[i] = result.get()
    
    def _update_best_solution(self):
        """Update the best solution found so far"""
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_cost:
            self.best_pos = self.positions[best_idx].copy()
            self.best_cost = self.fitness[best_idx]
    
    def _calculate_current_pop_size(self, iteration):
        """Calculate the current population size based on cyclic reduction
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        
        Returns
        -------
        int
            Current population size
        """
        cycle_progress = (iteration % (self.max_iter // self.cycles)) / (self.max_iter // self.cycles)
        current_pop_size = max(1, int(self.min_pop_size + 
                                    (self.pop_size - self.min_pop_size) * 
                                    (1 - cycle_progress)))
        
        # Ensure population size does not exceed current array size
        return min(current_pop_size, len(self.positions))
    
    def _apply_first_defense(self, i, current_pop_size):
        """Apply first defense mechanism (sight)
        
        Parameters
        ----------
        i : int
            Index of the current porcupine
        current_pop_size : int
            Current population size
        """
        # Midpoint between current and random position
        y = (self.positions[i] + self.positions[np.random.randint(current_pop_size)]) / 2
        # Update with random perturbation toward global best
        self.positions[i] += np.random.randn() * np.abs(2 * np.random.random() * self.best_pos - y)
    
    def _apply_second_defense(self, i, current_pop_size, u1):
        """Apply second defense mechanism (sound)
        
        Parameters
        ----------
        i : int
            Index of the current porcupine
        current_pop_size : int
            Current population size
        u1 : ndarray
            Binary mask for position updates
        """
        # Midpoint with another random position
        y = (self.positions[i] + self.positions[np.random.randint(current_pop_size)]) / 2
        # Update using binary mask and random differences
        rand_diff = (self.positions[np.random.randint(current_pop_size)] - 
                    self.positions[np.random.randint(current_pop_size)])
        self.positions[i] = u1 * self.positions[i] + (1 - u1) * (y + np.random.random() * rand_diff)
    
    def _apply_third_defense(self, i, current_pop_size, u1, yt, s):
        """Apply third defense mechanism (odor)
        
        Parameters
        ----------
        i : int
            Index of the current porcupine
        current_pop_size : int
            Current population size
        u1 : ndarray
            Binary mask for position updates
        yt : float
            Time-dependent factor
        s : ndarray
            Random direction vector
        """
        # Exponential scaling based on relative fitness
        fitness_sum = np.sum(self.fitness[:current_pop_size])
        if np.isfinite(self.fitness[i]) and fitness_sum > 0:
            st = np.exp(self.fitness[i] / (fitness_sum + np.finfo(float).eps))
        else:
            st = 1.0  # Default scaling if fitness is invalid
        
        s = s * yt * st
        # Update with random position and scaled difference
        rand_diff = (self.positions[np.random.randint(current_pop_size)] - 
                    self.positions[np.random.randint(current_pop_size)])
        self.positions[i] = ((1 - u1) * self.positions[i] + 
                            u1 * (self.positions[np.random.randint(current_pop_size)] + 
                                st * rand_diff - s))
    
    def _apply_fourth_defense(self, i, current_pop_size, u2, yt, s):
        """Apply fourth defense mechanism (physical attack)
        
        Parameters
        ----------
        i : int
            Index of the current porcupine
        current_pop_size : int
            Current population size
        u2 : ndarray
            Random direction vector
        yt : float
            Time-dependent factor
        s : ndarray
            Random direction vector
        """
        # Exponential movement factor
        fitness_sum = np.sum(self.fitness[:current_pop_size])
        if np.isfinite(self.fitness[i]) and fitness_sum > 0:
            mt = np.exp(self.fitness[i] / (fitness_sum + np.finfo(float).eps))
        else:
            mt = 1.0  # Default scaling if fitness is invalid
        
        vt = self.positions[i]
        vtp = self.positions[np.random.randint(current_pop_size)]
        ft = np.random.random(self.dimensions) * (mt * (-vt + vtp))
        s = s * yt * ft
        
        # Update toward global best with convergence rate
        r2 = np.random.random()
        self.positions[i] = (self.best_pos + 
                            (self.alpha * (1 - r2) + r2) * 
                            (u2 * self.best_pos - self.positions[i]) - s)
    
    def optimize(
        self, 
        objective_func: Callable, 
        f_ieqcons: Optional[Callable] = None,
        n_processes: Optional[int] = None, 
        verbose: bool = False, 
        **kwargs
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Optimize the objective function using Crested Porcupine Optimizer
        
        Parameters
        ----------
        objective_func : callable
            The objective function to be minimized
        f_ieqcons : callable, optional
            Constraint function returning a 1D array of inequality constraints (g(x) >= 0)
        n_processes : int, optional
            Number of processes to use for parallel evaluation
        verbose : bool, optional
            Whether to display progress information
        **kwargs
            Additional arguments to pass to the objective function
            
        Returns
        -------
        tuple
            A tuple containing (best_position, best_cost, cost_history)
        """
        # Reset the optimizer
        self.reset()
        self._init_population()
        
        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)
        
        # Initial evaluation
        self._evaluate_population(objective_func, f_ieqcons, pool, **kwargs)
        self._update_best_solution()
        
        # Initialize convergence tracking
        ftol_history = deque(maxlen=self.ftol_iter)
        cost_history = np.zeros(self.max_iter)
        
        # Main optimization loop
        for t in range(self.max_iter):
            # Calculate current population size
            current_pop_size = self._calculate_current_pop_size(t)
            
            # Update each porcupine
            for i in range(current_pop_size):
                # Random binary mask for position updates
                u1 = np.random.random(self.dimensions) > np.random.random()
                
                if np.random.random() < np.random.random():  # Exploration phase
                    if np.random.random() < np.random.random():  # First defense mechanism (sight)
                        self._apply_first_defense(i, current_pop_size)
                    else:  # Second defense mechanism (sound)
                        self._apply_second_defense(i, current_pop_size, u1)
                else:  # Exploitation phase
                    # Time-dependent factor for scaling updates
                    yt = 2 * np.random.random() * (1 - t / self.max_iter) ** (t / self.max_iter)
                    # Random direction vector
                    u2 = (np.random.random(self.dimensions) < 0.5) * 2 - 1
                    s = np.random.random() * u2
                    
                    if np.random.random() < self.tf:  # Third defense mechanism (odor)
                        self._apply_third_defense(i, current_pop_size, u1, yt, s)
                    else:  # Fourth defense mechanism (physical attack)
                        self._apply_fourth_defense(i, current_pop_size, u2, yt, s)
                
                # Clip positions to bounds
                lb, ub = self.bounds
                self.positions[i] = np.clip(self.positions[i], lb, ub)
            
            # Evaluate new fitness
            prev_best_cost = self.best_cost
            self._evaluate_population(objective_func, f_ieqcons, pool, **kwargs)
            
            # Update personal and global bests
            for i in range(current_pop_size):
                if self.fitness[i] < objective_func(self.personal_best_pos[i], **kwargs):
                    self.personal_best_pos[i] = self.positions[i].copy()
            
            self._update_best_solution()
            
            # Trim population if size reduced
            if current_pop_size < len(self.positions):
                self.positions = self.positions[:current_pop_size].copy()
                self.personal_best_pos = self.personal_best_pos[:current_pop_size].copy()
                self.fitness = self.fitness[:current_pop_size].copy()
            
            # Update history
            cost_history[t] = self.best_cost
            hist = self.ToHistory(
                best_cost=self.best_cost,
                mean_cost=np.mean(self.fitness),
                position=self.positions.copy(),
                population_size=current_pop_size
            )
            self._populate_history(hist)
            
            # Check convergence
            relative_measure = self.ftol * (1 + np.abs(prev_best_cost))
            delta = np.abs(self.best_cost - prev_best_cost) < relative_measure
            if t < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            
            # Print progress if verbose
            if verbose:
                print(f"Iteration {t+1}/{self.max_iter}: Best Cost = {self.best_cost:.6f}, "
                      f"Population Size = {current_pop_size}")
        
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()
        
        return self.best_pos, self.best_cost, np.array(self.cost_history)
