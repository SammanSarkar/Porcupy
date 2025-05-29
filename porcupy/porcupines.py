import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any, List


class PorcupinePopulation:
    """
    Class representing a population of porcupines for the Crested Porcupine Optimizer.
    
    This class handles the population structure, initialization, and provides methods
    for population management and dynamics.
    
    Parameters
    ----------
    pop_size : int
        Number of porcupines in the population.
    dimensions : int
        Number of dimensions in the search space.
    bounds : tuple of numpy.ndarray
        A tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        (dimensions,).
    init_pos : numpy.ndarray, optional
        Initial positions for the porcupines. If None, random positions are generated.
    
    Attributes
    ----------
    pop_size : int
        Current number of porcupines in the population.
    dimensions : int
        Number of dimensions in the search space.
    bounds : tuple of numpy.ndarray
        Bounds for the search space.
    positions : numpy.ndarray
        Current positions of all porcupines, shape (pop_size, dimensions).
    fitness : numpy.ndarray
        Current fitness values of all porcupines, shape (pop_size,).
    personal_best_pos : numpy.ndarray
        Personal best positions found by each porcupine, shape (pop_size, dimensions).
    personal_best_fitness : numpy.ndarray
        Personal best fitness values for each porcupine, shape (pop_size,).
    best_pos : numpy.ndarray
        Global best position found so far, shape (dimensions,).
    best_fitness : float
        Global best fitness value found so far.
    """
    
    def __init__(
        self,
        pop_size: int,
        dimensions: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        init_pos: Optional[np.ndarray] = None
    ):
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.bounds = bounds
        
        # Initialize positions
        if init_pos is not None:
            if init_pos.shape != (pop_size, dimensions):
                raise ValueError(f"init_pos must have shape ({pop_size}, {dimensions})")
            self.positions = init_pos.copy()
        else:
            self.positions = self._initialize_positions()
        
        # Initialize fitness and best values
        self.fitness = np.full(pop_size, np.inf)
        self.personal_best_pos = self.positions.copy()
        self.personal_best_fitness = np.full(pop_size, np.inf)
        self.best_pos = None
        self.best_fitness = np.inf
    
    def _initialize_positions(self) -> np.ndarray:
        """Initialize positions within bounds
        
        Returns
        -------
        numpy.ndarray
            Initialized positions with shape (pop_size, dimensions)
        """
        lb, ub = self.bounds
        positions = np.zeros((self.pop_size, self.dimensions))
        
        for i in range(self.dimensions):
            positions[:, i] = np.random.uniform(lb[i], ub[i], self.pop_size)
        
        return positions
    
    def evaluate(self, objective_func: Callable, **kwargs) -> None:
        """Evaluate the fitness of all porcupines
        
        Parameters
        ----------
        objective_func : callable
            The objective function to minimize
        **kwargs
            Additional arguments for the objective function
        """
        for i in range(self.pop_size):
            self.fitness[i] = objective_func(self.positions[i], **kwargs)
            
            # Update personal best
            if self.fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_pos[i] = self.positions[i].copy()
                self.personal_best_fitness[i] = self.fitness[i]
                
                # Update global best
                if self.fitness[i] < self.best_fitness:
                    self.best_pos = self.positions[i].copy()
                    self.best_fitness = self.fitness[i]
    
    def resize(self, new_size: int) -> None:
        """Resize the population to a new size
        
        Parameters
        ----------
        new_size : int
            New population size
        """
        if new_size <= 0:
            raise ValueError("Population size must be positive")
        
        if new_size < self.pop_size:
            # Shrink population - keep the best individuals
            indices = np.argsort(self.fitness)[:new_size]
            self.positions = self.positions[indices].copy()
            self.fitness = self.fitness[indices].copy()
            self.personal_best_pos = self.personal_best_pos[indices].copy()
            self.personal_best_fitness = self.personal_best_fitness[indices].copy()
        elif new_size > self.pop_size:
            # Expand population - add new random individuals
            lb, ub = self.bounds
            new_positions = np.zeros((new_size - self.pop_size, self.dimensions))
            
            for i in range(self.dimensions):
                new_positions[:, i] = np.random.uniform(lb[i], ub[i], new_size - self.pop_size)
            
            self.positions = np.vstack([self.positions, new_positions])
            self.fitness = np.append(self.fitness, np.full(new_size - self.pop_size, np.inf))
            self.personal_best_pos = np.vstack([self.personal_best_pos, new_positions.copy()])
            self.personal_best_fitness = np.append(
                self.personal_best_fitness, 
                np.full(new_size - self.pop_size, np.inf)
            )
        
        self.pop_size = new_size
    
    def apply_bounds(self) -> None:
        """Apply boundary constraints to all positions"""
        lb, ub = self.bounds
        self.positions = np.clip(self.positions, lb, ub)


class DefenseMechanisms:
    """
    Class implementing the four defense mechanisms of the Crested Porcupine Optimizer.
    
    This class provides methods for the four defense mechanisms: sight, sound, odor,
    and physical attack, which are used to update the positions of porcupines.
    
    Parameters
    ----------
    alpha : float
        Convergence rate for the fourth defense mechanism.
    tf : float
        Tradeoff threshold between third and fourth mechanisms.
    
    Attributes
    ----------
    alpha : float
        Convergence rate for the fourth defense mechanism.
    tf : float
        Tradeoff threshold between third and fourth mechanisms.
    """
    
    def __init__(self, alpha: float = 0.2, tf: float = 0.8):
        self.alpha = alpha
        self.tf = tf
    
    def sight_defense(
        self, 
        position: np.ndarray, 
        other_position: np.ndarray, 
        best_position: np.ndarray
    ) -> np.ndarray:
        """Apply the first defense mechanism (sight)
        
        Parameters
        ----------
        position : numpy.ndarray
            Current position of the porcupine
        other_position : numpy.ndarray
            Position of another random porcupine
        best_position : numpy.ndarray
            Global best position
            
        Returns
        -------
        numpy.ndarray
            Updated position
        """
        # Midpoint between current and random position
        y = (position + other_position) / 2
        # Update with random perturbation toward global best
        return position + np.random.randn() * np.abs(2 * np.random.random() * best_position - y)
    
    def sound_defense(
        self, 
        position: np.ndarray, 
        other_position: np.ndarray, 
        rand_diff: np.ndarray
    ) -> np.ndarray:
        """Apply the second defense mechanism (sound)
        
        Parameters
        ----------
        position : numpy.ndarray
            Current position of the porcupine
        other_position : numpy.ndarray
            Position of another random porcupine
        rand_diff : numpy.ndarray
            Random difference vector
            
        Returns
        -------
        numpy.ndarray
            Updated position
        """
        # Random binary mask for position updates
        u1 = np.random.random(len(position)) > np.random.random()
        # Midpoint with another random position
        y = (position + other_position) / 2
        # Update using binary mask and random differences
        return u1 * position + (1 - u1) * (y + np.random.random() * rand_diff)
    
    def odor_defense(
        self, 
        position: np.ndarray, 
        other_position: np.ndarray, 
        rand_diff: np.ndarray,
        fitness: float,
        fitness_sum: float,
        t: int,
        max_iter: int
    ) -> np.ndarray:
        """Apply the third defense mechanism (odor)
        
        Parameters
        ----------
        position : numpy.ndarray
            Current position of the porcupine
        other_position : numpy.ndarray
            Position of another random porcupine
        rand_diff : numpy.ndarray
            Random difference vector
        fitness : float
            Current fitness of the porcupine
        fitness_sum : float
            Sum of all fitness values in the population
        t : int
            Current iteration
        max_iter : int
            Maximum number of iterations
            
        Returns
        -------
        numpy.ndarray
            Updated position
        """
        # Random binary mask for position updates
        u1 = np.random.random(len(position)) > np.random.random()
        
        # Time-dependent factor for scaling updates
        yt = 2 * np.random.random() * (1 - t / max_iter) ** (t / max_iter)
        
        # Random direction vector
        u2 = (np.random.random(len(position)) < 0.5) * 2 - 1
        s = np.random.random() * u2
        
        # Exponential scaling based on relative fitness
        if np.isfinite(fitness) and fitness_sum > 0:
            st = np.exp(fitness / (fitness_sum + np.finfo(float).eps))
        else:
            st = 1.0  # Default scaling if fitness is invalid
        
        s = s * yt * st
        
        # Update with random position and scaled difference
        return (1 - u1) * position + u1 * (other_position + st * rand_diff - s)
    
    def physical_attack(
        self, 
        position: np.ndarray, 
        other_position: np.ndarray, 
        best_position: np.ndarray,
        fitness: float,
        fitness_sum: float,
        t: int,
        max_iter: int
    ) -> np.ndarray:
        """Apply the fourth defense mechanism (physical attack)
        
        Parameters
        ----------
        position : numpy.ndarray
            Current position of the porcupine
        other_position : numpy.ndarray
            Position of another random porcupine
        best_position : numpy.ndarray
            Global best position
        fitness : float
            Current fitness of the porcupine
        fitness_sum : float
            Sum of all fitness values in the population
        t : int
            Current iteration
        max_iter : int
            Maximum number of iterations
            
        Returns
        -------
        numpy.ndarray
            Updated position
        """
        # Time-dependent factor for scaling updates
        yt = 2 * np.random.random() * (1 - t / max_iter) ** (t / max_iter)
        
        # Random direction vector
        u2 = (np.random.random(len(position)) < 0.5) * 2 - 1
        s = np.random.random() * u2
        
        # Exponential movement factor
        if np.isfinite(fitness) and fitness_sum > 0:
            mt = np.exp(fitness / (fitness_sum + np.finfo(float).eps))
        else:
            mt = 1.0  # Default scaling if fitness is invalid
        
        ft = np.random.random(len(position)) * (mt * (-position + other_position))
        s = s * yt * ft
        
        # Update toward global best with convergence rate
        r2 = np.random.random()
        return best_position + (self.alpha * (1 - r2) + r2) * (u2 * best_position - position) - s


class PopulationManager:
    """
    Class for managing the population dynamics in the Crested Porcupine Optimizer.
    
    This class handles the cyclic population reduction strategy and other
    population management tasks.
    
    Parameters
    ----------
    initial_pop_size : int
        Initial population size.
    min_pop_size : int
        Minimum population size during reduction.
    max_iter : int
        Maximum number of iterations.
    cycles : int
        Number of cycles for population reduction.
    
    Attributes
    ----------
    initial_pop_size : int
        Initial population size.
    min_pop_size : int
        Minimum population size during reduction.
    max_iter : int
        Maximum number of iterations.
    cycles : int
        Number of cycles for population reduction.
    """
    
    def __init__(
        self,
        initial_pop_size: int,
        min_pop_size: int,
        max_iter: int,
        cycles: int = 2
    ):
        self.initial_pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.max_iter = max_iter
        self.cycles = cycles
    
    def calculate_pop_size(self, iteration: int) -> int:
        """Calculate the population size for the current iteration
        
        Parameters
        ----------
        iteration : int
            Current iteration number
            
        Returns
        -------
        int
            Population size for the current iteration
        """
        # Calculate cycle progress
        cycle_length = self.max_iter // self.cycles
        cycle_progress = (iteration % cycle_length) / cycle_length
        
        # Calculate population size using linear reduction within each cycle
        pop_size = max(
            1, 
            int(self.min_pop_size + (self.initial_pop_size - self.min_pop_size) * 
                (1 - cycle_progress))
        )
        
        return pop_size
