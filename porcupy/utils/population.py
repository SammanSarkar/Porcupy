import numpy as np
from typing import Tuple, Optional, List, Callable, Dict, Any


class PopulationCycle:
    """
    Class for managing cyclic population reduction in the Crested Porcupine Optimizer.
    
    This class implements the cyclic population reduction strategy, which is a key
    feature of the CPO algorithm. The population size varies cyclically throughout
    the optimization process to balance exploration and exploitation.
    
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
    reduction_strategy : str
        Strategy for population reduction ('linear', 'cosine', or 'exponential').
    
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
    reduction_strategy : str
        Strategy for population reduction.
    """
    
    def __init__(
        self,
        initial_pop_size: int,
        min_pop_size: int,
        max_iter: int,
        cycles: int = 2,
        reduction_strategy: str = 'linear'
    ):
        self.initial_pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.max_iter = max_iter
        self.cycles = cycles
        self.reduction_strategy = reduction_strategy
        
        # Validate inputs
        if initial_pop_size <= 0:
            raise ValueError("Initial population size must be positive")
        if min_pop_size <= 0:
            raise ValueError("Minimum population size must be positive")
        if min_pop_size > initial_pop_size:
            raise ValueError("Minimum population size cannot exceed initial population size")
        if max_iter <= 0:
            raise ValueError("Maximum iterations must be positive")
        if cycles <= 0:
            raise ValueError("Number of cycles must be positive")
        if reduction_strategy not in ['linear', 'cosine', 'exponential']:
            raise ValueError("Reduction strategy must be 'linear', 'cosine', or 'exponential'")
    
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
        
        # Calculate population size based on reduction strategy
        if self.reduction_strategy == 'linear':
            pop_size = self._linear_reduction(cycle_progress)
        elif self.reduction_strategy == 'cosine':
            pop_size = self._cosine_reduction(cycle_progress)
        else:  # exponential
            pop_size = self._exponential_reduction(cycle_progress)
        
        return max(1, int(pop_size))
    
    def _linear_reduction(self, cycle_progress: float) -> float:
        """Linear population reduction
        
        Parameters
        ----------
        cycle_progress : float
            Progress within the current cycle (0 to 1)
            
        Returns
        -------
        float
            Population size
        """
        return self.min_pop_size + (self.initial_pop_size - self.min_pop_size) * (1 - cycle_progress)
    
    def _cosine_reduction(self, cycle_progress: float) -> float:
        """Cosine population reduction
        
        Parameters
        ----------
        cycle_progress : float
            Progress within the current cycle (0 to 1)
            
        Returns
        -------
        float
            Population size
        """
        cosine_factor = (1 + np.cos(cycle_progress * np.pi)) / 2
        return self.min_pop_size + (self.initial_pop_size - self.min_pop_size) * cosine_factor
    
    def _exponential_reduction(self, cycle_progress: float) -> float:
        """Exponential population reduction
        
        Parameters
        ----------
        cycle_progress : float
            Progress within the current cycle (0 to 1)
            
        Returns
        -------
        float
            Population size
        """
        exp_factor = np.exp(-3 * cycle_progress)
        return self.min_pop_size + (self.initial_pop_size - self.min_pop_size) * exp_factor


class SelectionStrategies:
    """
    Class implementing various selection strategies for population reduction.
    
    This class provides methods for selecting which individuals to keep when
    reducing the population size.
    """
    
    @staticmethod
    def best_selection(
        positions: np.ndarray,
        fitness: np.ndarray,
        new_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select the best individuals based on fitness
        
        Parameters
        ----------
        positions : numpy.ndarray
            Current positions of all individuals
        fitness : numpy.ndarray
            Current fitness values of all individuals
        new_size : int
            New population size
            
        Returns
        -------
        tuple
            A tuple containing (selected_positions, selected_fitness)
        """
        indices = np.argsort(fitness)[:new_size]
        return positions[indices].copy(), fitness[indices].copy()
    
    @staticmethod
    def tournament_selection(
        positions: np.ndarray,
        fitness: np.ndarray,
        new_size: int,
        tournament_size: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select individuals using tournament selection
        
        Parameters
        ----------
        positions : numpy.ndarray
            Current positions of all individuals
        fitness : numpy.ndarray
            Current fitness values of all individuals
        new_size : int
            New population size
        tournament_size : int
            Size of each tournament
            
        Returns
        -------
        tuple
            A tuple containing (selected_positions, selected_fitness)
        """
        current_size = len(fitness)
        selected_indices = []
        
        for _ in range(new_size):
            # Select tournament participants
            tournament_indices = np.random.choice(current_size, tournament_size, replace=False)
            # Find the best participant
            winner_idx = tournament_indices[np.argmin(fitness[tournament_indices])]
            selected_indices.append(winner_idx)
        
        return positions[selected_indices].copy(), fitness[selected_indices].copy()
    
    @staticmethod
    def roulette_wheel_selection(
        positions: np.ndarray,
        fitness: np.ndarray,
        new_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select individuals using roulette wheel selection
        
        Parameters
        ----------
        positions : numpy.ndarray
            Current positions of all individuals
        fitness : numpy.ndarray
            Current fitness values of all individuals
        new_size : int
            New population size
            
        Returns
        -------
        tuple
            A tuple containing (selected_positions, selected_fitness)
        """
        # Convert fitness to selection probability (lower fitness = higher probability)
        max_fitness = np.max(fitness)
        if np.isinf(max_fitness):
            # Handle case where some fitness values are infinite
            selection_prob = np.where(np.isinf(fitness), 0, max_fitness - fitness)
        else:
            selection_prob = max_fitness - fitness
        
        # Handle case where all fitness values are the same
        if np.sum(selection_prob) == 0:
            selection_prob = np.ones_like(selection_prob)
        
        # Normalize probabilities
        selection_prob = selection_prob / np.sum(selection_prob)
        
        # Select individuals
        selected_indices = np.random.choice(
            len(fitness), new_size, replace=True, p=selection_prob
        )
        
        return positions[selected_indices].copy(), fitness[selected_indices].copy()
    
    @staticmethod
    def diversity_selection(
        positions: np.ndarray,
        fitness: np.ndarray,
        new_size: int,
        elite_fraction: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select individuals based on fitness and diversity
        
        Parameters
        ----------
        positions : numpy.ndarray
            Current positions of all individuals
        fitness : numpy.ndarray
            Current fitness values of all individuals
        new_size : int
            New population size
        elite_fraction : float
            Fraction of individuals to select based on fitness
            
        Returns
        -------
        tuple
            A tuple containing (selected_positions, selected_fitness)
        """
        current_size = len(fitness)
        
        # Select elite individuals based on fitness
        elite_size = max(1, int(new_size * elite_fraction))
        elite_indices = np.argsort(fitness)[:elite_size]
        
        # Select remaining individuals based on diversity
        remaining_size = new_size - elite_size
        if remaining_size > 0:
            # Calculate distances between all individuals
            distances = np.zeros((current_size, current_size))
            for i in range(current_size):
                for j in range(i+1, current_size):
                    distances[i, j] = np.linalg.norm(positions[i] - positions[j])
                    distances[j, i] = distances[i, j]
            
            # Select individuals with maximum diversity
            selected_indices = list(elite_indices)
            remaining_indices = list(set(range(current_size)) - set(elite_indices))
            
            while len(selected_indices) < new_size and remaining_indices:
                # Calculate minimum distance to selected individuals
                min_distances = np.array([
                    np.min([distances[i, j] for j in selected_indices])
                    for i in remaining_indices
                ])
                
                # Select individual with maximum minimum distance
                max_idx = remaining_indices[np.argmax(min_distances)]
                selected_indices.append(max_idx)
                remaining_indices.remove(max_idx)
        else:
            selected_indices = elite_indices
        
        return positions[selected_indices].copy(), fitness[selected_indices].copy()
