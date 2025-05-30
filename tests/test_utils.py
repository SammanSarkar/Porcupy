"""
Test utilities for Porcupy visualization tests.

This module provides utility functions for generating mock data for testing
visualization components.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable


def generate_mock_positions(
    pop_size: int,
    dimensions: int,
    iterations: int,
    bounds: Tuple[np.ndarray, np.ndarray]
) -> List[np.ndarray]:
    """
    Generate mock position history for testing.
    
    Parameters
    ----------
    pop_size : int
        Population size.
    dimensions : int
        Number of dimensions.
    iterations : int
        Number of iterations.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    
    Returns
    -------
    list
        List of position arrays, each with shape (pop_size, dimensions).
    """
    lb, ub = bounds
    position_history = []
    
    # Generate initial positions
    positions = lb + (ub - lb) * np.random.random((pop_size, dimensions))
    
    # Add some convergence behavior
    best_position = lb + (ub - lb) * np.random.random(dimensions)
    
    for i in range(iterations):
        # Move positions gradually towards the best position
        convergence_factor = i / iterations
        noise_factor = 1 - convergence_factor
        
        # Add some random movement and convergence
        noise = noise_factor * (ub - lb) * (np.random.random((pop_size, dimensions)) - 0.5)
        attraction = convergence_factor * 0.1 * (best_position - positions)
        
        positions = positions + noise + attraction
        
        # Ensure positions stay within bounds
        positions = np.maximum(positions, lb)
        positions = np.minimum(positions, ub)
        
        position_history.append(positions.copy())
    
    return position_history


def generate_mock_fitness(
    position_history: List[np.ndarray],
    func: Callable
) -> List[float]:
    """
    Generate mock fitness history based on positions.
    
    Parameters
    ----------
    position_history : list
        List of position arrays, each with shape (pop_size, dimensions).
    func : callable
        Objective function to evaluate positions.
    
    Returns
    -------
    list
        List of best fitness values at each iteration.
    """
    fitness_history = []
    
    for positions in position_history:
        # Evaluate each position
        fitness_values = np.array([func(pos) for pos in positions])
        
        # Get the best fitness
        best_fitness = np.min(fitness_values)
        fitness_history.append(best_fitness)
    
    return fitness_history


def generate_mock_best_positions(
    position_history: List[np.ndarray],
    func: Callable
) -> List[np.ndarray]:
    """
    Generate mock best position history based on positions.
    
    Parameters
    ----------
    position_history : list
        List of position arrays, each with shape (pop_size, dimensions).
    func : callable
        Objective function to evaluate positions.
    
    Returns
    -------
    list
        List of best position arrays at each iteration.
    """
    best_position_history = []
    
    for positions in position_history:
        # Evaluate each position
        fitness_values = np.array([func(pos) for pos in positions])
        
        # Get the index of the best position
        best_idx = np.argmin(fitness_values)
        
        # Get the best position
        best_position = positions[best_idx].copy()
        best_position_history.append(best_position)
    
    return best_position_history


def generate_mock_pop_size_history(
    initial_pop_size: int,
    iterations: int,
    cycles: int,
    strategy: str = 'linear'
) -> List[int]:
    """
    Generate mock population size history.
    
    Parameters
    ----------
    initial_pop_size : int
        Initial population size.
    iterations : int
        Number of iterations.
    cycles : int
        Number of cycles.
    strategy : str, optional
        Population reduction strategy (default: 'linear').
    
    Returns
    -------
    list
        List of population sizes at each iteration.
    """
    pop_size_history = []
    
    # Calculate cycle length
    cycle_length = iterations // cycles
    
    for i in range(iterations):
        cycle = i // cycle_length
        cycle_progress = (i % cycle_length) / cycle_length
        
        if strategy == 'linear':
            # Linear reduction
            current_pop = initial_pop_size - (initial_pop_size - 10) * cycle_progress
        elif strategy == 'cosine':
            # Cosine reduction
            current_pop = initial_pop_size - (initial_pop_size - 10) * (1 - np.cos(cycle_progress * np.pi)) / 2
        elif strategy == 'exponential':
            # Exponential reduction
            current_pop = initial_pop_size - (initial_pop_size - 10) * (1 - np.exp(-5 * cycle_progress)) / (1 - np.exp(-5))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        pop_size_history.append(max(10, int(current_pop)))
    
    return pop_size_history


def generate_mock_defense_history(
    iterations: int,
    pop_size_history: List[int]
) -> Dict[str, List[int]]:
    """
    Generate mock defense mechanism history.
    
    Parameters
    ----------
    iterations : int
        Number of iterations.
    pop_size_history : list
        List of population sizes at each iteration.
    
    Returns
    -------
    dict
        Dictionary with keys as defense mechanisms and values as lists of activation counts.
    """
    defense_history = {
        'sight': [],
        'sound': [],
        'odor': [],
        'physical': []
    }
    
    for i in range(iterations):
        pop_size = pop_size_history[i]
        
        # Early iterations favor exploration (sight, sound)
        # Later iterations favor exploitation (odor, physical)
        exploration_ratio = 1 - (i / iterations)
        
        # Calculate counts for each defense mechanism
        sight_count = int(pop_size * exploration_ratio * 0.6)
        sound_count = int(pop_size * exploration_ratio * 0.4)
        odor_count = int(pop_size * (1 - exploration_ratio) * 0.7)
        physical_count = pop_size - sight_count - sound_count - odor_count
        
        # Ensure counts are non-negative
        sight_count = max(0, sight_count)
        sound_count = max(0, sound_count)
        odor_count = max(0, odor_count)
        physical_count = max(0, physical_count)
        
        # Adjust to match population size
        total = sight_count + sound_count + odor_count + physical_count
        if total != pop_size:
            physical_count += (pop_size - total)
        
        # Store counts
        defense_history['sight'].append(sight_count)
        defense_history['sound'].append(sound_count)
        defense_history['odor'].append(odor_count)
        defense_history['physical'].append(physical_count)
    
    return defense_history


def generate_mock_diversity_history(
    position_history: List[np.ndarray]
) -> List[float]:
    """
    Generate mock diversity history based on positions.
    
    Parameters
    ----------
    position_history : list
        List of position arrays, each with shape (pop_size, dimensions).
    
    Returns
    -------
    list
        List of diversity measures at each iteration.
    """
    diversity_history = []
    
    for positions in position_history:
        # Calculate diversity as average pairwise distance
        n_particles = positions.shape[0]
        
        if n_particles <= 1:
            diversity = 0.0
        else:
            # Calculate pairwise distances
            distances = np.zeros((n_particles, n_particles))
            for i in range(n_particles):
                for j in range(i+1, n_particles):
                    distances[i, j] = np.linalg.norm(positions[i] - positions[j])
                    distances[j, i] = distances[i, j]
            
            # Average pairwise distance
            diversity = np.sum(distances) / (n_particles * (n_particles - 1))
        
        diversity_history.append(diversity)
    
    return diversity_history


def generate_mock_velocities(
    position_history: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Generate mock velocity history based on position changes.
    
    Parameters
    ----------
    position_history : list
        List of position arrays, each with shape (pop_size, dimensions).
    
    Returns
    -------
    list
        List of velocity arrays, each with shape (pop_size, dimensions).
    """
    velocity_history = []
    
    for i in range(len(position_history)):
        if i == 0:
            # For the first iteration, use small random velocities
            pop_size, dimensions = position_history[i].shape
            velocities = 0.1 * np.random.randn(pop_size, dimensions)
        else:
            # Calculate velocities as position differences
            velocities = position_history[i] - position_history[i-1]
        
        velocity_history.append(velocities)
    
    return velocity_history


def generate_mock_defense_types(
    pop_size: int,
    iterations: int
) -> List[List[str]]:
    """
    Generate mock defense types for each porcupine at each iteration.
    
    Parameters
    ----------
    pop_size : int
        Population size.
    iterations : int
        Number of iterations.
    
    Returns
    -------
    list
        List of lists containing defense types for each porcupine at each iteration.
    """
    defense_types_history = []
    
    for i in range(iterations):
        # Early iterations favor exploration (sight, sound)
        # Later iterations favor exploitation (odor, physical)
        exploration_ratio = 1 - (i / iterations)
        
        defense_types = []
        for j in range(pop_size):
            if np.random.random() < exploration_ratio:
                # Exploration phase
                defense = np.random.choice(['sight', 'sound'], p=[0.6, 0.4])
            else:
                # Exploitation phase
                defense = np.random.choice(['odor', 'physical'], p=[0.7, 0.3])
            
            defense_types.append(defense)
        
        defense_types_history.append(defense_types)
    
    return defense_types_history


def mock_objective_function(x) -> float:
    """
    A simple mock objective function for testing.

    Parameters
    ----------
    x : ndarray or list
        Position vector.

    Returns
    -------
    float
        Function value.
    """
    # Convert to numpy array if it's a list
    if isinstance(x, list):
        x = np.array(x)
    return np.sum(x**2)  # Simple sphere function
