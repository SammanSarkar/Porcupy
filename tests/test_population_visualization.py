"""
Tests for the population visualization module.

This module contains tests for the population visualization functions in the Porcupy library.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from porcupy.utils.population_visualization import (
    plot_population_reduction_strategies,
    plot_population_diversity_map,
    animate_population_cycle,
    plot_exploration_exploitation_balance,
    plot_diversity_vs_convergence,
    calculate_diversity
)

from tests.test_utils import (
    generate_mock_positions,
    generate_mock_fitness,
    generate_mock_best_positions,
    generate_mock_pop_size_history,
    generate_mock_defense_history,
    generate_mock_diversity_history,
    mock_objective_function
)


# Setup common test data
@pytest.fixture
def test_data_2d():
    """Generate common 2D test data for visualization tests."""
    dimensions = 2
    pop_size = 20
    iterations = 30
    lb = -5 * np.ones(dimensions)
    ub = 5 * np.ones(dimensions)
    bounds = (lb, ub)
    
    position_history = generate_mock_positions(pop_size, dimensions, iterations, bounds)
    best_position_history = generate_mock_best_positions(position_history, mock_objective_function)
    fitness_history = generate_mock_fitness(position_history, mock_objective_function)
    pop_size_history = generate_mock_pop_size_history(pop_size, iterations, cycles=3)
    defense_history = generate_mock_defense_history(iterations, pop_size_history)
    diversity_history = generate_mock_diversity_history(position_history)
    
    return {
        'dimensions': dimensions,
        'pop_size': pop_size,
        'iterations': iterations,
        'bounds': bounds,
        'position_history': position_history,
        'best_position_history': best_position_history,
        'fitness_history': fitness_history,
        'pop_size_history': pop_size_history,
        'defense_history': defense_history,
        'diversity_history': diversity_history
    }


def test_plot_population_reduction_strategies():
    """Test the plot_population_reduction_strategies function."""
    # Test with default parameters
    fig = plot_population_reduction_strategies(
        max_iter=50,
        pop_size=30,
        cycles=5
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = plot_population_reduction_strategies(
        max_iter=50,
        pop_size=30,
        cycles=5,
        strategies=['linear', 'exponential'],
        figsize=(10, 5)
    )
    assert isinstance(fig, Figure)
    
    # Test with invalid strategy
    with pytest.raises(ValueError):
        plot_population_reduction_strategies(
            max_iter=50,
            pop_size=30,
            cycles=5,
            strategies=['invalid_strategy']
        )
    
    plt.close('all')


def test_plot_population_diversity_map(test_data_2d):
    """Test the plot_population_diversity_map function."""
    # Get test data
    position_history = test_data_2d['position_history']
    bounds = test_data_2d['bounds']
    
    # Test with default parameters
    fig = plot_population_diversity_map(
        positions_history=position_history,
        bounds=bounds,
        sample_iterations=[0, 10, 20]
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = plot_population_diversity_map(
        positions_history=position_history,
        bounds=bounds,
        sample_iterations=[0, 10, 20],
        figsize=(15, 8),
        cmap='plasma'
    )
    assert isinstance(fig, Figure)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        plot_population_diversity_map(
            positions_history=[np.random.random((10, 3)) for _ in range(30)],  # 3D positions
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5])),
            sample_iterations=[0, 10, 20]
        )
    
    # Test with empty position history
    with pytest.raises(ValueError):
        plot_population_diversity_map(
            positions_history=[],
            bounds=bounds,
            sample_iterations=[0, 10, 20]
        )
    
    plt.close('all')


def test_animate_population_cycle(test_data_2d):
    """Test the animate_population_cycle function."""
    # Get test data
    position_history = test_data_2d['position_history'][:10]  # Use fewer iterations for speed
    pop_size_history = test_data_2d['pop_size_history'][:10]
    bounds = test_data_2d['bounds']
    
    # Test with default parameters
    anim = animate_population_cycle(
        positions_history=position_history,
        pop_size_history=pop_size_history,
        bounds=bounds,
        max_iter=10,
        cycles=2
    )
    assert isinstance(anim, FuncAnimation)
    
    # Test with custom parameters
    anim = animate_population_cycle(
        positions_history=position_history,
        pop_size_history=pop_size_history,
        bounds=bounds,
        max_iter=10,
        cycles=2,
        interval=100,
        figsize=(12, 8),
        cmap='plasma'
    )
    assert isinstance(anim, FuncAnimation)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        animate_population_cycle(
            positions_history=[np.random.random((10, 3)) for _ in range(10)],  # 3D positions
            pop_size_history=pop_size_history[:10],
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5])),
            max_iter=10,
            cycles=2
        )
    
    # Test with empty position history
    with pytest.raises(ValueError):
        animate_population_cycle(
            positions_history=[],
            pop_size_history=[],
            bounds=bounds,
            max_iter=10,
            cycles=2
        )
    
    plt.close('all')


def test_plot_exploration_exploitation_balance(test_data_2d):
    """Test the plot_exploration_exploitation_balance function."""
    # Get test data
    position_history = test_data_2d['position_history']
    best_position_history = test_data_2d['best_position_history']
    bounds = test_data_2d['bounds']
    
    # Test with default parameters
    fig = plot_exploration_exploitation_balance(
        positions_history=position_history,
        best_positions_history=best_position_history,
        bounds=bounds,
        sample_iterations=[0, 10, 20]
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = plot_exploration_exploitation_balance(
        positions_history=position_history,
        best_positions_history=best_position_history,
        bounds=bounds,
        sample_iterations=[0, 10, 20],
        figsize=(15, 8)
    )
    assert isinstance(fig, Figure)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        plot_exploration_exploitation_balance(
            positions_history=[np.random.random((10, 3)) for _ in range(30)],  # 3D positions
            best_positions_history=[np.random.random(3) for _ in range(30)],
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5])),
            sample_iterations=[0, 10, 20]
        )
    
    # Test with empty position history
    with pytest.raises(ValueError):
        plot_exploration_exploitation_balance(
            positions_history=[],
            best_positions_history=[],
            bounds=bounds,
            sample_iterations=[0, 10, 20]
        )
    
    plt.close('all')


def test_plot_diversity_vs_convergence(test_data_2d):
    """Test the plot_diversity_vs_convergence function."""
    # Get test data
    diversity_history = test_data_2d['diversity_history']
    fitness_history = test_data_2d['fitness_history']
    
    # Test with default parameters
    fig = plot_diversity_vs_convergence(
        diversity_history=diversity_history,
        fitness_history=fitness_history,
        cycles=3,
        max_iter=30
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = plot_diversity_vs_convergence(
        diversity_history=diversity_history,
        fitness_history=fitness_history,
        cycles=3,
        max_iter=30,
        title="Custom Diversity vs Convergence",
        figsize=(12, 6)
    )
    assert isinstance(fig, Figure)
    
    # Test with empty data
    with pytest.raises(IndexError):
        plot_diversity_vs_convergence(
            diversity_history=[],
            fitness_history=[],
            cycles=3,
            max_iter=30
        )
    
    plt.close('all')


def test_calculate_diversity():
    """Test the calculate_diversity function."""
    # Test with multiple positions
    positions = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    
    diversity = calculate_diversity(positions)
    assert diversity > 0
    
    # Test with identical positions (zero diversity)
    positions = np.array([
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    
    diversity = calculate_diversity(positions)
    assert diversity == 0
    
    # Test with single position
    positions = np.array([[1, 1]])
    
    diversity = calculate_diversity(positions)
    assert diversity == 0
