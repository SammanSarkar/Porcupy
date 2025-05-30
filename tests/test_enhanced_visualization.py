"""
Tests for the enhanced visualization module.

This module contains tests for the enhanced visualization functions in the Porcupy library.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from porcupy.utils.enhanced_visualization import (
    plot_defense_mechanisms,
    plot_population_cycles,
    plot_diversity_history,
    plot_2d_porcupines,
    animate_porcupines_2d,
    plot_3d_porcupines,
    calculate_diversity,
    track_defense_mechanisms
)

from tests.test_utils import (
    generate_mock_positions,
    generate_mock_fitness,
    generate_mock_best_positions,
    generate_mock_pop_size_history,
    generate_mock_defense_history,
    generate_mock_diversity_history,
    generate_mock_velocities,
    generate_mock_defense_types,
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
    defense_types_history = generate_mock_defense_types(pop_size, iterations)
    
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
        'diversity_history': diversity_history,
        'defense_types_history': defense_types_history
    }


def test_plot_defense_mechanisms(test_data_2d):
    """Test the plot_defense_mechanisms function."""
    # Get test data
    defense_history = test_data_2d['defense_history']
    
    # Test with default parameters
    fig = plot_defense_mechanisms(defense_history)
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    colors = {
        'sight': 'cyan',
        'sound': 'magenta',
        'odor': 'yellow',
        'physical': 'black'
    }
    
    fig = plot_defense_mechanisms(
        defense_history=defense_history,
        title="Custom Defense Plot",
        figsize=(10, 5),
        colors=colors
    )
    assert isinstance(fig, Figure)
    
    # Test with empty data
    empty_defense_history = {
        'sight': [],
        'sound': [],
        'odor': [],
        'physical': []
    }
    
    with pytest.raises(IndexError):
        plot_defense_mechanisms(empty_defense_history)
    
    plt.close('all')


def test_plot_population_cycles(test_data_2d):
    """Test the plot_population_cycles function."""
    # Get test data
    pop_size_history = test_data_2d['pop_size_history']
    iterations = test_data_2d['iterations']
    
    # Test with default parameters
    fig = plot_population_cycles(
        pop_size_history=pop_size_history,
        cycles=3,
        max_iter=iterations
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = plot_population_cycles(
        pop_size_history=pop_size_history,
        cycles=3,
        max_iter=iterations,
        title="Custom Population Cycles",
        figsize=(12, 5)
    )
    assert isinstance(fig, Figure)
    
    # Test with empty data
    with pytest.raises(IndexError):
        plot_population_cycles([], cycles=3, max_iter=iterations)
    
    plt.close('all')


def test_plot_diversity_history(test_data_2d):
    """Test the plot_diversity_history function."""
    # Get test data
    diversity_history = test_data_2d['diversity_history']
    
    # Test with default parameters
    fig = plot_diversity_history(diversity_history)
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = plot_diversity_history(
        diversity_history=diversity_history,
        title="Custom Diversity Plot",
        figsize=(10, 5)
    )
    assert isinstance(fig, Figure)
    
    # Test with empty data
    with pytest.raises(IndexError):
        plot_diversity_history([])
    
    plt.close('all')


def test_plot_2d_porcupines(test_data_2d):
    """Test the plot_2d_porcupines function."""
    # Get test data
    positions = test_data_2d['position_history'][-1]
    bounds = test_data_2d['bounds']
    best_pos = test_data_2d['best_position_history'][-1]
    defense_types = test_data_2d['defense_types_history'][-1]
    
    # Test with default parameters
    fig = plot_2d_porcupines(
        positions=positions,
        func=mock_objective_function,
        bounds=bounds
    )
    assert isinstance(fig, Figure)
    
    # Test with best position
    fig = plot_2d_porcupines(
        positions=positions,
        func=mock_objective_function,
        bounds=bounds,
        best_pos=best_pos
    )
    assert isinstance(fig, Figure)
    
    # Test with defense types
    fig = plot_2d_porcupines(
        positions=positions,
        func=mock_objective_function,
        bounds=bounds,
        best_pos=best_pos,
        defense_types=defense_types
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = plot_2d_porcupines(
        positions=positions,
        func=mock_objective_function,
        bounds=bounds,
        best_pos=best_pos,
        defense_types=defense_types,
        title="Custom 2D Plot",
        figsize=(12, 10),
        cmap='plasma',
        contour_levels=15,
        quill_length=0.3
    )
    assert isinstance(fig, Figure)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        plot_2d_porcupines(
            positions=np.random.random((10, 3)),  # 3D positions
            func=mock_objective_function,
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5]))
        )
    
    plt.close('all')


def test_animate_porcupines_2d(test_data_2d):
    """Test the animate_porcupines_2d function."""
    # Get test data
    position_history = test_data_2d['position_history'][:5]  # Use fewer iterations for speed
    bounds = test_data_2d['bounds']
    best_pos_history = test_data_2d['best_position_history'][:5]
    defense_history = test_data_2d['defense_types_history'][:5]
    
    # Test with default parameters
    anim = animate_porcupines_2d(
        position_history=position_history,
        func=mock_objective_function,
        bounds=bounds
    )
    assert isinstance(anim, FuncAnimation)
    
    # Test with best positions
    anim = animate_porcupines_2d(
        position_history=position_history,
        func=mock_objective_function,
        bounds=bounds,
        best_pos_history=best_pos_history
    )
    assert isinstance(anim, FuncAnimation)
    
    # Test with defense history
    anim = animate_porcupines_2d(
        position_history=position_history,
        func=mock_objective_function,
        bounds=bounds,
        best_pos_history=best_pos_history,
        defense_history=defense_history
    )
    assert isinstance(anim, FuncAnimation)
    
    # Test with custom parameters
    anim = animate_porcupines_2d(
        position_history=position_history,
        func=mock_objective_function,
        bounds=bounds,
        best_pos_history=best_pos_history,
        defense_history=defense_history,
        interval=100,
        figsize=(8, 6),
        cmap='plasma',
        contour_levels=15,
        quill_length=0.3
    )
    assert isinstance(anim, FuncAnimation)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        animate_porcupines_2d(
            position_history=[np.random.random((10, 3)) for _ in range(5)],  # 3D positions
            func=mock_objective_function,
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5]))
        )
    
    # Test with empty position history
    with pytest.raises(ValueError):
        animate_porcupines_2d(
            position_history=[],
            func=mock_objective_function,
            bounds=bounds
        )
    
    plt.close('all')


def test_plot_3d_porcupines(test_data_2d):
    """Test the plot_3d_porcupines function."""
    # Get test data
    positions = test_data_2d['position_history'][-1]
    bounds = test_data_2d['bounds']
    best_pos = test_data_2d['best_position_history'][-1]
    defense_types = test_data_2d['defense_types_history'][-1]
    
    # Calculate fitness values
    fitness = np.array([mock_objective_function(pos) for pos in positions])
    
    # Test with default parameters
    fig = plot_3d_porcupines(
        positions=positions,
        fitness=fitness,
        func=mock_objective_function,
        bounds=bounds
    )
    assert isinstance(fig, Figure)
    
    # Test with best position
    fig = plot_3d_porcupines(
        positions=positions,
        fitness=fitness,
        func=mock_objective_function,
        bounds=bounds,
        best_pos=best_pos
    )
    assert isinstance(fig, Figure)
    
    # Test with defense types
    fig = plot_3d_porcupines(
        positions=positions,
        fitness=fitness,
        func=mock_objective_function,
        bounds=bounds,
        best_pos=best_pos,
        defense_types=defense_types
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = plot_3d_porcupines(
        positions=positions,
        fitness=fitness,
        func=mock_objective_function,
        bounds=bounds,
        best_pos=best_pos,
        defense_types=defense_types,
        title="Custom 3D Plot",
        figsize=(12, 10),
        cmap='plasma',
        alpha=0.5
    )
    assert isinstance(fig, Figure)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        plot_3d_porcupines(
            positions=np.random.random((10, 3)),  # 3D positions
            fitness=fitness,
            func=mock_objective_function,
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5]))
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


def test_track_defense_mechanisms(test_data_2d):
    """Test the track_defense_mechanisms function."""
    # Get test data
    positions = test_data_2d['position_history'][-1]
    prev_positions = test_data_2d['position_history'][-2]
    best_pos = test_data_2d['best_position_history'][-1]
    
    # Test with default parameters
    defense_types = track_defense_mechanisms(
        positions=positions,
        prev_positions=prev_positions,
        best_pos=best_pos
    )
    
    assert len(defense_types) == positions.shape[0]
    assert all(d in ['sight', 'sound', 'odor', 'physical'] for d in defense_types)
    
    # Test with custom tf parameter
    defense_types = track_defense_mechanisms(
        positions=positions,
        prev_positions=prev_positions,
        best_pos=best_pos,
        tf=0.5
    )
    
    assert len(defense_types) == positions.shape[0]
    assert all(d in ['sight', 'sound', 'odor', 'physical'] for d in defense_types)
