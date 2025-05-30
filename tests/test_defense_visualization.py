"""
Tests for the defense visualization module.

This module contains tests for the defense visualization functions in the Porcupy library.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

# Add the parent directory to the path to ensure imports work
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy.utils.defense_visualization import (
    visualize_defense_territories,
    visualize_defense_mechanisms,
    animate_defense_mechanisms,
    plot_defense_effectiveness,
    visualize_quill_directions
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
    velocity_history = generate_mock_velocities(position_history)
    
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
        'defense_types_history': defense_types_history,
        'velocity_history': velocity_history
    }


def test_visualize_defense_territories(test_data_2d):
    """Test the visualize_defense_territories function."""
    # Get test data
    positions = test_data_2d['position_history'][-1]
    bounds = test_data_2d['bounds']
    defense_types = test_data_2d['defense_types_history'][-1]
    
    # Test with default parameters
    fig = visualize_defense_territories(
        positions=positions,
        defense_types=defense_types,
        bounds=bounds
    )
    assert isinstance(fig, Figure)
    
    # Test with custom territory sizes
    territory_sizes = np.random.random(positions.shape[0]) * 2
    
    fig = visualize_defense_territories(
        positions=positions,
        defense_types=defense_types,
        bounds=bounds,
        territory_sizes=territory_sizes,
        title="Custom Territory Plot",
        figsize=(12, 10)
    )
    assert isinstance(fig, Figure)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        visualize_defense_territories(
            positions=np.random.random((10, 3)),  # 3D positions
            defense_types=defense_types[:10],
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5]))
        )
    
    plt.close('all')


def test_visualize_defense_mechanisms(test_data_2d):
    """Test the visualize_defense_mechanisms function."""
    # Get test data
    positions = test_data_2d['position_history'][-1]
    prev_positions = test_data_2d['position_history'][-2]
    bounds = test_data_2d['bounds']
    defense_types = test_data_2d['defense_types_history'][-1]
    best_position = test_data_2d['best_position_history'][-1]
    
    # Test with default parameters
    fig = visualize_defense_mechanisms(
        positions=positions,
        prev_positions=prev_positions,
        defense_types=defense_types,
        bounds=bounds
    )
    assert isinstance(fig, Figure)
    
    # Test with best position
    fig = visualize_defense_mechanisms(
        positions=positions,
        prev_positions=prev_positions,
        defense_types=defense_types,
        bounds=bounds,
        best_position=best_position
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualize_defense_mechanisms(
        positions=positions,
        prev_positions=prev_positions,
        defense_types=defense_types,
        bounds=bounds,
        best_position=best_position,
        title="Custom Defense Mechanisms",
        figsize=(12, 10)
    )
    assert isinstance(fig, Figure)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        visualize_defense_mechanisms(
            positions=np.random.random((10, 3)),  # 3D positions
            prev_positions=np.random.random((10, 3)),
            defense_types=defense_types[:10],
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5]))
        )
    
    plt.close('all')


def test_animate_defense_mechanisms(test_data_2d):
    """Test the animate_defense_mechanisms function."""
    # Get test data
    position_history = test_data_2d['position_history'][:5]  # Use fewer iterations for speed
    defense_types_history = test_data_2d['defense_types_history'][:5]
    bounds = test_data_2d['bounds']
    best_position_history = test_data_2d['best_position_history'][:5]
    
    # Test with default parameters
    anim = animate_defense_mechanisms(
        position_history=position_history,
        defense_history=defense_types_history,
        bounds=bounds
    )
    assert isinstance(anim, FuncAnimation)
    
    # Test with best position history
    anim = animate_defense_mechanisms(
        position_history=position_history,
        defense_history=defense_types_history,
        bounds=bounds,
        best_position_history=best_position_history
    )
    assert isinstance(anim, FuncAnimation)
    
    # Test with custom parameters
    anim = animate_defense_mechanisms(
        position_history=position_history,
        defense_history=defense_types_history,
        bounds=bounds,
        best_position_history=best_position_history,
        interval=100,
        figsize=(12, 10)
    )
    assert isinstance(anim, FuncAnimation)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        animate_defense_mechanisms(
            position_history=[np.random.random((10, 3)) for _ in range(5)],  # 3D positions
            defense_history=defense_types_history,
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5]))
        )
    
    # Test with empty position history
    with pytest.raises(ValueError):
        animate_defense_mechanisms(
            position_history=[],
            defense_history=[],
            bounds=bounds
        )
    
    plt.close('all')


def test_plot_defense_effectiveness(test_data_2d):
    """Test the plot_defense_effectiveness function."""
    # Get test data
    defense_history = test_data_2d['defense_history']
    fitness_history = test_data_2d['fitness_history']
    
    # Test with default parameters
    fig = plot_defense_effectiveness(
        defense_history=defense_history,
        fitness_history=fitness_history
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = plot_defense_effectiveness(
        defense_history=defense_history,
        fitness_history=fitness_history,
        title="Custom Effectiveness Plot",
        figsize=(12, 8)
    )
    assert isinstance(fig, Figure)
    
    # Test with empty data
    with pytest.raises(IndexError):
        plot_defense_effectiveness(
            defense_history={
                'sight': [],
                'sound': [],
                'odor': [],
                'physical': []
            },
            fitness_history=[]
        )
    
    plt.close('all')


def test_visualize_quill_directions(test_data_2d):
    """Test the visualize_quill_directions function."""
    # Get test data
    positions = test_data_2d['position_history'][-1]
    velocities = test_data_2d['velocity_history'][-1]
    bounds = test_data_2d['bounds']
    
    # Test with default parameters
    fig = visualize_quill_directions(
        positions=positions,
        velocities=velocities,
        bounds=bounds
    )
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualize_quill_directions(
        positions=positions,
        velocities=velocities,
        bounds=bounds,
        title="Custom Quill Directions",
        figsize=(12, 10)
    )
    assert isinstance(fig, Figure)
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        visualize_quill_directions(
            positions=np.random.random((10, 3)),  # 3D positions
            velocities=np.random.random((10, 3)),
            bounds=(np.array([-5, -5, -5]), np.array([5, 5, 5]))
        )
    
    plt.close('all')
