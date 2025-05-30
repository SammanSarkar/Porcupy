"""
Tests for the visualization manager module.

This module contains tests for the visualization manager in the Porcupy library.
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

from porcupy.utils.visualization_manager import CPOVisualizer
from porcupy.utils.interactive_visualization import (
    OptimizationDashboard,
    ParameterTuningDashboard
)

from tests.test_utils import (
    generate_mock_positions,
    generate_mock_fitness,
    generate_mock_best_positions,
    generate_mock_pop_size_history,
    generate_mock_defense_history,
    generate_mock_diversity_history,
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


@pytest.fixture
def visualizer_with_data(test_data_2d):
    """Create a visualizer with test data."""
    bounds = test_data_2d['bounds']
    
    # Create visualizer
    visualizer = CPOVisualizer(
        objective_func=mock_objective_function,
        bounds=bounds
    )
    
    # Record data for each iteration
    for i in range(test_data_2d['iterations']):
        visualizer.record_iteration(
            iteration=i,
            positions=test_data_2d['position_history'][i],
            best_position=test_data_2d['best_position_history'][i],
            best_fitness=test_data_2d['fitness_history'][i],
            pop_size=test_data_2d['pop_size_history'][i],
            defense_types=test_data_2d['defense_types_history'][i]
        )
    
    return visualizer


def test_cpo_visualizer_init():
    """Test the CPOVisualizer initialization."""
    # Test initialization with objective function and bounds
    dimensions = 2
    lb = -5 * np.ones(dimensions)
    ub = 5 * np.ones(dimensions)
    bounds = (lb, ub)
    
    visualizer = CPOVisualizer(
        objective_func=mock_objective_function,
        bounds=bounds
    )
    
    assert visualizer.objective_func == mock_objective_function
    assert visualizer.bounds == bounds
    assert len(visualizer.position_history) == 0
    assert len(visualizer.best_position_history) == 0
    assert len(visualizer.fitness_history) == 0
    assert len(visualizer.pop_size_history) == 0
    assert len(visualizer.defense_history) == 0
    assert len(visualizer.diversity_history) == 0
    
    # Test initialization without objective function and bounds
    visualizer = CPOVisualizer()
    
    assert visualizer.objective_func is None
    assert visualizer.bounds is None


def test_record_iteration(test_data_2d):
    """Test the record_iteration method."""
    # Create visualizer
    dimensions = 2
    lb = -5 * np.ones(dimensions)
    ub = 5 * np.ones(dimensions)
    bounds = (lb, ub)
    
    visualizer = CPOVisualizer(
        objective_func=mock_objective_function,
        bounds=bounds
    )
    
    # Record a single iteration
    iteration = 0
    positions = test_data_2d['position_history'][iteration]
    best_position = test_data_2d['best_position_history'][iteration]
    best_fitness = test_data_2d['fitness_history'][iteration]
    pop_size = test_data_2d['pop_size_history'][iteration]
    defense_types = test_data_2d['defense_types_history'][iteration]
    
    visualizer.record_iteration(
        iteration=iteration,
        positions=positions,
        best_position=best_position,
        best_fitness=best_fitness,
        pop_size=pop_size,
        defense_types=defense_types
    )
    
    # Check that data was recorded correctly
    assert len(visualizer.position_history) == 1
    assert len(visualizer.best_position_history) == 1
    assert len(visualizer.fitness_history) == 1
    assert len(visualizer.pop_size_history) == 1
    assert len(visualizer.diversity_history) == 1
    
    # Check defense history
    for defense in ['sight', 'sound', 'odor', 'physical']:
        assert defense in visualizer.defense_history
        assert len(visualizer.defense_history[defense]) == 1
    
    # Record without defense types
    visualizer.record_iteration(
        iteration=iteration + 1,
        positions=positions,
        best_position=best_position,
        best_fitness=best_fitness,
        pop_size=pop_size
    )
    
    # Check that data was recorded correctly
    assert len(visualizer.position_history) == 2
    assert len(visualizer.best_position_history) == 2
    assert len(visualizer.fitness_history) == 2
    assert len(visualizer.pop_size_history) == 2
    assert len(visualizer.diversity_history) == 2


def test_create_dashboard(visualizer_with_data):
    """Test the create_dashboard method."""
    # Create dashboard
    dashboard = visualizer_with_data.create_dashboard()
    
    assert isinstance(dashboard, OptimizationDashboard)
    
    # Test with custom parameters
    dashboard = visualizer_with_data.create_dashboard(
        update_interval=0.2,
        figsize=(12, 8)
    )
    
    assert isinstance(dashboard, OptimizationDashboard)
    
    # Test without objective function and bounds
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.create_dashboard()
    
    # Clean up
    dashboard.close()


def test_visualize_defense_mechanisms(visualizer_with_data):
    """Test the visualize_defense_mechanisms method."""
    # Test with default parameters
    fig = visualizer_with_data.visualize_defense_mechanisms()
    
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualizer_with_data.visualize_defense_mechanisms(
        title="Custom Defense Plot",
        figsize=(10, 5)
    )
    
    assert isinstance(fig, Figure)
    
    # Test with empty data
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.visualize_defense_mechanisms()
    
    plt.close('all')


def test_visualize_population_cycles(visualizer_with_data):
    """Test the visualize_population_cycles method."""
    # Test with default parameters
    fig = visualizer_with_data.visualize_population_cycles(
        cycles=3,
        max_iter=30
    )
    
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualizer_with_data.visualize_population_cycles(
        cycles=3,
        max_iter=30,
        title="Custom Population Cycles",
        figsize=(12, 5)
    )
    
    assert isinstance(fig, Figure)
    
    # Test with empty data
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.visualize_population_cycles(cycles=3, max_iter=30)
    
    plt.close('all')


def test_visualize_diversity_history(visualizer_with_data):
    """Test the visualize_diversity_history method."""
    # Test with default parameters
    fig = visualizer_with_data.visualize_diversity_history()
    
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualizer_with_data.visualize_diversity_history(
        title="Custom Diversity Plot",
        figsize=(10, 5)
    )
    
    assert isinstance(fig, Figure)
    
    # Test with empty data
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.visualize_diversity_history()
    
    plt.close('all')


def test_visualize_porcupines_2d(visualizer_with_data):
    """Test the visualize_porcupines_2d method."""
    # Test with default parameters
    fig = visualizer_with_data.visualize_porcupines_2d()
    
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualizer_with_data.visualize_porcupines_2d(
        iteration=5,
        title="Custom 2D Plot",
        figsize=(12, 10),
        cmap='plasma',
        contour_levels=15,
        quill_length=0.3
    )
    
    assert isinstance(fig, Figure)
    
    # Test with empty data
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.visualize_porcupines_2d()
    
    # Test without objective function and bounds
    visualizer = CPOVisualizer()
    visualizer.position_history = [np.random.random((10, 2))]
    
    with pytest.raises(ValueError):
        visualizer.visualize_porcupines_2d()
    
    plt.close('all')


def test_animate_optimization(visualizer_with_data):
    """Test the animate_optimization method."""
    # Test with default parameters
    anim = visualizer_with_data.animate_optimization()
    
    assert isinstance(anim, FuncAnimation)
    
    # Test with custom parameters
    anim = visualizer_with_data.animate_optimization(
        interval=100,
        figsize=(8, 6),
        cmap='plasma',
        contour_levels=15,
        quill_length=0.3
    )
    
    assert isinstance(anim, FuncAnimation)
    
    # Test with empty data
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.animate_optimization()
    
    # Test without objective function and bounds
    visualizer = CPOVisualizer()
    visualizer.position_history = [np.random.random((10, 2))]
    
    with pytest.raises(ValueError):
        visualizer.animate_optimization()
    
    plt.close('all')


def test_visualize_defense_territories(visualizer_with_data):
    """Test the visualize_defense_territories method."""
    # Test with default parameters
    fig = visualizer_with_data.visualize_defense_territories()
    
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualizer_with_data.visualize_defense_territories(
        iteration=5,
        title="Custom Defense Territories",
        figsize=(12, 10)
    )
    
    assert isinstance(fig, Figure)
    
    # Test with empty data
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.visualize_defense_territories()
    
    # Test without bounds
    visualizer = CPOVisualizer()
    visualizer.position_history = [np.random.random((10, 2))]
    
    with pytest.raises(ValueError):
        visualizer.visualize_defense_territories()
    
    plt.close('all')


def test_visualize_exploration_exploitation(visualizer_with_data):
    """Test the visualize_exploration_exploitation method."""
    # Test with default parameters
    fig = visualizer_with_data.visualize_exploration_exploitation()
    
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualizer_with_data.visualize_exploration_exploitation(
        sample_iterations=[0, 5, 10, 15, 20, 25],
        figsize=(15, 10)
    )
    
    assert isinstance(fig, Figure)
    
    # Test with empty data
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.visualize_exploration_exploitation()
    
    # Test without bounds
    visualizer = CPOVisualizer()
    visualizer.position_history = [np.random.random((10, 2))]
    visualizer.best_position_history = [np.random.random(2)]
    
    with pytest.raises(ValueError):
        visualizer.visualize_exploration_exploitation()
    
    plt.close('all')


def test_visualize_diversity_vs_convergence(visualizer_with_data):
    """Test the visualize_diversity_vs_convergence method."""
    # Test with default parameters
    fig = visualizer_with_data.visualize_diversity_vs_convergence(
        cycles=3,
        max_iter=30
    )
    
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualizer_with_data.visualize_diversity_vs_convergence(
        cycles=3,
        max_iter=30,
        title="Custom Diversity vs Convergence",
        figsize=(12, 6)
    )
    
    assert isinstance(fig, Figure)
    
    # Test with empty data
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.visualize_diversity_vs_convergence(cycles=3, max_iter=30)
    
    plt.close('all')


def test_visualize_defense_effectiveness(visualizer_with_data):
    """Test the visualize_defense_effectiveness method."""
    # Test with default parameters
    fig = visualizer_with_data.visualize_defense_effectiveness()
    
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualizer_with_data.visualize_defense_effectiveness(
        title="Custom Effectiveness Plot",
        figsize=(12, 8)
    )
    
    assert isinstance(fig, Figure)
    
    # Test with empty data
    visualizer = CPOVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.visualize_defense_effectiveness()
    
    plt.close('all')


def test_compare_reduction_strategies(visualizer_with_data):
    """Test the compare_reduction_strategies method."""
    # Test with default parameters
    fig = visualizer_with_data.compare_reduction_strategies(
        max_iter=30,
        pop_size=20,
        cycles=3
    )
    
    assert isinstance(fig, Figure)
    
    # Test with custom parameters
    fig = visualizer_with_data.compare_reduction_strategies(
        max_iter=30,
        pop_size=20,
        cycles=3,
        strategies=['linear', 'exponential'],
        figsize=(12, 6)
    )
    
    assert isinstance(fig, Figure)
    
    plt.close('all')


def test_create_parameter_tuning_dashboard(visualizer_with_data):
    """Test the create_parameter_tuning_dashboard method."""
    # Test with default parameters
    dashboard = visualizer_with_data.create_parameter_tuning_dashboard(
        parameter_name="Population Size",
        parameter_range=[10, 20, 30, 40, 50]
    )
    
    assert isinstance(dashboard, ParameterTuningDashboard)
    
    # Test with custom parameters
    dashboard = visualizer_with_data.create_parameter_tuning_dashboard(
        parameter_name="Population Size",
        parameter_range=[10, 20, 30, 40, 50],
        result_metric="Custom Metric",
        figsize=(12, 8)
    )
    
    assert isinstance(dashboard, ParameterTuningDashboard)
    
    # Clean up
    dashboard.close()
