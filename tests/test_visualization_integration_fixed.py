"""
Integration tests for visualization with the CPO optimizer.

This module contains tests for integrating visualization components with the CPO optimizer.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Add the parent directory to the path to ensure imports work
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy.cpo_class import CrestPorcupineOptimizer
from porcupy.utils.visualization_manager import CPOVisualizer
from porcupy.functions import sphere, rastrigin


@pytest.mark.integration
def test_visualization_with_cpo_optimizer():
    """Test visualization integration with the CPO optimizer."""
    # Define a simple 2D test function
    def func(x):
        return sphere(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -5 * np.ones(dimensions)
    ub = 5 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer
    optimizer = CrestPorcupineOptimizer(
        pop_size=20,
        dimensions=dimensions,
        max_iter=10,  # Use fewer iterations for testing
        bounds=bounds,
        cycles=2
    )
    
    # Initialize the visualizer
    visualizer = CPOVisualizer(objective_func=func, bounds=bounds)
    
    # Run the optimization without callback
    best_pos, best_cost, cost_history = optimizer.optimize(func)
    
    # Manually record visualization data for testing
    for i in range(10):  # Record 10 iterations for testing
        # Generate random positions and defense types for demonstration
        positions = np.random.uniform(lb, ub, (optimizer.pop_size, dimensions))
        defenses = [np.random.choice(['sight', 'sound', 'odor', 'physical']) for _ in range(optimizer.pop_size)]
        
        # Record the iteration data
        visualizer.record_iteration(
            iteration=i,
            positions=positions,
            best_position=best_pos,
            best_fitness=best_cost,
            pop_size=optimizer.pop_size,
            defense_types=defenses
        )
    
    # Check that data was recorded correctly
    assert len(visualizer.position_history) == 10
    assert len(visualizer.best_position_history) == 10
    assert len(visualizer.fitness_history) == 10
    assert len(visualizer.pop_size_history) == 10
    assert len(visualizer.diversity_history) == 10
    
    # Check defense history
    for defense in ['sight', 'sound', 'odor', 'physical']:
        assert defense in visualizer.defense_history
        assert len(visualizer.defense_history[defense]) == 10
    
    # Test creating visualizations
    fig1 = visualizer.visualize_defense_mechanisms()
    assert isinstance(fig1, Figure)
    
    fig2 = visualizer.visualize_population_cycles(cycles=2, max_iter=10)
    assert isinstance(fig2, Figure)
    
    fig3 = visualizer.visualize_diversity_history()
    assert isinstance(fig3, Figure)
    
    fig4 = visualizer.visualize_porcupines_2d()
    assert isinstance(fig4, Figure)
    
    plt.close('all')


@pytest.mark.integration
def test_visualization_with_complex_function():
    """Test visualization with a more complex function."""
    # Define a more complex 2D test function
    def func(x):
        return rastrigin(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -5.12 * np.ones(dimensions)
    ub = 5.12 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer
    optimizer = CrestPorcupineOptimizer(
        pop_size=30,
        dimensions=dimensions,
        max_iter=15,  # Use fewer iterations for testing
        bounds=bounds,
        cycles=3
    )
    
    # Initialize the visualizer
    visualizer = CPOVisualizer(objective_func=func, bounds=bounds)
    
    # Run the optimization without callback
    best_pos, best_cost, cost_history = optimizer.optimize(func)
    
    # Manually record visualization data for testing
    for i in range(15):  # Record 15 iterations for testing
        # Generate random positions and defense types for demonstration
        positions = np.random.uniform(lb, ub, (optimizer.pop_size, dimensions))
        defenses = [np.random.choice(['sight', 'sound', 'odor', 'physical']) for _ in range(optimizer.pop_size)]
        
        # Record the iteration data
        visualizer.record_iteration(
            iteration=i,
            positions=positions,
            best_position=best_pos,
            best_fitness=best_cost,
            pop_size=optimizer.pop_size,
            defense_types=defenses
        )
    
    # Test creating more advanced visualizations
    fig1 = visualizer.visualize_defense_territories()
    assert isinstance(fig1, Figure)
    
    fig2 = visualizer.visualize_exploration_exploitation(
        sample_iterations=[0, 5, 10, 14]
    )
    assert isinstance(fig2, Figure)
    
    fig3 = visualizer.visualize_diversity_vs_convergence(
        cycles=3,
        max_iter=15
    )
    assert isinstance(fig3, Figure)
    
    fig4 = visualizer.visualize_defense_effectiveness()
    assert isinstance(fig4, Figure)
    
    plt.close('all')
