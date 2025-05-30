"""
Tests for the interactive visualization module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from porcupy.utils.interactive_visualization import (
    OptimizationDashboard,
    ParameterTuningDashboard,
    create_interactive_optimization_plot
)


def simple_objective_func(x):
    """Simple test objective function."""
    if isinstance(x, list):
        x = np.array(x)
    
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


class TestOptimizationDashboard:
    """Tests for the OptimizationDashboard class."""
    
    def test_init(self):
        """Test initialization of the dashboard."""
        # Create required parameters
        objective_func = simple_objective_func
        bounds = (np.array([-5, -5]), np.array([5, 5]))
        
        dashboard = OptimizationDashboard(objective_func, bounds, figsize=(10, 8))
        assert dashboard is not None
        assert dashboard.figsize == (10, 8)
        assert dashboard.objective_func == objective_func
        assert np.array_equal(dashboard.bounds[0], bounds[0])
        assert np.array_equal(dashboard.bounds[1], bounds[1])
        
        # Clean up
        plt.close(dashboard.fig)
    
    def test_update(self):
        """Test updating the dashboard with new optimization data."""
        # Create required parameters
        objective_func = simple_objective_func
        bounds = (np.array([-5, -5]), np.array([5, 5]))
        
        dashboard = OptimizationDashboard(objective_func, bounds, figsize=(10, 8))
        
        # Create test data
        iteration = 1
        best_cost = 10.0
        pop_size = 50
        positions = np.array([[1, 2], [3, 4], [5, 6]])
        best_position = np.array([0, 0])
        defense_types = ['sight', 'sound', 'odor', 'physical']
        
        # Update the dashboard
        dashboard.update(
            iteration=iteration,
            best_cost=best_cost,
            pop_size=pop_size,
            positions=positions,
            best_position=best_position,
            defense_types=defense_types
        )
        
        # Check that the data was stored correctly
        assert dashboard.current_iteration == iteration
        assert dashboard.iterations == [iteration]
        assert dashboard.best_costs == [best_cost]
        assert dashboard.pop_sizes == [pop_size]
        assert len(dashboard.diversity_history) == 1  # Diversity is calculated internally
        
        # Check defense counts
        for defense in dashboard.defense_counts:
            assert len(dashboard.defense_counts[defense]) == 1
        
        # Check position history
        assert len(dashboard.position_history) == 1
        assert np.array_equal(dashboard.position_history[0], positions)
        assert len(dashboard.best_position_history) == 1
        assert np.array_equal(dashboard.best_position_history[0], best_position)
        
        # Clean up
        plt.close(dashboard.fig)
    
    def test_save_dashboard(self, tmp_path):
        """Test saving the dashboard."""
        # Create required parameters
        objective_func = simple_objective_func
        bounds = (np.array([-5, -5]), np.array([5, 5]))
        
        dashboard = OptimizationDashboard(objective_func, bounds, figsize=(8, 6))
        
        # Add some data
        dashboard.update(
            iteration=1,
            best_cost=10.0,
            pop_size=50,
            positions=np.array([[1, 2], [3, 4], [5, 6]]),
            best_position=np.array([0, 0]),
            defense_types=['sight', 'sound', 'odor', 'physical']
        )
        
        # Save to a temporary file
        save_path = os.path.join(tmp_path, "test_dashboard.png")
        dashboard.fig.savefig(save_path, dpi=72)
        
        # Check that the file exists
        assert os.path.exists(save_path)
        
        # Clean up
        plt.close(dashboard.fig)


class TestParameterTuningDashboard:
    """Tests for the ParameterTuningDashboard class."""
    
    def test_init(self):
        """Test initialization of the dashboard."""
        dashboard = ParameterTuningDashboard(
            parameter_name="Population Size",
            parameter_range=[10, 20, 30, 40, 50],
            result_metric="Best Cost",
            figsize=(10, 8)
        )
        
        assert dashboard is not None
        assert dashboard.parameter_name == "Population Size"
        assert dashboard.parameter_range == [10, 20, 30, 40, 50]
        assert dashboard.result_metric == "Best Cost"
        assert dashboard.figsize == (10, 8)
        
        # Clean up
        plt.close(dashboard.fig)
    
    def test_update(self):
        """Test updating the dashboard with new results."""
        dashboard = ParameterTuningDashboard(
            parameter_name="Population Size",
            parameter_range=[10, 20, 30],
            result_metric="Best Cost"
        )
        
        # Update with some test data
        dashboard.update(parameter_value=10, result=5.0, 
                        convergence_history=[10.0, 8.0, 7.0, 6.0, 5.0])
        dashboard.update(parameter_value=20, result=4.0,
                        convergence_history=[9.0, 7.0, 6.0, 5.0, 4.0])
        
        assert len(dashboard.results) == 2
        # Results are stored as tuples (parameter_value, result, convergence_history)
        assert dashboard.results[0][0] == 10  # parameter_value
        assert dashboard.results[0][1] == 5.0  # result
        assert dashboard.results[1][0] == 20  # parameter_value
        assert dashboard.results[1][1] == 4.0  # result
        
        # Clean up
        plt.close(dashboard.fig)
    
    def test_save_dashboard(self, tmp_path):
        """Test saving the dashboard."""
        dashboard = ParameterTuningDashboard(
            parameter_name="Population Size",
            parameter_range=[10, 20, 30],
            result_metric="Best Cost"
        )
        
        # Add some data
        dashboard.update(parameter_value=10, result=5.0)
        
        # Save to a temporary file
        save_path = os.path.join(tmp_path, "test_tuning_dashboard.png")
        dashboard.save_dashboard(save_path, dpi=72)
        
        # Check that the file exists
        assert os.path.exists(save_path)
        
        # Clean up
        plt.close(dashboard.fig)


def test_create_interactive_optimization_plot():
    """Test creating an interactive optimization plot."""
    # Create test data
    bounds = (np.array([-5, -5]), np.array([5, 5]))
    initial_positions = np.array([[-1, -1], [0, 0], [1, 1]])
    
    # Create the plot
    fig, ax, scatter = create_interactive_optimization_plot(
        simple_objective_func, bounds, initial_positions, figsize=(8, 6)
    )
    
    assert fig is not None
    assert ax is not None
    assert scatter is not None
    
    # Clean up
    plt.close(fig)
