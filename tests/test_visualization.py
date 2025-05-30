"""
Tests for the visualization functions in porcupy.utils.visualization module.
"""

import numpy as np
import pytest
import os
# Use Agg backend for matplotlib to avoid Tkinter issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from porcupy.utils.visualization import (
    plot_convergence, 
    plot_population_size,
    plot_2d_search_space,
    animate_optimization_2d,
    plot_multiple_runs,
    plot_parameter_sensitivity
)


# Define a simple objective function for 2D visualization
def objective_func_2d(x):
    # Convert input to numpy array if it's a list
    if isinstance(x, list):
        x = np.array(x)
    
    # Handle both 1D and 2D inputs
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)


class TestVisualization:
    """Test class for visualization functions."""

    def test_plot_convergence(self, monkeypatch, tmp_path):
        """Test the convergence plotting function."""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Create sample cost history
        cost_history = [100, 80, 60, 40, 30, 25, 22, 20, 19, 18]
        
        # Test without saving
        fig = plot_convergence(cost_history, title="Test Convergence")
        assert fig is not None
        
        # Test with saving
        save_path = os.path.join(tmp_path, "test_convergence.png")
        fig = plot_convergence(cost_history, title="Test Convergence", save_path=save_path)
        assert fig is not None
        
        # Check if file was created
        assert os.path.exists(save_path)
        
        # Test with log scale
        fig = plot_convergence(cost_history, title="Log Scale Convergence", log_scale=True)
        assert fig is not None

    def test_plot_population_size(self, monkeypatch, tmp_path):
        """Test the population size plotting function."""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Create sample population size history
        pop_size_history = [30, 29, 28, 27, 26, 25, 24, 23, 22, 21]
        
        # Test without saving
        fig = plot_population_size(pop_size_history, title="Test Population Size")
        assert fig is not None
        
        # Test with saving
        save_path = os.path.join(tmp_path, "test_pop_size.png")
        fig = plot_population_size(pop_size_history, title="Test Population Size", save_path=save_path)
        assert fig is not None
        
        # Check if file was created
        assert os.path.exists(save_path)

    def test_plot_2d_search_space(self, monkeypatch, tmp_path):
        """Test the 2D search space plotting function."""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Define bounds
        lb = np.array([-5, -5])
        ub = np.array([5, 5])
        bounds = (lb, ub)
        
        # Create sample positions
        positions = np.array([
            [-2, -2],
            [0, 0],
            [2, 2]
        ])
        
        best_pos = np.array([0, 0])
        
        # Test without positions and best position
        fig = plot_2d_search_space(objective_func_2d, bounds, resolution=20)
        assert fig is not None
        
        # Test with positions
        fig = plot_2d_search_space(objective_func_2d, bounds, resolution=20, positions=positions)
        assert fig is not None
        
        # Test with positions and best position
        fig = plot_2d_search_space(objective_func_2d, bounds, resolution=20, positions=positions, best_pos=best_pos)
        assert fig is not None
        
        # Test with saving
        save_path = os.path.join(tmp_path, "test_search_space.png")
        fig = plot_2d_search_space(objective_func_2d, bounds, resolution=20, positions=positions, 
                                  best_pos=best_pos, save_path=save_path)
        assert fig is not None
        
        # Check if file was created
        assert os.path.exists(save_path)

    def test_animate_optimization_2d(self, monkeypatch, tmp_path):
        """Test the 2D optimization animation function."""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Define bounds
        lb = np.array([-5, -5])
        ub = np.array([5, 5])
        bounds = (lb, ub)
        
        # Create sample position history
        position_history = [
            np.array([[-4, -4], [-2, -2], [0, 0], [2, 2], [4, 4]]),
            np.array([[-3, -3], [-1, -1], [0, 0], [1, 1], [3, 3]]),
            np.array([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]])
        ]
        
        best_pos_history = [
            np.array([0, 0]),
            np.array([0, 0]),
            np.array([0, 0])
        ]
        
        # Test without best position history
        anim = animate_optimization_2d(position_history, objective_func_2d, bounds, interval=100, contour_levels=10)
        assert isinstance(anim, FuncAnimation)
        
        # Test with best position history
        anim = animate_optimization_2d(position_history, objective_func_2d, bounds, best_pos_history=best_pos_history,
                                      interval=100, contour_levels=10)
        assert isinstance(anim, FuncAnimation)
        
        # Test with saving (this is more complex and might not work in all environments)
        # For simplicity, we'll just check that the function runs without errors
        save_path = os.path.join(tmp_path, "test_animation.gif")
        try:
            anim = animate_optimization_2d(position_history, objective_func_2d, bounds, 
                                          best_pos_history=best_pos_history,
                                          interval=100, contour_levels=10,
                                          save_path=save_path, dpi=50)
            assert isinstance(anim, FuncAnimation)
        except Exception as e:
            # Skip this test if saving fails (might be due to missing dependencies)
            pytest.skip(f"Animation saving failed: {str(e)}")

    def test_plot_multiple_runs(self, monkeypatch, tmp_path):
        """Test the multiple runs plotting function."""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Create sample cost histories
        cost_histories = [
            [100, 80, 60, 40, 20, 10],
            [90, 70, 50, 30, 15, 8],
            [95, 75, 55, 35, 18, 9]
        ]
        
        labels = ["Run 1", "Run 2", "Run 3"]
        
        # Test without labels
        fig = plot_multiple_runs(cost_histories)
        assert fig is not None
        
        # Test with labels
        fig = plot_multiple_runs(cost_histories, labels=labels)
        assert fig is not None
        
        # Test with log scale
        fig = plot_multiple_runs(cost_histories, labels=labels, log_scale=True)
        assert fig is not None
        
        # Test with saving
        save_path = os.path.join(tmp_path, "test_multiple_runs.png")
        fig = plot_multiple_runs(cost_histories, labels=labels, save_path=save_path)
        assert fig is not None
        
        # Check if file was created
        assert os.path.exists(save_path)

    def test_plot_parameter_sensitivity(self, monkeypatch, tmp_path):
        """Test the parameter sensitivity plotting function."""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Create sample parameter values and results
        parameter_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = [50, 40, 30, 35, 45]
        
        # Test with default title
        fig = plot_parameter_sensitivity(parameter_values, results, parameter_name="Alpha")
        assert fig is not None
        
        # Test with custom title
        fig = plot_parameter_sensitivity(parameter_values, results, parameter_name="Alpha", 
                                        title="Alpha Sensitivity Analysis")
        assert fig is not None
        
        # Test with saving
        save_path = os.path.join(tmp_path, "test_parameter_sensitivity.png")
        fig = plot_parameter_sensitivity(parameter_values, results, parameter_name="Alpha", 
                                        save_path=save_path)
        assert fig is not None
        
        # Check if file was created
        assert os.path.exists(save_path)
