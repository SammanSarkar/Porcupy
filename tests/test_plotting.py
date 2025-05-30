"""
Tests for the plotting functions in porcupy.utils.plotting module.
"""

import numpy as np
import pytest
import os
import matplotlib.pyplot as plt
from porcupy.utils.plotting import plot_convergence


class TestPlotting:
    """Test class for plotting functions."""

    def test_plot_convergence(self, monkeypatch, tmp_path):
        """Test the convergence plotting function."""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Create sample cost history
        cost_history = np.array([100, 80, 60, 40, 30, 25, 22, 20, 19, 18])
        
        # Test without saving
        plot_convergence(cost_history, title="Test Convergence")
        
        # Test with saving
        save_path = os.path.join(tmp_path, "test_convergence.png")
        plot_convergence(cost_history, title="Test Convergence", save_path=save_path)
        
        # Check if file was created
        assert os.path.exists(save_path)
    
    def test_plot_convergence_empty_history(self, monkeypatch):
        """Test the convergence plotting function with empty history."""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Create empty cost history
        cost_history = np.array([])
        
        # Should not raise an error but just create an empty plot
        with pytest.raises(ValueError):
            plot_convergence(cost_history, title="Empty Convergence")
    
    def test_plot_convergence_single_value(self, monkeypatch):
        """Test the convergence plotting function with a single value."""
        # Mock plt.show to avoid displaying plots during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Create cost history with a single value
        cost_history = np.array([100])
        
        # Should create a plot with a single point
        plot_convergence(cost_history, title="Single Value Convergence")
    
    def test_plot_convergence_invalid_input(self):
        """Test the convergence plotting function with invalid input."""
        # Test with non-numeric input
        with pytest.raises(TypeError):
            plot_convergence("not a numeric array", title="Invalid Input")
        
        # Test with 2D array
        with pytest.raises(ValueError):
            plot_convergence(np.array([[1, 2], [3, 4]]), title="2D Array Input")
