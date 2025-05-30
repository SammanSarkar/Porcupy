"""
Tests for the helper functions in porcupy.utils.helpers module.
"""

import numpy as np
import pytest
from porcupy.utils.helpers import initialize_population, clip_to_bounds


class TestHelpers:
    """Test class for helper functions."""

    def test_initialize_population(self):
        """Test population initialization with valid parameters."""
        # Test with scalar bounds
        pop_size = 10
        dim = 3
        lb = -5
        ub = 5
        
        pop = initialize_population(pop_size, dim, lb, ub)
        
        assert pop.shape == (pop_size, dim)
        assert np.all(pop >= lb)
        assert np.all(pop <= ub)
        
        # Test with array bounds
        lb_array = np.array([-5, -10, -15])
        ub_array = np.array([5, 10, 15])
        
        pop = initialize_population(pop_size, dim, lb_array, ub_array)
        
        assert pop.shape == (pop_size, dim)
        assert np.all(pop >= lb_array)
        assert np.all(pop <= ub_array)

    def test_initialize_population_invalid_params(self):
        """Test population initialization with invalid parameters."""
        # Test with negative population size
        with pytest.raises(ValueError):
            initialize_population(-5, 3, -5, 5)
        
        # Test with zero dimensions
        with pytest.raises(ValueError):
            initialize_population(10, 0, -5, 5)
        
        # Test with lower bound > upper bound
        with pytest.raises(ValueError):
            initialize_population(10, 3, 5, -5)
        
        # Test with incompatible array bounds
        with pytest.raises(ValueError):
            lb_array = np.array([-5, -10])  # Only 2 dimensions
            ub_array = np.array([5, 10, 15])  # 3 dimensions
            initialize_population(10, 3, lb_array, ub_array)

    def test_clip_to_bounds(self):
        """Test position clipping with valid parameters."""
        # Test with scalar bounds
        positions = np.array([
            [-10, 0, 10],
            [0, 20, -5],
            [3, -3, 7]
        ])
        lb = -5
        ub = 5
        
        clipped = clip_to_bounds(positions, lb, ub)
        
        assert clipped.shape == positions.shape
        assert np.all(clipped >= lb)
        assert np.all(clipped <= ub)
        assert np.array_equal(clipped, np.array([
            [-5, 0, 5],
            [0, 5, -5],
            [3, -3, 5]
        ]))
        
        # Test with array bounds
        lb_array = np.array([-5, -10, -15])
        ub_array = np.array([5, 10, 15])
        
        positions = np.array([
            [-10, 0, 10],
            [0, 20, -20],
            [6, -12, 7]
        ])
        
        clipped = clip_to_bounds(positions, lb_array, ub_array)
        
        assert clipped.shape == positions.shape
        assert np.all(clipped >= lb_array)
        assert np.all(clipped <= ub_array)
        assert np.array_equal(clipped, np.array([
            [-5, 0, 10],
            [0, 10, -15],
            [5, -10, 7]
        ]))

    def test_clip_to_bounds_invalid_params(self):
        """Test position clipping with invalid parameters."""
        positions = np.array([
            [-10, 0, 10],
            [0, 20, -5]
        ])
        
        # Test with incompatible array bounds
        with pytest.raises(ValueError):
            lb_array = np.array([-5, -10])  # Only 2 dimensions
            ub_array = np.array([5, 10, 15])  # 3 dimensions
            clip_to_bounds(positions, lb_array, ub_array)
