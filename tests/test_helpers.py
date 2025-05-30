"""
Tests for the helper functions in porcupy.utils.helpers module.
"""

import numpy as np
import pytest
from porcupy.utils.helpers import initialize_population, clip_to_bounds


class TestHelpers:
    """Test class for helper functions."""

    def test_initialize_population_scalar_bounds(self):
        """Test population initialization with scalar bounds."""
        pop_size = 10
        dim = 3
        lb = -5
        ub = 5
        
        pop = initialize_population(pop_size, dim, lb, ub)
        
        assert pop.shape == (pop_size, dim)
        assert np.all(pop >= lb)
        assert np.all(pop <= ub)

    def test_initialize_population_array_bounds(self):
        """Test population initialization with array bounds."""
        pop_size = 10
        dim = 3
        lb_array = np.array([-5, -10, -15])
        ub_array = np.array([5, 10, 15])
        
        pop = initialize_population(pop_size, dim, lb_array, ub_array)
        
        assert pop.shape == (pop_size, dim)
        for i in range(dim):
            assert np.all(pop[:, i] >= lb_array[i])
            assert np.all(pop[:, i] <= ub_array[i])

    def test_initialize_population_negative_pop_size(self):
        """Test population initialization with negative population size."""
        with pytest.raises(ValueError):
            initialize_population(-5, 3, -5, 5)
    
    def test_initialize_population_zero_dimensions(self):
        """Test population initialization with zero dimensions."""
        with pytest.raises(ValueError):
            initialize_population(10, 0, -5, 5)
    
    def test_initialize_population_inverted_bounds(self):
        """Test population initialization with lower bound > upper bound."""
        with pytest.raises(ValueError):
            initialize_population(10, 3, 5, -5)
    
    def test_initialize_population_mismatched_bounds(self):
        """Test population initialization with mismatched bounds."""
        with pytest.raises(ValueError):
            lb_array = np.array([-5, -10])  # Only 2 dimensions
            ub_array = np.array([5, 10, 15])  # 3 dimensions
            initialize_population(10, 3, lb_array, ub_array)

    def test_clip_to_bounds_scalar(self):
        """Test position clipping with scalar bounds."""
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
        
        expected = np.array([
            [-5, 0, 5],
            [0, 5, -5],
            [3, -3, 5]
        ])
        np.testing.assert_array_equal(clipped, expected)
    
    def test_clip_to_bounds_array(self):
        """Test position clipping with array bounds."""
        positions = np.array([
            [-10, 0, 10],
            [0, 20, -20],
            [6, -12, 7]
        ])
        lb_array = np.array([-5, -10, -15])
        ub_array = np.array([5, 10, 15])
        
        clipped = clip_to_bounds(positions, lb_array, ub_array)
        
        assert clipped.shape == positions.shape
        for i in range(positions.shape[1]):
            assert np.all(clipped[:, i] >= lb_array[i])
            assert np.all(clipped[:, i] <= ub_array[i])
        
        expected = np.array([
            [-5, 0, 10],
            [0, 10, -15],
            [5, -10, 7]
        ])
        np.testing.assert_array_equal(clipped, expected)
