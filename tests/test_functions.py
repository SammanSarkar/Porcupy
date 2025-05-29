import pytest
import numpy as np
from porcupy.functions import sphere, rosenbrock, rastrigin

def test_sphere():
    """
    Test the sphere function.
    """
    assert sphere([0, 0, 0]) == 0, "Sphere function should be 0 at origin"
    assert sphere([1, 1]) == 2, "Sphere function incorrect for [1, 1]"
    assert sphere([-2, 3]) == 13, "Sphere function incorrect for [-2, 3]"

def test_rosenbrock():
    """
    Test the Rosenbrock function.
    """
    assert rosenbrock([1, 1]) == 0, "Rosenbrock function should be 0 at [1, 1]"
    assert abs(rosenbrock([0, 0]) - 1) < 1e-10, "Rosenbrock function incorrect for [0, 0]"

def test_rastrigin():
    """
    Test the Rastrigin function.
    """
    assert rastrigin([0, 0]) == 0, "Rastrigin function should be 0 at [0, 0]"
    assert abs(rastrigin([1, 1]) - 2) < 1e-10, "Rastrigin function incorrect for [1, 1]"

def test_cpo_rastrigin():
    """
    Test CPO on the Rastrigin function.
    """
    from porcupy.cpo import cpo
    lb = [-5, -5]
    ub = [5, 5]
    best_pos, best_cost, cost_history = cpo(rastrigin, lb, ub, pop_size=30, max_iter=100)
    
    assert len(best_pos) == 2, "Best position should have dimension 2"
    assert best_cost < 1.0, "Rastrigin function should converge near 0"
    assert len(cost_history) == 100, "Cost history should match max_iter"
    assert np.all(cost_history >= 0), "Cost history should be non-negative"