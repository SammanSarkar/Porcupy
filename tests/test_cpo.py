import pytest
import numpy as np
from porcupy.cpo import cpo
from porcupy.functions import sphere

def test_cpo_sphere():
    """
    Test CPO on the sphere function.
    """
    lb = [-5, -5]
    ub = [5, 5]
    best_pos, best_cost, cost_history = cpo(sphere, lb, ub, pop_size=30, max_iter=100)
    
    assert len(best_pos) == 2, "Best position should have dimension 2"
    assert best_cost < 0.1, "Sphere function should converge near 0"
    assert len(cost_history) == 100, "Cost history should match max_iter"
    assert np.all(cost_history >= 0), "Cost history should be non-negative"

def test_cpo_constraints():
    """
    Test CPO with inequality constraints.
    """
    def constraint(x):
        # Constraint: x[0] + x[1] >= 0
        return np.array([x[0] + x[1]])
    
    lb = [-5, -5]
    ub = [5, 5]
    best_pos, best_cost, _ = cpo(sphere, lb, ub, pop_size=30, max_iter=50, f_ieqcons=constraint)
    
    assert np.all(constraint(best_pos) >= 0), "Best position should satisfy constraints"

def test_cpo_invalid_inputs():
    """
    Test CPO with invalid inputs.
    """
    with pytest.raises(ValueError):
        cpo(sphere, lb=[-5], ub=[5, 5], pop_size=30, max_iter=100)  # Mismatched bounds
    with pytest.raises(ValueError):
        cpo(sphere, lb=[-5, -5], ub=[5, 5], pop_size=0, max_iter=100)  # Invalid pop_size
    with pytest.raises(ValueError):
        cpo("not a function", lb=[-5, -5], ub=[5, 5], pop_size=30, max_iter=100)  # Invalid fobj