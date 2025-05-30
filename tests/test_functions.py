import pytest
import numpy as np
from porcupy.functions import (
    # Unimodal functions
    sphere, rosenbrock, schwefel_2_22, schwefel_1_2, schwefel_2_21, step, quartic,
    # Multimodal functions
    rastrigin, ackley, griewank, schwefel, michalewicz,
    # Utility functions
    get_function_by_name, get_function_bounds, get_function_optimum
)

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

def test_schwefel_2_22():
    """
    Test the Schwefel 2.22 function.
    """
    assert schwefel_2_22([0, 0]) == 0, "Schwefel 2.22 function should be 0 at origin"
    assert schwefel_2_22([1, 1]) == 3, "Schwefel 2.22 function incorrect for [1, 1]"
    # For [-2, 3], sum(abs) = 5, prod(abs) = 6, total = 11
    assert schwefel_2_22([-2, 3]) == 11, "Schwefel 2.22 function incorrect for [-2, 3]"

def test_schwefel_1_2():
    """
    Test the Schwefel 1.2 function.
    """
    assert schwefel_1_2([0, 0]) == 0, "Schwefel 1.2 function should be 0 at origin"
    # For [1, 2]: (1)^2 + (1+2)^2 = 1 + 9 = 10
    assert schwefel_1_2([1, 2]) == 10, "Schwefel 1.2 function incorrect for [1, 2]"
    # For [-2, 3]: (-2)^2 + (-2+3)^2 = 4 + 1 = 5
    assert schwefel_1_2([-2, 3]) == 5, "Schwefel 1.2 function incorrect for [-2, 3]"

def test_schwefel_2_21():
    """
    Test the Schwefel 2.21 function.
    """
    assert schwefel_2_21([0, 0]) == 0, "Schwefel 2.21 function should be 0 at origin"
    assert schwefel_2_21([1, 2]) == 2, "Schwefel 2.21 function incorrect for [1, 2]"
    assert schwefel_2_21([-3, 2]) == 3, "Schwefel 2.21 function incorrect for [-3, 2]"

def test_step():
    """
    Test the Step function.
    """
    assert step([0, 0]) == 0, "Step function should be 0 at origin"
    assert step([0.4, -0.4]) == 0, "Step function incorrect for [0.4, -0.4]"
    # For [1.5, 2.7]: floor(1.5 + 0.5)^2 + floor(2.7 + 0.5)^2 = 2^2 + 3^2 = 4 + 9 = 13
    assert step([1.5, 2.7]) == 13, "Step function incorrect for [1.5, 2.7]"

def test_quartic():
    """
    Test the Quartic function.
    """
    # Since quartic has random component, we can only test basic properties
    result = quartic([0, 0])
    assert result >= 0, "Quartic function should be non-negative at origin"
    assert result < 1, "Quartic function at origin should be less than 1"
    
    # Test with non-zero inputs
    result = quartic([1, 2])
    assert result > 0, "Quartic function should be positive for non-zero inputs"

def test_ackley():
    """
    Test the Ackley function.
    """
    assert abs(ackley([0, 0])) < 1e-10, "Ackley function should be 0 at origin"
    assert ackley([1, 1]) > 0, "Ackley function should be positive for non-zero inputs"
    assert ackley([5, 5]) > ackley([1, 1]), "Ackley function should increase with distance from origin"

def test_griewank():
    """
    Test the Griewank function.
    """
    assert abs(griewank([0, 0])) < 1e-10, "Griewank function should be 0 at origin"
    assert griewank([1, 1]) > 0, "Griewank function should be positive for non-zero inputs"

def test_schwefel():
    """
    Test the Schwefel function.
    """
    # Test near the optimum
    x_opt = np.array([420.9687] * 2)
    assert abs(schwefel(x_opt)) < 1e-4, "Schwefel function should be near 0 at optimum"
    assert schwefel([0, 0]) > 0, "Schwefel function should be positive away from optimum"

def test_michalewicz():
    """
    Test the Michalewicz function.
    """
    assert michalewicz([0, 0]) == 0, "Michalewicz function should be 0 at origin"
    assert michalewicz([np.pi/2, np.pi/2]) < 0, "Michalewicz function should be negative at [pi/2, pi/2]"

def test_get_function_by_name():
    """
    Test the get_function_by_name utility function.
    """
    assert get_function_by_name("sphere") == sphere, "Should return sphere function"
    assert get_function_by_name("rosenbrock") == rosenbrock, "Should return rosenbrock function"
    assert get_function_by_name("rastrigin") == rastrigin, "Should return rastrigin function"
    
    # Test error handling
    with pytest.raises(ValueError):
        get_function_by_name("nonexistent_function")

def test_get_function_bounds():
    """
    Test the get_function_bounds utility function.
    """
    # Test for sphere function
    lb, ub = get_function_bounds("sphere", 3)
    assert lb.shape == (3,), "Lower bounds should have correct shape"
    assert ub.shape == (3,), "Upper bounds should have correct shape"
    assert np.all(lb == -100), "Lower bounds should be -100 for sphere"
    assert np.all(ub == 100), "Upper bounds should be 100 for sphere"
    
    # Test for schwefel function
    lb, ub = get_function_bounds("schwefel", 2)
    assert np.all(lb == -500), "Lower bounds should be -500 for schwefel"
    assert np.all(ub == 500), "Upper bounds should be 500 for schwefel"
    
    # Test error handling
    with pytest.raises(ValueError):
        get_function_bounds("nonexistent_function", 2)

def test_get_function_optimum():
    """
    Test the get_function_optimum utility function.
    """
    # Test for sphere function
    x_opt, f_opt = get_function_optimum("sphere", 3)
    assert x_opt.shape == (3,), "Optimal position should have correct shape"
    assert np.all(x_opt == 0), "Optimal position should be 0 for sphere"
    assert f_opt == 0, "Optimal value should be 0 for sphere"
    
    # Test for schwefel function
    x_opt, f_opt = get_function_optimum("schwefel", 2)
    assert np.allclose(x_opt, 420.9687), "Optimal position should be 420.9687 for schwefel"
    assert f_opt == 0, "Optimal value should be 0 for schwefel"
    
    # Test error handling
    with pytest.raises(ValueError):
        get_function_optimum("nonexistent_function", 2)

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