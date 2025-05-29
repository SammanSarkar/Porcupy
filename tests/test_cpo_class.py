import numpy as np
import pytest
from porcupy import CrestPorcupineOptimizer
from porcupy.functions import sphere, rosenbrock, rastrigin


class TestCrestPorcupineOptimizer:
    """Test suite for the CrestPorcupineOptimizer class."""
    
    def test_initialization(self):
        """Test that the optimizer initializes correctly."""
        dimensions = 5
        lb = np.full(dimensions, -10)
        ub = np.full(dimensions, 10)
        
        optimizer = CrestPorcupineOptimizer(
            dimensions=dimensions,
            bounds=(lb, ub),
            pop_size=30,
            max_iter=100
        )
        
        assert optimizer.dimensions == dimensions
        assert optimizer.pop_size == 30
        assert optimizer.max_iter == 100
        assert np.array_equal(optimizer.bounds[0], lb)
        assert np.array_equal(optimizer.bounds[1], ub)
        assert optimizer.min_pop_size == max(10, 30 // 2)
        assert optimizer.cycles == 2
        assert optimizer.alpha == 0.2
        assert optimizer.tf == 0.8
    
    def test_invalid_initialization(self):
        """Test that invalid initialization parameters raise appropriate errors."""
        dimensions = 5
        lb = np.full(dimensions, -10)
        ub = np.full(dimensions, 10)
        
        # Test invalid dimensions
        with pytest.raises(ValueError):
            CrestPorcupineOptimizer(
                dimensions=0,
                bounds=(lb, ub),
                pop_size=30,
                max_iter=100
            )
        
        # Test invalid pop_size
        with pytest.raises(ValueError):
            CrestPorcupineOptimizer(
                dimensions=dimensions,
                bounds=(lb, ub),
                pop_size=0,
                max_iter=100
            )
        
        # Test invalid max_iter
        with pytest.raises(ValueError):
            CrestPorcupineOptimizer(
                dimensions=dimensions,
                bounds=(lb, ub),
                pop_size=30,
                max_iter=0
            )
        
        # Test invalid ftol_iter
        with pytest.raises(ValueError):
            CrestPorcupineOptimizer(
                dimensions=dimensions,
                bounds=(lb, ub),
                pop_size=30,
                max_iter=100,
                ftol_iter=0
            )
    
    def test_reset(self):
        """Test that the reset method properly reinitializes the optimizer."""
        dimensions = 5
        lb = np.full(dimensions, -10)
        ub = np.full(dimensions, 10)
        
        optimizer = CrestPorcupineOptimizer(
            dimensions=dimensions,
            bounds=(lb, ub),
            pop_size=30,
            max_iter=100
        )
        
        # Add some history
        optimizer.cost_history = [1, 2, 3]
        optimizer.mean_cost_history = [4, 5, 6]
        optimizer.position_history = [np.ones((30, dimensions))]
        optimizer.pop_size_history = [30]
        optimizer.best_cost = 1.0
        optimizer.best_pos = np.ones(dimensions)
        
        # Reset the optimizer
        optimizer.reset()
        
        # Check that history is cleared
        assert len(optimizer.cost_history) == 0
        assert len(optimizer.mean_cost_history) == 0
        assert len(optimizer.position_history) == 0
        assert len(optimizer.pop_size_history) == 0
        assert optimizer.best_cost == np.inf
        assert optimizer.best_pos is None
    
    def test_optimize_sphere(self):
        """Test optimization of the sphere function."""
        dimensions = 5
        lb = np.full(dimensions, -10)
        ub = np.full(dimensions, 10)
        
        optimizer = CrestPorcupineOptimizer(
            dimensions=dimensions,
            bounds=(lb, ub),
            pop_size=20,
            max_iter=50
        )
        
        best_pos, best_cost, cost_history = optimizer.optimize(
            objective_func=sphere,
            verbose=False
        )
        
        # Check that optimization improved the solution
        assert best_cost < sphere(np.random.uniform(lb, ub))
        assert len(cost_history) == 50
        assert np.all(best_pos >= lb) and np.all(best_pos <= ub)
    
    def test_optimize_rosenbrock(self):
        """Test optimization of the rosenbrock function."""
        dimensions = 5
        lb = np.full(dimensions, -10)
        ub = np.full(dimensions, 10)
        
        optimizer = CrestPorcupineOptimizer(
            dimensions=dimensions,
            bounds=(lb, ub),
            pop_size=20,
            max_iter=50
        )
        
        best_pos, best_cost, cost_history = optimizer.optimize(
            objective_func=rosenbrock,
            verbose=False
        )
        
        # Check that optimization improved the solution
        assert best_cost < rosenbrock(np.random.uniform(lb, ub))
        assert len(cost_history) == 50
        assert np.all(best_pos >= lb) and np.all(best_pos <= ub)
    
    def test_optimize_rastrigin(self):
        """Test optimization of the rastrigin function."""
        dimensions = 5
        lb = np.full(dimensions, -5.12)
        ub = np.full(dimensions, 5.12)
        
        optimizer = CrestPorcupineOptimizer(
            dimensions=dimensions,
            bounds=(lb, ub),
            pop_size=20,
            max_iter=50
        )
        
        best_pos, best_cost, cost_history = optimizer.optimize(
            objective_func=rastrigin,
            verbose=False
        )
        
        # Check that optimization improved the solution
        assert best_cost < rastrigin(np.random.uniform(lb, ub))
        assert len(cost_history) == 50
        assert np.all(best_pos >= lb) and np.all(best_pos <= ub)
    
    def test_constraint_handling(self):
        """Test that constraints are properly handled."""
        dimensions = 2
        lb = np.array([-100, -100])
        ub = np.array([100, 100])
        
        # Define a constraint function
        # g(x) >= 0 for feasible solutions
        def constraint(x):
            # Constraint: x[0]^2 + x[1]^2 <= 50^2
            # Rewritten as: 50^2 - x[0]^2 - x[1]^2 >= 0
            return [2500 - x[0]**2 - x[1]**2]
        
        optimizer = CrestPorcupineOptimizer(
            dimensions=dimensions,
            bounds=(lb, ub),
            pop_size=20,
            max_iter=50
        )
        
        best_pos, best_cost, cost_history = optimizer.optimize(
            objective_func=sphere,
            f_ieqcons=constraint,
            verbose=False
        )
        
        # Check that the constraint is satisfied
        assert constraint(best_pos)[0] >= 0
        assert np.linalg.norm(best_pos) <= 50
    
    def test_population_reduction(self):
        """Test that population size is reduced during optimization."""
        dimensions = 5
        lb = np.full(dimensions, -10)
        ub = np.full(dimensions, 10)
        
        optimizer = CrestPorcupineOptimizer(
            dimensions=dimensions,
            bounds=(lb, ub),
            pop_size=30,
            min_pop_size=10,
            max_iter=50,
            cycles=2
        )
        
        best_pos, best_cost, cost_history = optimizer.optimize(
            objective_func=sphere,
            verbose=False
        )
        
        # Check that population size varies during optimization
        assert len(optimizer.pop_size_history) == 50
        assert min(optimizer.pop_size_history) < 30
        assert max(optimizer.pop_size_history) == 30
    
    def test_convergence_criteria(self):
        """Test that optimization stops when convergence criteria are met."""
        dimensions = 5
        lb = np.full(dimensions, -10)
        ub = np.full(dimensions, 10)
        
        optimizer = CrestPorcupineOptimizer(
            dimensions=dimensions,
            bounds=(lb, ub),
            pop_size=20,
            max_iter=1000,  # Large max_iter
            ftol=1e-6,      # Tight tolerance
            ftol_iter=5     # Need 5 consecutive iterations within tolerance
        )
        
        best_pos, best_cost, cost_history = optimizer.optimize(
            objective_func=sphere,
            verbose=False
        )
        
        # Check that optimization stopped before max_iter
        assert len(cost_history) < 1000
