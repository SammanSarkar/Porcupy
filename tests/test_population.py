import numpy as np
import pytest
from porcupy.utils.population import PopulationCycle, SelectionStrategies


class TestPopulationCycle:
    """Test suite for the PopulationCycle class."""
    
    def test_initialization(self):
        """Test that the population cycle initializes correctly."""
        pop_cycle = PopulationCycle(
            initial_pop_size=50,
            min_pop_size=10,
            max_iter=100,
            cycles=2,
            reduction_strategy='linear'
        )
        
        assert pop_cycle.initial_pop_size == 50
        assert pop_cycle.min_pop_size == 10
        assert pop_cycle.max_iter == 100
        assert pop_cycle.cycles == 2
        assert pop_cycle.reduction_strategy == 'linear'
    
    def test_invalid_initialization(self):
        """Test that invalid initialization parameters raise appropriate errors."""
        # Test invalid initial_pop_size
        with pytest.raises(ValueError):
            PopulationCycle(
                initial_pop_size=0,
                min_pop_size=10,
                max_iter=100,
                cycles=2
            )
        
        # Test invalid min_pop_size
        with pytest.raises(ValueError):
            PopulationCycle(
                initial_pop_size=50,
                min_pop_size=0,
                max_iter=100,
                cycles=2
            )
        
        # Test min_pop_size > initial_pop_size
        with pytest.raises(ValueError):
            PopulationCycle(
                initial_pop_size=50,
                min_pop_size=60,
                max_iter=100,
                cycles=2
            )
        
        # Test invalid max_iter
        with pytest.raises(ValueError):
            PopulationCycle(
                initial_pop_size=50,
                min_pop_size=10,
                max_iter=0,
                cycles=2
            )
        
        # Test invalid cycles
        with pytest.raises(ValueError):
            PopulationCycle(
                initial_pop_size=50,
                min_pop_size=10,
                max_iter=100,
                cycles=0
            )
        
        # Test invalid reduction_strategy
        with pytest.raises(ValueError):
            PopulationCycle(
                initial_pop_size=50,
                min_pop_size=10,
                max_iter=100,
                cycles=2,
                reduction_strategy='invalid'
            )
    
    def test_linear_reduction(self):
        """Test that linear reduction works correctly."""
        pop_cycle = PopulationCycle(
            initial_pop_size=50,
            min_pop_size=10,
            max_iter=100,
            cycles=2,
            reduction_strategy='linear'
        )
        
        # Test at the beginning of a cycle
        pop_size = pop_cycle.calculate_pop_size(0)
        assert pop_size == 50
        
        # Test at the middle of a cycle
        pop_size = pop_cycle.calculate_pop_size(25)
        assert pop_size < 50 and pop_size > 10
        
        # Test at the end of a cycle
        pop_size = pop_cycle.calculate_pop_size(49)
        assert pop_size == 10
        
        # Test at the beginning of the second cycle
        pop_size = pop_cycle.calculate_pop_size(50)
        assert pop_size == 50
    
    def test_cosine_reduction(self):
        """Test that cosine reduction works correctly."""
        pop_cycle = PopulationCycle(
            initial_pop_size=50,
            min_pop_size=10,
            max_iter=100,
            cycles=2,
            reduction_strategy='cosine'
        )
        
        # Test at the beginning of a cycle
        pop_size = pop_cycle.calculate_pop_size(0)
        assert pop_size == 50
        
        # Test at the middle of a cycle
        pop_size = pop_cycle.calculate_pop_size(25)
        assert pop_size < 50 and pop_size > 10
        
        # Test at the end of a cycle
        pop_size = pop_cycle.calculate_pop_size(49)
        assert pop_size == 10
        
        # Test at the beginning of the second cycle
        pop_size = pop_cycle.calculate_pop_size(50)
        assert pop_size == 50
    
    def test_exponential_reduction(self):
        """Test that exponential reduction works correctly."""
        pop_cycle = PopulationCycle(
            initial_pop_size=50,
            min_pop_size=10,
            max_iter=100,
            cycles=2,
            reduction_strategy='exponential'
        )
        
        # Test at the beginning of a cycle
        pop_size = pop_cycle.calculate_pop_size(0)
        assert pop_size == 50
        
        # Test at the middle of a cycle
        pop_size = pop_cycle.calculate_pop_size(25)
        assert pop_size < 50 and pop_size > 10
        
        # Test at the end of a cycle
        pop_size = pop_cycle.calculate_pop_size(49)
        assert pop_size == 10
        
        # Test at the beginning of the second cycle
        pop_size = pop_cycle.calculate_pop_size(50)
        assert pop_size == 50


class TestSelectionStrategies:
    """Test suite for the SelectionStrategies class."""
    
    def setup_method(self):
        """Set up test data."""
        self.positions = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5]
        ])
        self.fitness = np.array([5, 4, 3, 2, 1])
    
    def test_best_selection(self):
        """Test that best selection works correctly."""
        new_positions, new_fitness = SelectionStrategies.best_selection(
            self.positions, self.fitness, 3
        )
        
        assert new_positions.shape == (3, 2)
        assert new_fitness.shape == (3,)
        assert np.array_equal(new_fitness, np.array([1, 2, 3]))
        assert np.array_equal(new_positions, np.array([[5, 5], [4, 4], [3, 3]]))
    
    def test_tournament_selection(self):
        """Test that tournament selection works correctly."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        new_positions, new_fitness = SelectionStrategies.tournament_selection(
            self.positions, self.fitness, 3, tournament_size=2
        )
        
        assert new_positions.shape == (3, 2)
        assert new_fitness.shape == (3,)
        
        # Check that selected individuals have fitness values from the original set
        for f in new_fitness:
            assert f in self.fitness
    
    def test_roulette_wheel_selection(self):
        """Test that roulette wheel selection works correctly."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        new_positions, new_fitness = SelectionStrategies.roulette_wheel_selection(
            self.positions, self.fitness, 3
        )
        
        assert new_positions.shape == (3, 2)
        assert new_fitness.shape == (3,)
        
        # Check that selected individuals have fitness values from the original set
        for f in new_fitness:
            assert f in self.fitness
    
    def test_diversity_selection(self):
        """Test that diversity selection works correctly."""
        new_positions, new_fitness = SelectionStrategies.diversity_selection(
            self.positions, self.fitness, 3, elite_fraction=0.33
        )
        
        assert new_positions.shape == (3, 2)
        assert new_fitness.shape == (3,)
        
        # Check that at least one of the selected individuals is the best
        assert min(new_fitness) == min(self.fitness)
        
        # Check that selected individuals have fitness values from the original set
        for f in new_fitness:
            assert f in self.fitness
