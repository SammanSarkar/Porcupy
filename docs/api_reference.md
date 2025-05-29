# Porcupy API Reference

This document provides detailed information about the Porcupy library API.

## Core Classes

### Optimizer

The base class for all optimization algorithms in Porcupy.

```python
class Optimizer(abc.ABC):
    def __init__(
        self,
        dimensions: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 30,
        max_iter: int = 100,
        options: Optional[Dict[str, Any]] = None,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
    )
```

**Parameters:**

- `dimensions`: Number of dimensions in the search space.
- `bounds`: A tuple of size 2 where the first entry is the minimum bound while the second entry is the maximum bound. Each array must be of shape `(dimensions,)`.
- `pop_size`: Number of search agents in the population (default: 30).
- `max_iter`: Maximum number of iterations (default: 100).
- `options`: A dictionary containing algorithm-specific parameters.
- `ftol`: Relative error in objective_func(best_pos) acceptable for convergence (default: -np.inf).
- `ftol_iter`: Number of iterations over which the relative error in objective_func(best_pos) is acceptable for convergence (default: 1).

**Methods:**

- `optimize(objective_func, n_processes=None, verbose=False, **kwargs)`: Abstract method to optimize the objective function.
- `reset()`: Reset the attributes of the optimizer.

### CrestPorcupineOptimizer

The main implementation of the Crested Porcupine Optimizer algorithm.

```python
class CrestPorcupineOptimizer(Optimizer):
    def __init__(
        self,
        dimensions: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 30,
        min_pop_size: Optional[int] = None,
        max_iter: int = 100,
        cycles: int = 2,
        alpha: float = 0.2,
        tf: float = 0.8,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
    )
```

**Parameters:**

- `dimensions`: Number of dimensions in the search space.
- `bounds`: A tuple of size 2 where the first entry is the minimum bound while the second entry is the maximum bound. Each array must be of shape `(dimensions,)`.
- `pop_size`: Number of search agents (porcupines) in the initial population (default: 30).
- `min_pop_size`: Minimum population size during reduction cycles (default: pop_size // 2).
- `max_iter`: Maximum number of iterations (default: 100).
- `cycles`: Number of cycles for population reduction (default: 2).
- `alpha`: Convergence rate for fourth defense mechanism (default: 0.2).
- `tf`: Tradeoff threshold between third and fourth mechanisms (default: 0.8).
- `ftol`: Relative error in objective_func(best_pos) acceptable for convergence (default: -np.inf).
- `ftol_iter`: Number of iterations over which the relative error in objective_func(best_pos) is acceptable for convergence (default: 1).

**Methods:**

- `optimize(objective_func, f_ieqcons=None, n_processes=None, verbose=False, **kwargs)`: Optimize the objective function using Crested Porcupine Optimizer.

## Backend Components

### PorcupinePopulation

Class representing a population of porcupines for the Crested Porcupine Optimizer.

```python
class PorcupinePopulation:
    def __init__(
        self,
        pop_size: int,
        dimensions: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        init_pos: Optional[np.ndarray] = None
    )
```

**Parameters:**

- `pop_size`: Number of porcupines in the population.
- `dimensions`: Number of dimensions in the search space.
- `bounds`: A tuple of size 2 where the first entry is the minimum bound while the second entry is the maximum bound. Each array must be of shape `(dimensions,)`.
- `init_pos`: Initial positions for the porcupines. If None, random positions are generated.

**Methods:**

- `evaluate(objective_func, **kwargs)`: Evaluate the fitness of all porcupines.
- `resize(new_size)`: Resize the population to a new size.
- `apply_bounds()`: Apply boundary constraints to all positions.

### DefenseMechanisms

Class implementing the four defense mechanisms of the Crested Porcupine Optimizer.

```python
class DefenseMechanisms:
    def __init__(self, alpha: float = 0.2, tf: float = 0.8)
```

**Parameters:**

- `alpha`: Convergence rate for the fourth defense mechanism.
- `tf`: Tradeoff threshold between third and fourth mechanisms.

**Methods:**

- `sight_defense(position, other_position, best_position)`: Apply the first defense mechanism (sight).
- `sound_defense(position, other_position, rand_diff)`: Apply the second defense mechanism (sound).
- `odor_defense(position, other_position, rand_diff, fitness, fitness_sum, t, max_iter)`: Apply the third defense mechanism (odor).
- `physical_attack(position, other_position, best_position, fitness, fitness_sum, t, max_iter)`: Apply the fourth defense mechanism (physical attack).

### PopulationManager

Class for managing the population dynamics in the Crested Porcupine Optimizer.

```python
class PopulationManager:
    def __init__(
        self,
        initial_pop_size: int,
        min_pop_size: int,
        max_iter: int,
        cycles: int = 2
    )
```

**Parameters:**

- `initial_pop_size`: Initial population size.
- `min_pop_size`: Minimum population size during reduction.
- `max_iter`: Maximum number of iterations.
- `cycles`: Number of cycles for population reduction.

**Methods:**

- `calculate_pop_size(iteration)`: Calculate the population size for the current iteration.

## Population Management

### PopulationCycle

Class for managing cyclic population reduction in the Crested Porcupine Optimizer.

```python
class PopulationCycle:
    def __init__(
        self,
        initial_pop_size: int,
        min_pop_size: int,
        max_iter: int,
        cycles: int = 2,
        reduction_strategy: str = 'linear'
    )
```

**Parameters:**

- `initial_pop_size`: Initial population size.
- `min_pop_size`: Minimum population size during reduction.
- `max_iter`: Maximum number of iterations.
- `cycles`: Number of cycles for population reduction.
- `reduction_strategy`: Strategy for population reduction ('linear', 'cosine', or 'exponential').

**Methods:**

- `calculate_pop_size(iteration)`: Calculate the population size for the current iteration.

### SelectionStrategies

Class implementing various selection strategies for population reduction.

**Methods:**

- `best_selection(positions, fitness, new_size)`: Select the best individuals based on fitness.
- `tournament_selection(positions, fitness, new_size, tournament_size=3)`: Select individuals using tournament selection.
- `roulette_wheel_selection(positions, fitness, new_size)`: Select individuals using roulette wheel selection.
- `diversity_selection(positions, fitness, new_size, elite_fraction=0.2)`: Select individuals based on fitness and diversity.

## Visualization

### Functions

- `plot_convergence(cost_history, title="Convergence Curve", xlabel="Iterations", ylabel="Cost", figsize=(10, 6), log_scale=False, save_path=None)`: Plot the convergence history of the optimization process.
- `plot_population_size(pop_size_history, title="Population Size History", xlabel="Iterations", ylabel="Population Size", figsize=(10, 6), save_path=None)`: Plot the population size history of the optimization process.
- `plot_2d_search_space(func, bounds, resolution=100, positions=None, best_pos=None, title="2D Search Space", figsize=(10, 8), cmap='viridis', contour_levels=20, save_path=None)`: Plot a 2D search space with positions of the porcupines.
- `animate_optimization_2d(position_history, func, bounds, best_pos_history=None, interval=200, figsize=(10, 8), cmap='viridis', contour_levels=20, save_path=None, dpi=100)`: Create an animation of the optimization process in 2D.
- `plot_multiple_runs(cost_histories, labels=None, title="Comparison of Multiple Runs", xlabel="Iterations", ylabel="Cost", figsize=(10, 6), log_scale=False, save_path=None)`: Plot the convergence histories of multiple optimization runs.
- `plot_parameter_sensitivity(parameter_values, results, parameter_name, result_name="Best Cost", title=None, figsize=(10, 6), save_path=None)`: Plot the sensitivity of results to a parameter.

## Benchmark Functions

### Unimodal Functions

- `sphere(x)`: Sphere function.
- `rosenbrock(x)`: Rosenbrock function.
- `schwefel_2_22(x)`: Schwefel 2.22 function.
- `schwefel_1_2(x)`: Schwefel 1.2 function.
- `schwefel_2_21(x)`: Schwefel 2.21 function.
- `step(x)`: Step function.
- `quartic(x)`: Quartic function with noise.

### Multimodal Functions

- `rastrigin(x)`: Rastrigin function.
- `ackley(x)`: Ackley function.
- `griewank(x)`: Griewank function.
- `schwefel(x)`: Schwefel function.
- `michalewicz(x)`: Michalewicz function.

### Utility Functions

- `get_function_by_name(name)`: Get a benchmark function by name.
- `get_function_bounds(name, dimensions)`: Get the recommended bounds for a benchmark function.
- `get_function_optimum(name, dimensions)`: Get the global optimum for a benchmark function.

## Legacy API

### cpo

The original procedural implementation of the Crested Porcupine Optimizer.

```python
def cpo(fobj, lb, ub, pop_size=30, max_iter=100, f_ieqcons=None, verbose=False)
```

**Parameters:**

- `fobj`: Objective function to minimize. Takes a 1D numpy array as input and returns a scalar.
- `lb`: Lower bounds for each dimension of the search space.
- `ub`: Upper bounds for each dimension of the search space.
- `pop_size`: Number of search agents (porcupines) in the initial population (default: 30).
- `max_iter`: Maximum number of iterations (default: 100).
- `f_ieqcons`: Constraint function returning a 1D array of inequality constraints (g(x) >= 0). Infeasible solutions are assigned infinite fitness (default: None).
- `verbose`: If True, print progress information for each iteration (default: False).

**Returns:**

- `best_pos`: Best solution found (1D array of length `dim`).
- `best_cost`: Best fitness value found.
- `cost_history`: Best fitness value recorded at each iteration (1D array of length `max_iter`).
