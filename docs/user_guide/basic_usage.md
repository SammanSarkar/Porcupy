# Basic Usage

This guide covers the fundamental concepts and usage patterns of Porcupy.

## Core Concepts

### 1. Problem Definition

Before using Porcupy, you need to define your optimization problem:

- **Objective Function**: The function to minimize
- **Search Space**: Bounds for each dimension
- **Constraints** (optional): Any constraints on the solution

### 2. Population-Based Optimization

Porcupy maintains a population of candidate solutions (porcupines) that evolve over iterations:

- **Population Size**: Number of candidate solutions
- **Iterations**: Number of generations to evolve
- **Fitness**: Value of the objective function for each candidate

## Basic Optimization

### Minimizing a Simple Function

```python
import numpy as np
from porcupy import CPO

def sphere(x):
    return np.sum(x**2)

# Initialize optimizer
optimizer = CPO(
    dimensions=2,
    bounds=([-5, -5], [5, 5]),
    pop_size=30,
    max_iter=50
)

# Run optimization
best_solution, best_fitness, _ = optimizer.optimize(sphere)
```

### Working with Constraints

Porcupy supports both inequality and equality constraints:

```python
def constraint(x):
    return x[0] + x[1] - 1  # x + y >= 1

def objective(x):
    return x[0]**2 + x[1]**2

optimizer = CPO(dimensions=2, bounds=([0, 0], [1, 1]))
best_solution, best_fitness, _ = optimizer.optimize(
    objective_func=objective,
    f_ieqcons=lambda x: constraint(x)  # Inequality constraint
)
```

## Understanding the Optimization Process

### Monitoring Progress

You can monitor the optimization progress using the `verbose` parameter:

```python
optimizer = CPO(
    dimensions=10,
    bounds=([-5]*10, [5]*10),
    pop_size=50,
    max_iter=100,
    verbose=True  # Print progress
)
```

### Accessing Optimization History

When `track_history=True` (default), you can access the optimization history:

```python
best_solution, best_fitness, history = optimizer.optimize(sphere)

# Access history
best_fitness_history = history['best_fitness']
best_solutions = history['best_solutions']
population_history = history['population']
```

## Practical Tips

### Parameter Tuning

- **Population Size**: Start with 10-50 for simple problems, 100+ for complex ones
- **Iterations**: Start with 100-1000 depending on problem complexity
- **Bounds**: Set realistic bounds based on your problem domain

### Handling Noisy Functions

For noisy objective functions, consider:
- Increasing population size
- Running multiple optimizations
- Using the `ftol` parameter for early stopping

## Common Patterns

### Restarting Optimization

```python
# Continue optimization from previous best solution
optimizer.positions[0] = best_solution
best_solution, best_fitness, _ = optimizer.optimize(sphere)
```

### Using Custom Initial Population

```python
# Generate custom initial population
custom_pop = np.random.uniform(low=-5, high=5, size=(50, 10))
optimizer = CPO(dimensions=10, bounds=([-5]*10, [5]*10))
optimizer.initialize_population(custom_pop)
best_solution, best_fitness, _ = optimizer.optimize(sphere)
```

## Next Steps

- Learn about [GPU Acceleration](gpu_acceleration.md)
- Explore [Visualization](visualization.md) tools
- Check out [Advanced Features](advanced_features.md)
