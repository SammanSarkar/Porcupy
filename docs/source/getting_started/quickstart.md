# Quick Start Guide

This guide will help you get started with Porcupy by walking you through a simple optimization example.

## Your First Optimization

Let's start with a basic example of optimizing the Rastrigin function, a common benchmark for optimization algorithms.

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import rastrigin

# Define the problem
dimensions = 5  # Number of dimensions
bounds = ([-5.12] * dimensions, [5.12] * dimensions)  # Search space bounds

# Initialize the optimizer
optimizer = CPO(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=50,     # Number of candidate solutions
    max_iter=100,    # Maximum number of iterations
    ftol=1e-6,       # Function tolerance for early stopping
    ftol_iter=10     # Number of iterations to check for convergence
)

# Run optimization
best_solution, best_fitness, history = optimizer.optimize(rastrigin)

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

## Understanding the Output

- `best_solution`: The best solution found (numpy array)
- `best_fitness`: The fitness value of the best solution
- `history`: Dictionary containing optimization history (if `track_history=True`)

## Visualizing the Results

You can visualize the optimization progress using the built-in plotting utilities:

```python
import matplotlib.pyplot as plt

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(history['best_fitness'])
plt.title('Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()
```

## Using GPU Acceleration

To use GPU acceleration, simply import `GPUCPO` instead of `CPO`:

```python
from porcupy import GPUCPO

# The rest of the code remains the same
optimizer = GPUCPO(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=50,
    max_iter=100
)
```

## Next Steps

- Explore more [examples](examples/basic_optimization.md)
- Learn about [advanced features](user_guide/advanced_features.md)
- Check the [API reference](api_reference/core.md) for detailed documentation
