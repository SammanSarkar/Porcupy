# Configuration Guide

This guide explains how to configure Porcupy for optimal performance and specific use cases.

## Basic Configuration

### Algorithm Parameters

Porcupy's `CPO` class accepts several parameters that control the optimization process:

```python
optimizer = CPO(
    dimensions=10,           # Number of dimensions in the problem
    bounds=([-5]*10, [5]*10), # Search space bounds (min, max) for each dimension
    pop_size=50,             # Population size (number of candidate solutions)
    max_iter=100,            # Maximum number of iterations
    ftol=1e-6,               # Function tolerance for early stopping
    ftol_iter=10,            # Number of iterations to check for convergence
    seed=None,               # Random seed for reproducibility
    verbose=True,            # Whether to print progress
    track_history=True       # Whether to track optimization history
)
```

### Defense Mechanism Parameters

You can customize the defense mechanisms used by the porcupines:

```python
optimizer = CPO(
    # ... other parameters ...
    defense_params={
        'sight_weight': 1.0,    # Weight for sight-based defense
        'sound_weight': 1.0,    # Weight for sound-based defense
        'odor_weight': 1.0,     # Weight for odor-based defense
        'physical_weight': 1.0, # Weight for physical defense
        'adaptation_rate': 0.1  # Rate at which weights adapt
    }
)
```

## Advanced Configuration

### Parallel Processing

Porcupy supports parallel evaluation of the population using Python's multiprocessing:

```python
from multiprocessing import Pool

# Create a pool of worker processes
with Pool(processes=4) as pool:
    best_solution, best_fitness, _ = optimizer.optimize(
        objective_func=my_function,
        pool=pool  # Pass the pool to the optimizer
    )
```

### Custom Stopping Criteria

You can implement custom stopping criteria by subclassing the `CPO` class:

```python
class MyCPO(CPO):
    def _check_stopping_criteria(self, iteration, fitness_history):
        # Stop if the best fitness is below a threshold
        if fitness_history and fitness_history[-1] < 0.01:
            return True
        return super()._check_stopping_criteria(iteration, fitness_history)
```

## GPU Configuration

When using `GPUCPO`, you can configure GPU-specific parameters:

```python
from porcupy import GPUCPO

optimizer = GPUCPO(
    # ... standard CPO parameters ...
    gpu_params={
        'dtype': 'float32',  # Data type for GPU computations
        'device_id': 0,      # GPU device ID (for multi-GPU systems)
        'memory_fraction': 0.8  # Maximum fraction of GPU memory to use
    }
)
```

## Environment Variables

Porcupy respects the following environment variables:

- `PORCUPY_DEBUG`: Set to 1 to enable debug logging
- `CUDA_VISIBLE_DEVICES`: For GPU selection (e.g., "0,1" to use first two GPUs)
- `OMP_NUM_THREADS`: Controls the number of threads for CPU parallelization

## Performance Tuning

### For CPU Optimization
- Set `OMP_NUM_THREADS` to the number of physical cores
- Use smaller population sizes for faster convergence
- Consider using `numpy` with MKL for better performance

### For GPU Optimization
- Use powers of 2 for population size (e.g., 64, 128, 256)
- Batch objective function evaluations when possible
- Monitor GPU memory usage to avoid out-of-memory errors

## Configuration Examples

### Constrained Optimization

```python
def constraint(x):
    return x[0] + x[1] - 1.0  # x + y >= 1

def objective(x):
    return x[0]**2 + x[1]**2

optimizer = CPO(dimensions=2, bounds=([0, 0], [1, 1]))
best_solution, best_fitness, _ = optimizer.optimize(
    objective_func=objective,
    f_ieqcons=lambda x: constraint(x)  # Inequality constraint
)
```

### Custom Initial Population

```python
# Generate custom initial population
custom_population = np.random.uniform(low=-5, high=5, size=(50, 10))

optimizer = CPO(dimensions=10, bounds=([-5]*10, [5]*10))
optimizer.initialize_population(custom_population)
best_solution, best_fitness, _ = optimizer.optimize(rastrigin)
```
