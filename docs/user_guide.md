# Porcupy User Guide

This guide provides an introduction to the Porcupy library and demonstrates how to use it for optimization problems.

## Introduction

Porcupy is a Python library that implements the Crested Porcupine Optimizer (CPO) algorithm, a nature-inspired metaheuristic that mimics the defensive behaviors of crested porcupines. The algorithm uses four defensive mechanisms (sight, sound, odor, and physical attack) to balance exploration and exploitation, with cyclic population reduction for convergence.

## Installation

You can install Porcupy using pip:

```bash
pip install porcupy
```

Or you can install it from source:

```bash
git clone https://github.com/SammanSarkar/Porcupy.git
cd Porcupy
pip install -e .
```

## Basic Usage

Here's a simple example of how to use Porcupy to optimize a function:

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import sphere, get_function_bounds

# Define the problem
dimensions = 10
bounds = get_function_bounds('sphere', dimensions)

# Create the optimizer
optimizer = CPO(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=30,
    max_iter=100,
    cycles=2,
    alpha=0.2,
    tf=0.8
)

# Run the optimization
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=sphere,
    verbose=True
)

print(f"Best position: {best_pos}")
print(f"Best cost: {best_cost}")
```

## Using the Legacy API

If you prefer the procedural API, you can use the original `cpo` function:

```python
import numpy as np
from porcupy import cpo
from porcupy.functions import sphere

# Define the problem
dimensions = 10
lb = np.full(dimensions, -100)
ub = np.full(dimensions, 100)

# Run the optimization
best_pos, best_cost, cost_history = cpo(
    fobj=sphere,
    lb=lb,
    ub=ub,
    pop_size=30,
    max_iter=100,
    verbose=True
)

print(f"Best position: {best_pos}")
print(f"Best cost: {best_cost}")
```

## Advanced Usage

### Parallel Processing

You can use parallel processing to speed up the optimization process:

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import rosenbrock, get_function_bounds

# Define the problem
dimensions = 20
bounds = get_function_bounds('rosenbrock', dimensions)

# Create the optimizer
optimizer = CPO(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=50,
    max_iter=200
)

# Run the optimization with parallel processing
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=rosenbrock,
    n_processes=4,  # Use 4 processes
    verbose=True
)

print(f"Best position: {best_pos}")
print(f"Best cost: {best_cost}")
```

### Constraint Handling

You can add constraints to your optimization problem:

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import sphere

# Define the problem
dimensions = 2
lb = np.array([-100, -100])
ub = np.array([100, 100])

# Define a constraint function
# g(x) >= 0 for feasible solutions
def constraint(x):
    # Constraint: x[0]^2 + x[1]^2 <= 50^2
    # Rewritten as: 50^2 - x[0]^2 - x[1]^2 >= 0
    return [2500 - x[0]**2 - x[1]**2]

# Create the optimizer
optimizer = CPO(
    dimensions=dimensions,
    bounds=(lb, ub),
    pop_size=30,
    max_iter=100
)

# Run the optimization with constraints
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=sphere,
    f_ieqcons=constraint,
    verbose=True
)

print(f"Best position: {best_pos}")
print(f"Best cost: {best_cost}")
```

## Visualization

Porcupy provides several visualization tools to help you understand the optimization process:

### Convergence Plot

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import rastrigin, get_function_bounds
from porcupy.utils.visualization import plot_convergence

# Define the problem
dimensions = 10
bounds = get_function_bounds('rastrigin', dimensions)

# Create the optimizer
optimizer = CPO(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=30,
    max_iter=100
)

# Run the optimization
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=rastrigin,
    verbose=True
)

# Plot the convergence history
fig = plot_convergence(
    cost_history=optimizer.cost_history,
    title="Rastrigin Function Optimization",
    log_scale=True
)
fig.show()
```

### 2D Search Space Visualization

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import rastrigin, get_function_bounds
from porcupy.utils.visualization import plot_2d_search_space

# Define the problem
dimensions = 2
bounds = get_function_bounds('rastrigin', dimensions)

# Create the optimizer
optimizer = CPO(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=30,
    max_iter=100
)

# Run the optimization
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=rastrigin,
    verbose=True
)

# Plot the final positions in the search space
fig = plot_2d_search_space(
    func=rastrigin,
    bounds=bounds,
    positions=optimizer.positions,
    best_pos=best_pos,
    title="Rastrigin Function - Final Positions"
)
fig.show()
```

### Animation of the Optimization Process

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import rastrigin, get_function_bounds
from porcupy.utils.visualization import animate_optimization_2d

# Define the problem
dimensions = 2
bounds = get_function_bounds('rastrigin', dimensions)

# Create the optimizer
optimizer = CPO(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=30,
    max_iter=50
)

# Run the optimization
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=rastrigin,
    verbose=True
)

# Create an animation of the optimization process
anim = animate_optimization_2d(
    position_history=optimizer.position_history,
    func=rastrigin,
    bounds=bounds,
    save_path="rastrigin_optimization.gif"
)
```

## Comparing Multiple Runs

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import ackley, get_function_bounds
from porcupy.utils.visualization import plot_multiple_runs

# Define the problem
dimensions = 10
bounds = get_function_bounds('ackley', dimensions)

# Run the optimization multiple times with different parameters
cost_histories = []
labels = []

for pop_size in [20, 50, 100]:
    optimizer = CPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=pop_size,
        max_iter=100
    )
    
    best_pos, best_cost, cost_history = optimizer.optimize(
        objective_func=ackley,
        verbose=False
    )
    
    cost_histories.append(optimizer.cost_history)
    labels.append(f"Population Size = {pop_size}")

# Plot the comparison
fig = plot_multiple_runs(
    cost_histories=cost_histories,
    labels=labels,
    title="Effect of Population Size on Convergence",
    log_scale=True
)
fig.show()
```

## Parameter Sensitivity Analysis

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import griewank, get_function_bounds
from porcupy.utils.visualization import plot_parameter_sensitivity

# Define the problem
dimensions = 10
bounds = get_function_bounds('griewank', dimensions)

# Test different values of alpha
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
best_costs = []

for alpha in alpha_values:
    optimizer = CPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=30,
        max_iter=100,
        alpha=alpha
    )
    
    best_pos, best_cost, cost_history = optimizer.optimize(
        objective_func=griewank,
        verbose=False
    )
    
    best_costs.append(best_cost)

# Plot the sensitivity analysis
fig = plot_parameter_sensitivity(
    parameter_values=alpha_values,
    results=best_costs,
    parameter_name="Alpha",
    title="Sensitivity to Alpha Parameter"
)
fig.show()
```

## Custom Benchmark Functions

You can also define your own benchmark functions:

```python
import numpy as np
from porcupy import CPO

# Define a custom function
def custom_function(x):
    """A custom benchmark function."""
    return np.sum(x**2) + np.sum(np.sin(x))

# Define the problem
dimensions = 5
lb = np.full(dimensions, -10)
ub = np.full(dimensions, 10)

# Create the optimizer
optimizer = CPO(
    dimensions=dimensions,
    bounds=(lb, ub),
    pop_size=30,
    max_iter=100
)

# Run the optimization
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=custom_function,
    verbose=True
)

print(f"Best position: {best_pos}")
print(f"Best cost: {best_cost}")
```

## Advanced Population Management

You can customize the population management strategy:

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import schwefel, get_function_bounds
from porcupy.utils.population import PopulationCycle

# Define the problem
dimensions = 10
bounds = get_function_bounds('schwefel', dimensions)

# Create a custom population cycle manager
pop_cycle = PopulationCycle(
    initial_pop_size=50,
    min_pop_size=10,
    max_iter=100,
    cycles=3,
    reduction_strategy='cosine'  # Use cosine reduction strategy
)

# Create the optimizer
optimizer = CPO(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=50,
    min_pop_size=10,
    max_iter=100,
    cycles=3
)

# Run the optimization
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=schwefel,
    verbose=True
)

print(f"Best position: {best_pos}")
print(f"Best cost: {best_cost}")
```

## Conclusion

This guide covered the basic and advanced usage of the Porcupy library. For more details, refer to the [API Reference](api_reference.md) or check out the examples in the `examples` directory.
