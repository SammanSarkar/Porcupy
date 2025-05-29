"""
Basic Optimization Example

This example demonstrates how to use the Crested Porcupine Optimizer (CPO)
for a simple optimization problem.
"""

import numpy as np
from porcupy import CrestPorcupineOptimizer
from porcupy.functions import sphere, get_function_bounds
from porcupy.utils.visualization import plot_convergence

# Define the problem
dimensions = 10
bounds = get_function_bounds('sphere', dimensions)

# Create the optimizer
optimizer = CrestPorcupineOptimizer(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=30,
    max_iter=100
)

# Run the optimization
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=sphere,
    verbose=True
)

print(f"Best position: {best_pos}")
print(f"Best cost: {best_cost}")

# Plot the convergence history
try:
    import matplotlib.pyplot as plt
    
    fig = plot_convergence(
        cost_history=optimizer.cost_history,
        title="Sphere Function Optimization",
        log_scale=True
    )
    
    # Plot the population size history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(optimizer.pop_size_history) + 1), optimizer.pop_size_history, 'r-')
    plt.title("Population Size History")
    plt.xlabel("Iterations")
    plt.ylabel("Population Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()
except ImportError:
    print("Matplotlib is not installed. Install it with 'pip install matplotlib' to see the plots.")
