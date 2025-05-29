"""
Constrained Optimization Example

This example demonstrates how to use the Crested Porcupine Optimizer (CPO)
for a constrained optimization problem.
"""

import numpy as np
from porcupy import CrestPorcupineOptimizer
from porcupy.functions import sphere
from porcupy.utils.visualization import plot_2d_search_space, plot_convergence

# Define the problem
dimensions = 2
lb = np.array([-100, -100])
ub = np.array([100, 100])

# Define a constraint function
# g(x) >= 0 for feasible solutions
def constraint(x):
    """
    Constraint: x[0]^2 + x[1]^2 <= 50^2
    Rewritten as: 50^2 - x[0]^2 - x[1]^2 >= 0
    """
    return [2500 - x[0]**2 - x[1]**2]

# Create the optimizer
optimizer = CrestPorcupineOptimizer(
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
print(f"Constraint value: {constraint(best_pos)[0]}")
print(f"Distance from origin: {np.linalg.norm(best_pos)}")

# Visualization
try:
    import matplotlib.pyplot as plt
    
    # Plot convergence history
    plot_convergence(
        cost_history=optimizer.cost_history,
        title="Constrained Sphere Function Optimization",
        log_scale=True
    )
    
    # Plot the search space with constraint
    def constrained_sphere(x):
        """Sphere function with penalty for constraint violation"""
        c = constraint(x)[0]
        if c >= 0:
            return sphere(x)
        else:
            return sphere(x) - 1000 * c  # Large penalty for constraint violation
    
    # Plot the final positions in the search space
    fig = plot_2d_search_space(
        func=constrained_sphere,
        bounds=(lb, ub),
        positions=optimizer.positions,
        best_pos=best_pos,
        title="Constrained Sphere Function - Final Positions"
    )
    
    # Add constraint boundary (circle with radius 50)
    ax = plt.gca()
    circle = plt.Circle((0, 0), 50, fill=False, color='r', linestyle='--', linewidth=2, label='Constraint Boundary')
    ax.add_patch(circle)
    plt.legend()
    
    plt.show()
except ImportError:
    print("Matplotlib is not installed. Install it with 'pip install matplotlib' to see the plots.")
