"""
Parameter Sensitivity Analysis Example

This example demonstrates how to perform a parameter sensitivity analysis
for the Crested Porcupine Optimizer (CPO).
"""

import numpy as np
from porcupy import CrestPorcupineOptimizer
from porcupy.functions import rastrigin, get_function_bounds
from porcupy.utils.visualization import plot_parameter_sensitivity, plot_multiple_runs

# Define the problem
dimensions = 10
bounds = get_function_bounds('rastrigin', dimensions)

# Parameters to test
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
tf_values = [0.6, 0.7, 0.8, 0.9]
cycle_values = [1, 2, 3, 4]

# Function to run optimization with specific parameters
def run_optimization(alpha=0.2, tf=0.8, cycles=2, runs=3):
    """Run optimization with specific parameters multiple times"""
    best_costs = []
    cost_histories = []
    
    for _ in range(runs):
        optimizer = CrestPorcupineOptimizer(
            dimensions=dimensions,
            bounds=bounds,
            pop_size=30,
            max_iter=100,
            alpha=alpha,
            tf=tf,
            cycles=cycles
        )
        
        best_pos, best_cost, cost_history = optimizer.optimize(
            objective_func=rastrigin,
            verbose=False
        )
        
        best_costs.append(best_cost)
        cost_histories.append(optimizer.cost_history)
    
    return np.mean(best_costs), cost_histories

# Test alpha parameter
alpha_results = []
for alpha in alpha_values:
    mean_cost, _ = run_optimization(alpha=alpha)
    alpha_results.append(mean_cost)
    print(f"Alpha = {alpha}: Mean Best Cost = {mean_cost}")

# Test tf parameter
tf_results = []
for tf in tf_values:
    mean_cost, _ = run_optimization(tf=tf)
    tf_results.append(mean_cost)
    print(f"TF = {tf}: Mean Best Cost = {mean_cost}")

# Test cycles parameter
cycle_results = []
cycle_histories = []
for cycles in cycle_values:
    mean_cost, histories = run_optimization(cycles=cycles)
    cycle_results.append(mean_cost)
    cycle_histories.append(np.mean(histories, axis=0))  # Average over runs
    print(f"Cycles = {cycles}: Mean Best Cost = {mean_cost}")

# Visualization
try:
    import matplotlib.pyplot as plt
    
    # Plot alpha sensitivity
    plot_parameter_sensitivity(
        parameter_values=alpha_values,
        results=alpha_results,
        parameter_name="Alpha",
        title="Sensitivity to Alpha Parameter"
    )
    
    # Plot tf sensitivity
    plot_parameter_sensitivity(
        parameter_values=tf_values,
        results=tf_results,
        parameter_name="TF",
        title="Sensitivity to TF Parameter"
    )
    
    # Plot cycles sensitivity
    plot_parameter_sensitivity(
        parameter_values=cycle_values,
        results=cycle_results,
        parameter_name="Cycles",
        title="Sensitivity to Cycles Parameter"
    )
    
    # Plot convergence for different cycle values
    plot_multiple_runs(
        cost_histories=cycle_histories,
        labels=[f"Cycles = {c}" for c in cycle_values],
        title="Effect of Cycles on Convergence",
        log_scale=True
    )
    
    plt.show()
except ImportError:
    print("Matplotlib is not installed. Install it with 'pip install matplotlib' to see the plots.")
