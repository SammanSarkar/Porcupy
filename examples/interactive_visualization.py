"""
Interactive Visualization Example for the Crested Porcupine Optimizer

This example demonstrates the interactive visualization capabilities of the Porcupy library,
including real-time dashboards for monitoring optimization progress and parameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy.cpo_class import CrestPorcupineOptimizer
from porcupy.utils.visualization_manager import CPOVisualizer
from porcupy.utils.interactive_visualization import (
    OptimizationDashboard,
    ParameterTuningDashboard,
    create_interactive_optimization_plot
)
from porcupy.functions import rastrigin, sphere, rosenbrock, ackley


def run_interactive_dashboard_example():
    """Run the interactive dashboard example."""
    print("Interactive Dashboard Example")
    print("============================")
    
    # Define the objective function
    def func(x):
        return rastrigin(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -5.12 * np.ones(dimensions)
    ub = 5.12 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer
    optimizer = CrestPorcupineOptimizer(
        dimensions=dimensions,
        bounds=(lb, ub),
        pop_size=30,
        max_iter=50,
        cycles=5
    )
    
    # Initialize the dashboard
    dashboard = OptimizationDashboard(
        objective_func=func,
        bounds=bounds,
        dimensions=dimensions,
        update_interval=0.5,
        figsize=(15, 10)
    )
    
    # Run the optimization
    print("Running optimization with dashboard visualization...")
    best_pos, best_cost, cost_history = optimizer.optimize(func, verbose=True)
    print(f"Optimization completed. Best cost: {best_cost:.6f}")
    
    # For demonstration purposes, we'll simulate the dashboard updates
    # In a real implementation, these updates would happen during optimization
    print("\nSimulating dashboard updates...")
    
    # Generate random positions and defense types for each iteration
    for i in range(50):
        # Calculate population size for this iteration
        pop_size = optimizer._calculate_current_pop_size(i)
        
        # Generate random positions within bounds
        positions = np.random.uniform(lb, ub, (pop_size, dimensions))
        
        # Generate random defense types
        defenses = []
        for _ in range(pop_size):
            defense = np.random.choice(['sight', 'sound', 'odor', 'physical'])
            defenses.append(defense)
        
        # Update the dashboard
        dashboard.update(
            iteration=i,
            best_cost=cost_history[i] if i < len(cost_history) else best_cost,
            pop_size=pop_size,
            positions=positions,
            best_position=best_pos,
            defense_types=defenses
        )
        
        # Simulate some processing time
        time.sleep(0.1)
    
    # Show the final dashboard (using plt.show() since dashboard doesn't have a show method)
    plt.figure(dashboard.fig.number)
    plt.show()
    
    print("Interactive dashboard example completed.")


def run_parameter_tuning_example():
    """Run the parameter tuning dashboard example."""
    print("\nParameter Tuning Dashboard Example")
    print("=================================")
    
    # Define the objective function
    def func(x):
        return rastrigin(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -5.12 * np.ones(dimensions)
    ub = 5.12 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Define parameter ranges for tuning
    param_ranges = {
        'pop_size': [10, 20, 30, 40, 50],
        'cycles': [1, 2, 3, 4, 5],
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    # Initialize the parameter tuning dashboard
    # Choose one parameter to tune for this example
    tuning_dashboard = ParameterTuningDashboard(
        parameter_name="pop_size",
        parameter_range=param_ranges['pop_size'],
        result_metric="Best Cost",
        figsize=(12, 8)
    )
    
    # For demonstration purposes, we'll simulate the parameter tuning process
    # In a real implementation, this would involve running multiple optimizations
    print("Simulating parameter tuning process...")
    
    # Generate results for each parameter combination
    results = {}
    
    for pop_size in param_ranges['pop_size']:
        for cycles in param_ranges['cycles']:
            for alpha in param_ranges['alpha']:
                # Create a parameter key
                param_key = f"pop_size={pop_size},cycles={cycles},alpha={alpha}"
                
                # Simulate running the optimization
                print(f"Testing parameters: {param_key}")
                
                # Initialize the optimizer with these parameters
                optimizer = CrestPorcupineOptimizer(
                    dimensions=dimensions,
                    bounds=bounds,
                    pop_size=pop_size,
                    cycles=cycles,
                    alpha=alpha,
                    max_iter=50
                )
                
                # Run a quick optimization
                best_pos, best_cost, cost_history = optimizer.optimize(func, verbose=False)
                
                # Store the results
                results[param_key] = {
                    'best_cost': best_cost,
                    'convergence_rate': np.mean(np.diff(cost_history)),
                    'cost_history': cost_history
                }
                
                # Update the dashboard - only use the parameter we're tuning
                tuning_dashboard.update(
                    parameter_value=pop_size,  # Using pop_size as our parameter
                    result=best_cost,
                    convergence_history=cost_history
                )
    
    # Show the final tuning dashboard (using plt.show() since dashboard doesn't have a show method)
    plt.figure(tuning_dashboard.fig.number)
    plt.show()
    
    # Find the best parameter combination
    best_param_key = min(results, key=lambda k: results[k]['best_cost'])
    best_params = {
        param: float(value) for param, value in 
        [pair.split('=') for pair in best_param_key.split(',')]
    }
    
    print(f"\nBest parameter combination: {best_param_key}")
    print(f"Best cost achieved: {results[best_param_key]['best_cost']:.6f}")
    
    print("Parameter tuning example completed.")


def run_interactive_plot_example():
    """Run the interactive optimization plot example."""
    print("\nInteractive Optimization Plot Example")
    print("===================================")
    
    # Define the objective function
    def func(x):
        return rastrigin(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -5.12 * np.ones(dimensions)
    ub = 5.12 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer
    optimizer = CrestPorcupineOptimizer(
        dimensions=dimensions,
        bounds=(lb, ub),
        pop_size=30,
        max_iter=50,
        cycles=5
    )
    
    # Run the optimization
    print("Running optimization...")
    best_pos, best_cost, cost_history = optimizer.optimize(func, verbose=True)
    print(f"Optimization completed. Best cost: {best_cost:.6f}")
    
    # For demonstration purposes, we'll generate some random data for visualization
    # In a real implementation, this data would come from the optimizer during the optimization process
    positions_history = []
    
    # Generate random positions for each iteration
    for i in range(50):
        # Calculate population size for this iteration
        pop_size = optimizer._calculate_current_pop_size(i)
        
        # Generate random positions within bounds
        positions = np.random.uniform(lb, ub, (pop_size, dimensions))
        positions_history.append(positions)
    
    # Create the interactive optimization plot
    print("Creating interactive optimization plot...")
    # For interactive plot, we need initial positions
    initial_positions = positions_history[0] if positions_history else np.random.uniform(lb, ub, (30, dimensions))
    
    interactive_plot = create_interactive_optimization_plot(
        objective_func=func,
        bounds=bounds,
        initial_positions=initial_positions
    )
    
    # Show the interactive plot (using plt.show() since it returns a tuple, not an object with show method)
    fig, ax, scatter = interactive_plot
    plt.show()
    
    print("Interactive optimization plot example completed.")


if __name__ == "__main__":
    run_interactive_dashboard_example()
    run_parameter_tuning_example()
    run_interactive_plot_example()
