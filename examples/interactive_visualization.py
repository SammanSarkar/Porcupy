"""
Interactive Visualization Example for the Crested Porcupine Optimizer

This example demonstrates the interactive visualization capabilities of the
Porcupy library, including real-time dashboards and parameter tuning visualizations.
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
from porcupy.functions import rastrigin, sphere, rosenbrock, ackley


def run_dashboard_example():
    """Run the interactive dashboard example."""
    print("Interactive Dashboard Example")
    print("============================")
    
    # Define a 2D test function
    def func(x):
        return rastrigin(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -5.12 * np.ones(dimensions)
    ub = 5.12 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer
    optimizer = CrestPorcupineOptimizer(
        pop_size=30,
        dimensions=dimensions,
        max_iter=50,
        lb=lb,
        ub=ub,
        cycles=5
    )
    
    # Initialize the visualizer
    visualizer = CPOVisualizer(objective_func=func, bounds=bounds)
    
    # Create the dashboard
    print("Creating interactive dashboard...")
    dashboard = visualizer.create_dashboard(update_interval=0.2)
    
    # Start the dashboard monitoring
    dashboard.start_monitoring()
    
    print("Running optimization with real-time dashboard updates...")
    print("(Note: This may take a moment to initialize the dashboard)")
    
    # Run the optimization with real-time dashboard updates
    def callback(iteration, positions, best_pos, best_cost, pop_size):
        # Generate random defense types for demonstration
        defenses = []
        for _ in range(len(positions)):
            defense = np.random.choice(['sight', 'sound', 'odor', 'physical'])
            defenses.append(defense)
        
        # Update the dashboard
        dashboard.update(
            iteration=iteration,
            best_cost=best_cost,
            pop_size=pop_size,
            positions=positions,
            best_position=best_pos,
            defense_types=defenses
        )
        
        # Add a small delay to see the dashboard updates
        time.sleep(0.1)
        
        return False  # Continue optimization
    
    # Run the optimization
    result = optimizer.optimize(func, callback=callback)
    
    print(f"Optimization completed. Best cost: {result['best_cost']:.6f}")
    
    # Stop the dashboard monitoring
    dashboard.stop_monitoring()
    
    # Save the final dashboard state
    dashboard.save_dashboard("final_dashboard.png")
    
    print("Dashboard monitoring stopped. Final state saved as 'final_dashboard.png'")
    print("Close the dashboard window to continue...")
    
    # Keep the dashboard open for a while to allow user interaction
    plt.show()
    
    # Close the dashboard
    dashboard.close()


def run_parameter_tuning_example():
    """Run the parameter tuning visualization example."""
    print("\nParameter Tuning Visualization Example")
    print("====================================")
    
    # Define a test function
    def func(x):
        return sphere(x)
    
    # Define bounds for the search space
    dimensions = 10
    lb = -100 * np.ones(dimensions)
    ub = 100 * np.ones(dimensions)
    
    # Initialize the visualizer
    visualizer = CPOVisualizer()
    
    # Create a parameter tuning dashboard
    print("Creating parameter tuning dashboard...")
    dashboard = visualizer.create_parameter_tuning_dashboard(
        parameter_name="Population Size",
        parameter_range=[10, 20, 30, 40, 50, 60],
        result_metric="Best Cost"
    )
    
    # Test different population sizes
    print("Testing different population sizes...")
    for pop_size in [10, 20, 30, 40, 50, 60]:
        print(f"  Testing population size: {pop_size}")
        
        # Initialize the CPO optimizer with the current population size
        optimizer = CrestPorcupineOptimizer(
            pop_size=pop_size,
            dimensions=dimensions,
            max_iter=100,
            lb=lb,
            ub=ub
        )
        
        # Run the optimization
        result = optimizer.optimize(func)
        
        # Get the convergence history
        convergence_history = optimizer.cost_history
        
        # Update the dashboard
        dashboard.update(
            parameter_value=pop_size,
            result=result['best_cost'],
            convergence_history=convergence_history
        )
    
    # Save the dashboard
    dashboard.save_dashboard("parameter_tuning.png")
    
    print("Parameter tuning completed. Dashboard saved as 'parameter_tuning.png'")
    print("Close the dashboard window to continue...")
    
    # Keep the dashboard open for a while to allow user interaction
    plt.show()
    
    # Close the dashboard
    dashboard.close()


def run_interactive_exploration_example():
    """Run the interactive exploration example."""
    print("\nInteractive Exploration Example")
    print("==============================")
    
    # Define a 2D test function
    def func(x):
        return ackley(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -32.768 * np.ones(dimensions)
    ub = 32.768 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer
    optimizer = CrestPorcupineOptimizer(
        pop_size=30,
        dimensions=dimensions,
        max_iter=1,  # Just need initial positions
        lb=lb,
        ub=ub
    )
    
    # Generate initial positions
    initial_positions = optimizer._init_population()
    
    # Create an interactive plot
    print("Creating interactive exploration plot...")
    print("(Click on the plot to evaluate the function at specific points)")
    
    from porcupy.utils.interactive_visualization import create_interactive_optimization_plot
    
    fig, ax, scatter = create_interactive_optimization_plot(
        objective_func=func,
        bounds=bounds,
        initial_positions=initial_positions,
        figsize=(10, 8)
    )
    
    # Save the initial plot
    plt.savefig("interactive_exploration.png")
    
    print("Interactive plot created. Click on the plot to explore the function.")
    print("Close the plot window to continue...")
    
    # Keep the plot open for user interaction
    plt.show()


if __name__ == "__main__":
    run_dashboard_example()
    run_parameter_tuning_example()
    run_interactive_exploration_example()
    
    print("\nAll interactive visualization examples completed.")
