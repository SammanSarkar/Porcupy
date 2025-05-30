"""
Population Dynamics Visualization Example for the Crested Porcupine Optimizer

This example demonstrates how to visualize the population dynamics of the
Crested Porcupine Optimizer algorithm, including cyclic population reduction,
diversity changes, and exploration-exploitation balance.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy.cpo_class import CrestPorcupineOptimizer
from porcupy.utils.visualization_manager import CPOVisualizer
from porcupy.utils.population_visualization import (
    plot_population_reduction_strategies,
    plot_population_diversity_map,
    animate_population_cycle,
    plot_exploration_exploitation_balance,
    plot_diversity_vs_convergence
)
from porcupy.functions import rastrigin, sphere, rosenbrock, ackley


def run_population_dynamics_example():
    """Run the population dynamics visualization example."""
    print("Population Dynamics Visualization Example")
    print("========================================")
    
    # Define the objective function
    def func(x):
        return rastrigin(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -5.12 * np.ones(dimensions)
    ub = 5.12 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Plot different population reduction strategies
    print("\nPlotting population reduction strategies...")
    fig1 = plot_population_reduction_strategies(
        max_iter=100,
        pop_size=50,
        cycles=5,
        strategies=['linear', 'cosine', 'exponential']
    )
    
    # Initialize the CPO optimizer
    print("\nRunning optimization with default strategy...")
    optimizer = CrestPorcupineOptimizer(
        dimensions=dimensions,
        bounds=(lb, ub),
        pop_size=50,
        max_iter=100,
        cycles=5
    )
    
    # Initialize the visualizer
    visualizer = CPOVisualizer(objective_func=func, bounds=bounds)
    
    # Run the optimization
    best_pos, best_cost, cost_history = optimizer.optimize(func, verbose=True)
    print(f"Optimization completed. Best cost: {best_cost:.6f}")
    
    # For demonstration purposes, we'll generate some random data for visualization
    # In a real implementation, this data would come from the optimizer during the optimization process
    iterations = 100
    positions_history = []
    pop_size_history = []
    diversity_history = []
    
    # Generate random positions and population sizes for each iteration
    for i in range(iterations):
        # Calculate population size for this iteration
        pop_size = optimizer._calculate_current_pop_size(i)
        pop_size_history.append(pop_size)
        
        # Generate random positions within bounds
        positions = np.random.uniform(lb, ub, (pop_size, dimensions))
        positions_history.append(positions)
        
        # Calculate diversity (average distance between positions)
        if pop_size > 1:
            distances = []
            for j in range(pop_size):
                for k in range(j+1, pop_size):
                    distances.append(np.linalg.norm(positions[j] - positions[k]))
            diversity = np.mean(distances) if distances else 0
        else:
            diversity = 0
        diversity_history.append(diversity)
        
        # Record the iteration data
        visualizer.record_iteration(
            iteration=i,
            positions=positions,
            best_position=best_pos,
            best_fitness=cost_history[i] if i < len(cost_history) else best_cost,
            pop_size=pop_size,
            defense_types=None  # Not needed for population dynamics
        )
    
    # Create population cycles visualization
    print("\nCreating population cycles visualization...")
    fig2 = visualizer.visualize_population_cycles(
        cycles=5,
        max_iter=100,
        title="Population Size Cycles"
    )
    
    # Create diversity history visualization
    print("Creating diversity history visualization...")
    fig3 = visualizer.visualize_diversity_history(
        title="Population Diversity History"
    )
    
    # Create diversity vs convergence visualization
    print("Creating diversity vs convergence visualization...")
    fig4 = visualizer.visualize_diversity_vs_convergence(
        cycles=5,
        max_iter=100,
        title="Diversity vs Convergence"
    )
    
    # Create exploration vs exploitation visualization
    print("Creating exploration vs exploitation visualization...")
    fig5 = plot_exploration_exploitation_balance(
        positions_history=positions_history,
        best_positions_history=[best_pos] * len(positions_history),
        bounds=bounds,
        sample_iterations=[0, 20, 40, 60, 80, 99],  # Sample iterations
        save_path=None
    )
    
    # Create population diversity map
    print("Creating population diversity map...")
    fig6 = plot_population_diversity_map(
        positions_history=positions_history,
        bounds=bounds,
        sample_iterations=[0, 20, 40, 60, 80, 99],  # Sample iterations
        save_path=None
    )
    
    # Create population cycle animation
    print("Creating population cycle animation...")
    # Use a subset of iterations for the animation to keep it manageable
    indices = np.arange(0, len(positions_history), 2)
    position_subset = [positions_history[i] for i in indices]
    pop_size_subset = [pop_size_history[i] for i in indices]
    
    anim = animate_population_cycle(
        positions_history=position_subset,
        pop_size_history=pop_size_subset,
        bounds=bounds,
        max_iter=len(indices),
        cycles=5,
        interval=200,
        save_path=None
    )
    
    # Show all figures
    plt.show()
    
    print("\nPopulation dynamics visualizations completed.")


if __name__ == "__main__":
    run_population_dynamics_example()
