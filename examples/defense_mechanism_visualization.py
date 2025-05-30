"""
Defense Mechanism Visualization Example for the Crested Porcupine Optimizer

This example demonstrates how to visualize the unique defense mechanisms of the
Crested Porcupine Optimizer algorithm, including sight, sound, odor, and physical defenses.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy.cpo_class import CrestPorcupineOptimizer
from porcupy.utils.visualization_manager import CPOVisualizer
from porcupy.functions import rastrigin, sphere, rosenbrock, ackley


def run_defense_mechanism_example():
    """Run the defense mechanism visualization example."""
    print("Defense Mechanism Visualization Example")
    print("======================================")
    
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
    
    # Initialize the visualizer
    visualizer = CPOVisualizer(objective_func=func, bounds=bounds)
    
    # Run the optimization
    print("Running optimization...")
    best_pos, best_cost, cost_history = optimizer.optimize(func, verbose=True)
    print(f"Optimization completed. Best cost: {best_cost:.6f}")
    
    # For demonstration purposes, we'll generate some random data for visualization
    # In a real implementation, this data would come from the optimizer during the optimization process
    iterations = 50
    
    # Generate random positions and defense types for each iteration
    for i in range(iterations):
        # Calculate population size for this iteration
        pop_size = optimizer._calculate_current_pop_size(i)
        
        # Generate random positions within bounds
        positions = np.random.uniform(lb, ub, (pop_size, dimensions))
        
        # Generate random defense types based on iteration progress
        # Early iterations favor exploration (sight, sound)
        # Later iterations favor exploitation (odor, physical)
        exploration_ratio = 1 - (i / iterations)
        
        defenses = []
        for j in range(pop_size):
            if np.random.random() < exploration_ratio:
                # Exploration phase
                defense = np.random.choice(['sight', 'sound'], p=[0.6, 0.4])
            else:
                # Exploitation phase
                defense = np.random.choice(['odor', 'physical'], p=[0.7, 0.3])
            defenses.append(defense)
        
        # Record the iteration data
        visualizer.record_iteration(
            iteration=i,
            positions=positions,
            best_position=best_pos,
            best_fitness=cost_history[i] if i < len(cost_history) else best_cost,
            pop_size=pop_size,
            defense_types=defenses
        )
    
    # Create defense mechanism activation visualization
    print("\nCreating defense mechanism activation visualization...")
    fig1 = visualizer.visualize_defense_mechanisms(
        title="Defense Mechanism Activation During Optimization"
    )
    
    # Create defense territories visualization
    print("Creating defense territories visualization...")
    fig2 = visualizer.visualize_defense_territories(
        title="Defense Territories in Search Space"
    )
    
    # Create defense effectiveness visualization
    print("Creating defense effectiveness visualization...")
    fig3 = visualizer.plot_defense_effectiveness(
        title="Effectiveness of Different Defense Mechanisms"
    )
    
    # Create animated visualization of defense mechanisms
    print("Creating animated visualization of defense mechanisms...")
    anim = visualizer.animate_defense_mechanisms(
        title="Defense Mechanism Animation"
    )
    
    # Show all figures
    plt.show()
    
    print("\nDefense mechanism visualizations completed.")


if __name__ == "__main__":
    run_defense_mechanism_example()