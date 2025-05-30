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
    print("=======================================")
    
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
        dimensions=dimensions,
        bounds=(lb, ub),
        pop_size=30,
        max_iter=50,
        cycles=5
    )
    
    # Initialize the visualizer
    visualizer = CPOVisualizer(objective_func=func, bounds=bounds)
    
    # Run the optimization with defense mechanism tracking
    def callback(iteration, positions, best_pos, best_cost, pop_size):
        # In a real implementation, these would come from the optimizer
        # Here we simulate defense mechanism activation based on iteration progress
        exploration_ratio = 1 - (iteration / optimizer.max_iter)
        
        defenses = []
        for i in range(len(positions)):
            # Early iterations favor exploration (sight, sound)
            # Later iterations favor exploitation (odor, physical)
            if np.random.random() < exploration_ratio:
                # Exploration phase
                defense = np.random.choice(['sight', 'sound'], p=[0.6, 0.4])
            else:
                # Exploitation phase
                defense = np.random.choice(['odor', 'physical'], p=[0.7, 0.3])
            
            defenses.append(defense)
        
        # Record the iteration data
        visualizer.record_iteration(
            iteration=iteration,
            positions=positions,
            best_position=best_pos,
            best_fitness=best_cost,
            pop_size=pop_size,
            defense_types=defenses
        )
        
        return False  # Continue optimization
    
    # Run the optimization
    print("Running optimization with defense mechanism tracking...")
    result = optimizer.optimize(func, callback=callback)
    print(f"Optimization completed. Best cost: {result['best_cost']:.6f}")
    
    # Create defense mechanism activation visualization
    print("\nCreating defense mechanism activation visualization...")
    fig1 = visualizer.visualize_defense_mechanisms(
        title="Defense Mechanism Activation During Optimization",
        save_path="defense_mechanisms_activation.png"
    )
    
    # Create defense territories visualization
    print("Creating defense territories visualization...")
    fig2 = visualizer.visualize_defense_territories(
        iteration=-1,  # Last iteration
        title="Porcupine Defense Territories",
        save_path="defense_territories.png"
    )
    
    # Create defense effectiveness visualization
    print("Creating defense effectiveness visualization...")
    fig3 = visualizer.visualize_defense_effectiveness(
        title="Defense Mechanism Effectiveness",
        save_path="defense_effectiveness.png"
    )
    
    # Create animation of defense mechanisms
    print("Creating animation of defense mechanisms...")
    # For this example, we'll use a smaller subset of iterations for the animation
    from porcupy.utils.defense_visualization import animate_defense_mechanisms
    
    # Use every 5th iteration for the animation to keep it manageable
    indices = np.arange(0, len(visualizer.position_history), 5)
    position_subset = [visualizer.position_history[i] for i in indices]
    
    # Create defense types history in the required format
    defense_history = []
    for i in indices:
        defenses = []
        for defense in ['sight', 'sound', 'odor', 'physical']:
            count = visualizer.defense_history[defense][i]
            defenses.extend([defense] * count)
        defense_history.append(defenses)
    
    best_pos_subset = [visualizer.best_position_history[i] for i in indices]
    
    # Create the animation
    anim = animate_defense_mechanisms(
        position_history=position_subset,
        defense_history=defense_history,
        bounds=bounds,
        best_position_history=best_pos_subset,
        interval=200,
        save_path="defense_mechanisms_animation.gif"
    )
    
    # Show all figures
    plt.show()
    
    print("\nAll visualizations completed and saved to current directory.")
    print("- defense_mechanisms_activation.png")
    print("- defense_territories.png")
    print("- defense_effectiveness.png")
    print("- defense_mechanisms_animation.gif")


if __name__ == "__main__":
    run_defense_mechanism_example()
