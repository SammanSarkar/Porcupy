"""
Visualization Demo for the Crested Porcupine Optimizer

This example demonstrates the enhanced visualization capabilities of the Porcupy library,
showcasing various ways to visualize the CPO algorithm's behavior and performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
import os

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy.cpo_class import CrestPorcupineOptimizer
from porcupy.utils.visualization_manager import CPOVisualizer
from porcupy.functions import rastrigin, sphere, rosenbrock, ackley


def demo_basic_visualizations():
    """Demonstrate basic visualization capabilities."""
    print("Demonstrating basic visualizations...")
    
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
    
    # Track defense mechanisms during optimization
    defense_types = []
    
    # Run the optimization
    result = optimizer.optimize(func, verbose=True)
    best_pos, best_cost, cost_history = result
    
    # For demonstration purposes, we'll generate some random data for visualization
    # In a real implementation, this data would come from the optimizer during the optimization process
    iterations = 50
    positions_history = []
    defense_types_history = []
    pop_size_history = []
    
    # Generate random positions and defense types for each iteration
    for i in range(iterations):
        # Calculate population size for this iteration
        pop_size = optimizer._calculate_current_pop_size(i)
        pop_size_history.append(pop_size)
        
        # Generate random positions within bounds
        positions = np.random.uniform(lb, ub, (pop_size, dimensions))
        positions_history.append(positions)
        
        # Generate random defense types
        defenses = []
        for _ in range(pop_size):
            defense = np.random.choice(['sight', 'sound', 'odor', 'physical'])
            defenses.append(defense)
        defense_types_history.append(defenses)
        
        # Record the iteration data
        visualizer.record_iteration(
            iteration=i,
            positions=positions,
            best_position=best_pos,
            best_fitness=cost_history[i] if i < len(cost_history) else best_cost,
            pop_size=pop_size,
            defense_types=defenses
        )
    
    # Create and display visualizations
    
    # 1. Defense mechanism activation
    fig1 = visualizer.visualize_defense_mechanisms(
        title="Defense Mechanism Activation During Optimization",
        save_path="defense_mechanisms.png"
    )
    
    # 2. Population size cycles
    fig2 = visualizer.visualize_population_cycles(
        cycles=5,
        max_iter=50,
        title="Population Size Reduction Cycles",
        save_path="population_cycles.png"
    )
    
    # 3. Diversity history
    fig3 = visualizer.visualize_diversity_history(
        title="Population Diversity Throughout Optimization",
        save_path="diversity_history.png"
    )
    
    # 4. 2D porcupine visualization
    fig4 = visualizer.visualize_porcupines_2d(
        iteration=-1,  # Last iteration
        title="Final Porcupine Positions",
        save_path="porcupines_2d.png"
    )
    
    # 5. Defense territories
    fig5 = visualizer.visualize_defense_territories(
        iteration=-1,  # Last iteration
        title="Porcupine Defense Territories",
        save_path="defense_territories.png"
    )
    
    # 6. Exploration vs exploitation
    fig6 = visualizer.visualize_exploration_exploitation(
        sample_iterations=[0, 10, 20, 30, 40, 49],  # Specific iterations to visualize
        save_path="exploration_exploitation.png"
    )
    
    # 7. Diversity vs convergence
    fig7 = visualizer.visualize_diversity_vs_convergence(
        cycles=5,
        max_iter=50,
        title="Diversity vs Convergence Relationship",
        save_path="diversity_vs_convergence.png"
    )
    
    # 8. Defense effectiveness
    fig8 = visualizer.visualize_defense_effectiveness(
        title="Defense Mechanism Effectiveness",
        save_path="defense_effectiveness.png"
    )
    
    # 9. Compare reduction strategies
    fig9 = visualizer.compare_reduction_strategies(
        max_iter=50,
        pop_size=30,
        cycles=5,
        strategies=['linear', 'cosine', 'exponential'],
        save_path="reduction_strategies.png"
    )
    
    # Show all figures
    plt.show()
    
    print("Basic visualizations completed. Images saved to current directory.")
    
    return visualizer


def demo_animation():
    """Demonstrate animation capabilities."""
    print("Demonstrating animation capabilities...")
    
    # Load the visualizer from the previous demo
    visualizer = demo_basic_visualizations()
    
    # Create an animation of the optimization process
    anim = visualizer.animate_optimization(
        interval=200,  # 200ms between frames
        save_path="optimization_animation.gif"
    )
    
    print("Animation created and saved as 'optimization_animation.gif'")


def demo_interactive_dashboard():
    """Demonstrate the interactive dashboard."""
    print("Demonstrating interactive dashboard...")
    
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
        max_iter=100,
        lb=lb,
        ub=ub,
        cycles=5
    )
    
    # Initialize the visualizer
    visualizer = CPOVisualizer(objective_func=func, bounds=bounds)
    
    # Create the dashboard
    dashboard = visualizer.create_dashboard(update_interval=0.2)
    
    # Start the dashboard monitoring
    dashboard.start_monitoring()
    
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
    
    # Stop the dashboard monitoring
    dashboard.stop_monitoring()
    
    # Save the final dashboard state
    dashboard.save_dashboard("final_dashboard.png")
    
    print("Interactive dashboard demonstration completed.")
    print("Final dashboard state saved as 'final_dashboard.png'")
    
    # Close the dashboard
    dashboard.close()


def demo_parameter_tuning():
    """Demonstrate parameter tuning visualization."""
    print("Demonstrating parameter tuning visualization...")
    
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
    dashboard = visualizer.create_parameter_tuning_dashboard(
        parameter_name="Population Size",
        parameter_range=[10, 20, 30, 40, 50, 60],
        result_metric="Best Cost"
    )
    
    # Test different population sizes
    for pop_size in [10, 20, 30, 40, 50, 60]:
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
    
    print("Parameter tuning visualization completed.")
    print("Parameter tuning dashboard saved as 'parameter_tuning.png'")
    
    # Close the dashboard
    dashboard.close()


if __name__ == "__main__":
    print("CPO Visualization Demo")
    print("======================")
    
    # Run the demos
    demo_basic_visualizations()
    demo_animation()
    demo_interactive_dashboard()
    demo_parameter_tuning()
    
    print("All demonstrations completed successfully!")
