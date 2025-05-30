"""
Population Dynamics Visualization Example for the Crested Porcupine Optimizer

This example demonstrates how to visualize the population dynamics of the
Crested Porcupine Optimizer algorithm, including cyclic population reduction,
diversity changes, and exploration-exploitation balance.
"""

import numpy as np
import matplotlib.pyplot as plt

from porcupy.cpo_class import CrestPorcupineOptimizer
from porcupy.utils.visualization_manager import CPOVisualizer
from porcupy.functions import rastrigin, sphere, rosenbrock, ackley


def run_population_dynamics_example():
    """Run the population dynamics visualization example."""
    print("Population Dynamics Visualization Example")
    print("========================================")
    
    # Define a 2D test function
    def func(x):
        return ackley(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -32.768 * np.ones(dimensions)
    ub = 32.768 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer with different population reduction strategies
    strategies = ['linear', 'cosine', 'exponential']
    results = {}
    visualizers = {}
    
    for strategy in strategies:
        print(f"\nRunning optimization with {strategy} population reduction strategy...")
        
        # Initialize the optimizer
        optimizer = CrestPorcupineOptimizer(
            pop_size=50,
            dimensions=dimensions,
            max_iter=100,
            lb=lb,
            ub=ub,
            cycles=5,
            pop_reduction_strategy=strategy
        )
        
        # Initialize the visualizer
        visualizer = CPOVisualizer(objective_func=func, bounds=bounds)
        
        # Run the optimization with visualization tracking
        def callback(iteration, positions, best_pos, best_cost, pop_size):
            # Generate random defense types for demonstration
            defenses = []
            for _ in range(len(positions)):
                defense = np.random.choice(['sight', 'sound', 'odor', 'physical'])
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
        result = optimizer.optimize(func, callback=callback)
        print(f"Optimization completed. Best cost: {result['best_cost']:.6f}")
        
        # Store results and visualizer
        results[strategy] = result
        visualizers[strategy] = visualizer
    
    # Compare population reduction strategies
    print("\nComparing population reduction strategies...")
    
    # Get a reference visualizer (any will do)
    ref_visualizer = next(iter(visualizers.values()))
    
    # Create population reduction strategies comparison
    fig1 = ref_visualizer.compare_reduction_strategies(
        max_iter=100,
        pop_size=50,
        cycles=5,
        strategies=strategies,
        save_path="population_reduction_strategies.png"
    )
    
    # Create population cycles visualization for each strategy
    for strategy, visualizer in visualizers.items():
        print(f"Creating population cycles visualization for {strategy} strategy...")
        fig = visualizer.visualize_population_cycles(
            cycles=5,
            max_iter=100,
            title=f"Population Size Cycles ({strategy.capitalize()} Reduction)",
            save_path=f"population_cycles_{strategy}.png"
        )
    
    # Create diversity history visualization for each strategy
    for strategy, visualizer in visualizers.items():
        print(f"Creating diversity history visualization for {strategy} strategy...")
        fig = visualizer.visualize_diversity_history(
            title=f"Population Diversity History ({strategy.capitalize()} Reduction)",
            save_path=f"diversity_history_{strategy}.png"
        )
    
    # Create diversity vs convergence visualization for each strategy
    for strategy, visualizer in visualizers.items():
        print(f"Creating diversity vs convergence visualization for {strategy} strategy...")
        fig = visualizer.visualize_diversity_vs_convergence(
            cycles=5,
            max_iter=100,
            title=f"Diversity vs Convergence ({strategy.capitalize()} Reduction)",
            save_path=f"diversity_vs_convergence_{strategy}.png"
        )
    
    # Create exploration vs exploitation visualization for one strategy
    print("Creating exploration vs exploitation visualization...")
    linear_visualizer = visualizers['linear']
    fig2 = linear_visualizer.visualize_exploration_exploitation(
        sample_iterations=[0, 20, 40, 60, 80, 99],  # Sample iterations
        save_path="exploration_exploitation.png"
    )
    
    # Create population diversity map for one strategy
    print("Creating population diversity map...")
    from porcupy.utils.population_visualization import plot_population_diversity_map
    
    # Use the linear strategy for this visualization
    linear_visualizer = visualizers['linear']
    position_history = linear_visualizer.position_history
    
    fig3 = plot_population_diversity_map(
        positions_history=position_history,
        bounds=bounds,
        sample_iterations=[0, 20, 40, 60, 80, 99],  # Sample iterations
        save_path="population_diversity_map.png"
    )
    
    # Create population cycle animation
    print("Creating population cycle animation...")
    from porcupy.utils.population_visualization import animate_population_cycle
    
    # Use the linear strategy for this animation
    linear_visualizer = visualizers['linear']
    position_history = linear_visualizer.position_history
    pop_size_history = linear_visualizer.pop_size_history
    
    # Use a subset of iterations for the animation to keep it manageable
    indices = np.arange(0, len(position_history), 2)
    position_subset = [position_history[i] for i in indices]
    pop_size_subset = [pop_size_history[i] for i in indices]
    
    anim = animate_population_cycle(
        positions_history=position_subset,
        pop_size_history=pop_size_subset,
        bounds=bounds,
        max_iter=len(indices),
        cycles=5,
        interval=200,
        save_path="population_cycle_animation.gif"
    )
    
    # Show all figures
    plt.show()
    
    print("\nAll visualizations completed and saved to current directory.")
    print("- population_reduction_strategies.png")
    print("- population_cycles_*.png (for each strategy)")
    print("- diversity_history_*.png (for each strategy)")
    print("- diversity_vs_convergence_*.png (for each strategy)")
    print("- exploration_exploitation.png")
    print("- population_diversity_map.png")
    print("- population_cycle_animation.gif")


if __name__ == "__main__":
    run_population_dynamics_example()
