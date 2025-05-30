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

from porcupy.cpo_class import CPO
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
    optimizer = CPO(
        dimensions=dimensions,
        bounds=(lb, ub),
        pop_size=30,
        max_iter=50,
        cycles=5
    )
    
    # Initialize the visualizer
    visualizer = CPOVisualizer(objective_func=func, bounds=bounds)
    
    # Run the optimization with history tracking enabled
    print("Running optimization with history tracking...")
    result = optimizer.optimize(func, verbose=True, track_history=True)
    best_pos, best_cost, cost_history = result
    
    # Get the optimization history from the optimizer
    positions_history = optimizer.positions_history
    defense_types_history = optimizer.defense_types_history
    pop_size_history = optimizer.pop_size_history
    best_positions_history = optimizer.best_positions_history
    fitness_history = optimizer.fitness_history
    
    # Record the optimization data in the visualizer
    for i in range(len(positions_history)):
        # Record the iteration data
        visualizer.record_iteration(
            positions=positions_history[i],
            best_position=best_positions_history[i] if i < len(best_positions_history) else best_pos,
            fitness=fitness_history[i] if i < len(fitness_history) else np.array([best_cost]),
            pop_size=pop_size_history[i],
            defense_types=defense_types_history[i]
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
    
    # Plot the 2D search space with final positions and color-coded defense mechanisms
    visualizer.visualize_porcupines_2d(
        iteration=-1,  # Last iteration
        title='Final Positions in 2D Search Space',
        save_path='2d_search_space.png'
    )
    
    # We'll use the enhanced visualization_defense_effectiveness method instead of manual plotting
    # This provides a more comprehensive view of the exploration-exploitation balance
    
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
    
    # 8. Defense effectiveness - Enhanced visualization
    print("\nGenerating enhanced defense mechanism effectiveness visualization...")
    fig8 = visualizer.visualize_defense_effectiveness(
        title="Defense Mechanism Effectiveness Analysis",
        figsize=(14, 10),
        save_path="defense_effectiveness.png"
    )
    
    print("This visualization provides four key insights:")
    print("1. Defense Mechanism Usage: Shows how each mechanism is used over iterations")
    print("2. Exploration-Exploitation Balance: Visualizes the balance between exploration (sight/sound) and exploitation (odor/physical)")
    print("3. Cumulative Fitness Improvement: Tracks how each defense mechanism contributes to fitness improvement")
    print("4. Overall Effectiveness: Pie chart showing the relative effectiveness of each mechanism")
    print("\nThe exploration-exploitation balance is particularly important in understanding the CPO algorithm's behavior.")
    print("Early iterations typically favor exploration, while later iterations shift toward exploitation.")
    print("This adaptive behavior helps the algorithm avoid local optima while still converging to good solutions.")
    
    # Display the figure for a few seconds to allow viewing
    plt.figure(fig8.number)
    plt.draw()
    plt.pause(2)  # Pause to show the figure
    
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
    """
    Demonstrate animation capabilities with enhanced exploration-exploitation balance visualization.
    """
    print("Demonstrating enhanced animation capabilities...")
    
    # Load the visualizer from the previous demo
    visualizer = demo_basic_visualizations()
    
    print("\nCreating enhanced animation with exploration-exploitation balance visualization...")
    print("This animation will show:")
    print("1. The main optimization process with color-coded defense mechanisms")
    print("2. A real-time plot of exploration vs. exploitation balance over iterations")
    print("3. A pie chart showing the current distribution of defense mechanisms")
    
    # Create an enhanced animation of the optimization process with defense mechanisms
    animation = visualizer.create_animation(
        positions_history=visualizer.position_history,
        best_position_history=visualizer.best_position_history,
        title='CPO Optimization Process with Defense Mechanisms',
        save_path='enhanced_optimization_animation.gif',
        fps=3,
        defense_types_history=visualizer.defense_history.get('defense_types', []),
        figsize=(14, 10),
        show_exploration_exploitation=True
    )
    
    print("\nAnimation saved to 'enhanced_optimization_animation.gif'")
    print("This visualization clearly shows how the algorithm balances exploration and exploitation:")
    print("- Blue/green points (sight/sound) represent exploration mechanisms")
    print("- Orange/red points (odor/physical) represent exploitation mechanisms")
    print("- The bottom-left chart shows how this balance changes over iterations")
    print("- The bottom-right pie chart shows the current distribution of mechanisms")
    
    # Create a standard animation without the exploration-exploitation balance for comparison
    print("\nCreating standard animation for comparison...")
    standard_animation = visualizer.create_animation(
        positions_history=visualizer.position_history,
        best_position_history=visualizer.best_position_history,
        title='Standard CPO Optimization Process',
        save_path='standard_optimization_animation.gif',
        fps=3,
        defense_types_history=visualizer.defense_history.get('defense_types', []),
        show_exploration_exploitation=False
    )
    
    print("\nStandard animation saved to 'standard_optimization_animation.gif'")
    
    # Create a trajectory visualization showing how the best solution evolves
    plt.figure(figsize=(10, 8))
    
    # Plot the objective function contour
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rastrigin(np.array([X[i, j], Y[i, j]]))
    
    plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Objective Function Value')
    
    # Plot the best position trajectory
    best_x = [pos[0] for pos in visualizer.best_position_history]
    best_y = [pos[1] for pos in visualizer.best_position_history]
    
    plt.plot(best_x, best_y, 'r-', linewidth=2, alpha=0.7, label='Best Position Trajectory')
    plt.scatter(best_x, best_y, c=range(len(best_x)), cmap='plasma', 
                s=50, zorder=5, label='Best Positions')
    
    # Mark the final best position
    plt.scatter(visualizer.best_position_history[-1][0], visualizer.best_position_history[-1][1], c='red', s=200, marker='*', 
                edgecolors='black', zorder=10, label='Final Best Position')
    
    plt.title('Best Position Trajectory During Optimization')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('best_position_trajectory.png')
    plt.close()
    
    print("Animation created and saved as 'optimization_animation.gif'")


def demo_interactive_dashboard():
    """Demonstrate the interactive dashboard."""
    print("Demonstrating interactive dashboard...")
    
    # Define a 2D test function
    def func(x, callback=None):
        return ackley(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -32.768 * np.ones(dimensions)
    ub = 32.768 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer
    optimizer = CPO(
        pop_size=30,
        dimensions=dimensions,
        max_iter=100,
        bounds=bounds,
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
    def func(x, callback=None):
        return sphere(x)
    
    # Define bounds for the search space
    dimensions = 10
    lb = -100 * np.ones(dimensions)
    ub = 100 * np.ones(dimensions)
    bounds = (lb, ub)
    
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
        optimizer = CPO(
            pop_size=pop_size,
            dimensions=dimensions,
            max_iter=100,
            bounds=bounds
        )
        
        # Run the optimization
        best_pos, best_cost, _ = optimizer.optimize(func)
        
        # Get the convergence history
        convergence_history = optimizer.cost_history
        
        # Update the dashboard
        dashboard.update(
            parameter_value=pop_size,
            result=best_cost,
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
