"""
3D Visualization Example for the Crested Porcupine Optimizer

This example demonstrates how to visualize the Crested Porcupine Optimizer
in a 3D search space, showing the porcupine positions and the objective function landscape.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy.cpo_class import CrestPorcupineOptimizer
from porcupy.functions import rastrigin, sphere, ackley
from porcupy.utils.enhanced_visualization import plot_3d_porcupines

def run_3d_visualization():
    """Run the 3D visualization example."""
    print("3D Visualization Example")
    print("=======================")
    
    # We'll visualize the Rastrigin function which has an interesting landscape
    func_name = "Rastrigin"
    func = rastrigin
    print(f"\nVisualizing {func_name} function in 3D...")
    
    # Define bounds for the search space (2D only for this visualization)
    dimensions = 2
    lb = -5.0 * np.ones(dimensions)
    ub = 5.0 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Create a 3D visualization with improved visibility
    create_3d_visualization(func, func_name, bounds)
    
    # Create an animation of the optimization process
    create_optimization_animation(func, func_name, bounds)
    
    print("\n3D visualization completed.")
    print("Visualization images and animation saved to current directory.")

def create_3d_visualization(func, func_name, bounds):
    """Create a 3D visualization with improved visibility."""
    # Create a simple population of porcupines for visualization
    pop_size = 20
    lb, ub = bounds
    positions = np.random.uniform(lb, ub, (pop_size, 2))
    
    # Calculate fitness for each position
    fitness = np.array([func(pos) for pos in positions])
    
    # Find the best position
    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx]
    
    # Generate random defense types for demonstration
    defense_types = []
    for _ in range(pop_size):
        defense = np.random.choice(['sight', 'sound', 'odor', 'physical'])
        defense_types.append(defense)
    
    # Create a grid of points for the surface
    resolution = 50
    x = np.linspace(lb[0], ub[0], resolution)
    y = np.linspace(lb[1], ub[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the function at each point
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])
    
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with transparency
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Cost')
    
    # Define colors for different defense mechanisms
    defense_colors = {
        'sight': 'blue',
        'sound': 'green',
        'odor': 'orange',
        'physical': 'red'
    }
    
    # Plot the porcupines with larger markers and on top of the surface
    for i, (pos, fit, defense) in enumerate(zip(positions, fitness, defense_types)):
        color = defense_colors.get(defense, 'white')
        # Add a small offset to ensure points are visible above the surface
        ax.scatter(pos[0], pos[1], fit + 2, c=color, edgecolors='black', s=150, zorder=10)
    
    # Plot the best position with a larger marker
    best_fitness = func(best_pos)
    ax.scatter(best_pos[0], best_pos[1], best_fitness + 3, c='red', s=250, marker='*', 
               label='Best Position', zorder=11)
    
    ax.set_title(f"3D {func_name} Function Landscape with Porcupines")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Cost')
    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                               markersize=10, label=defense) 
                     for defense, color in defense_colors.items()]
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                                    markersize=15, label='Best Position'))
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the figure
    plt.savefig(f"3d_{func_name.lower()}_visualization.png", dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

def create_optimization_animation(func, func_name, bounds):
    """Create an animation of the optimization process."""
    print("Creating optimization animation...")
    
    # Initialize the CPO optimizer
    lb, ub = bounds
    optimizer = CrestPorcupineOptimizer(
        pop_size=30,
        dimensions=2,
        max_iter=50,
        bounds=bounds
    )
    
    # Store positions and fitness values for animation
    all_positions = []
    all_fitness = []
    all_best_positions = []
    
    # Create a wrapper function that accepts a callback parameter but doesn't use it
    def objective_func(x, callback=None):
        return func(x)
    
    # Define a callback function to capture state at each iteration
    def callback(iteration, positions, best_pos, best_cost, pop_size):
        all_positions.append(positions.copy())
        fitness = np.array([func(pos) for pos in positions])
        all_fitness.append(fitness.copy())
        all_best_positions.append(best_pos.copy())
        print(f"Iteration {iteration}: Best Cost = {best_cost:.6f}, Population Size = {pop_size}")
        return False  # Continue optimization
    
    # Run the optimization with the callback
    optimizer.optimize(objective_func, callback=callback)
    
    # Create a grid of points for the surface
    resolution = 50
    x = np.linspace(lb[0], ub[0], resolution)
    y = np.linspace(lb[1], ub[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the function at each point
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])
    
    # Create the figure and 3D axis for animation
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with transparency
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Cost')
    
    # Set axis labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Cost')
    title = ax.set_title(f"Optimization Progress - Iteration 0")
    
    # Initialize scatter plot for porcupines
    scatter = ax.scatter([], [], [], c='blue', s=100, edgecolors='black')
    
    # Initialize scatter plot for best position
    best_scatter = ax.scatter([], [], [], c='red', s=200, marker='*')
    
    # Animation update function
    def update(frame):
        # Clear previous points
        ax.collections.remove(scatter)
        ax.collections.remove(best_scatter)
        
        # Update title
        title.set_text(f"Optimization Progress - Iteration {frame}")
        
        # Get positions and fitness for current frame
        positions = all_positions[frame]
        fitness = all_fitness[frame]
        best_pos = all_best_positions[frame]
        best_fitness = func(best_pos)
        
        # Plot porcupines with offset to ensure visibility
        scatter = ax.scatter(positions[:, 0], positions[:, 1], fitness + 2, 
                           c='blue', s=100, edgecolors='black', zorder=10)
        
        # Plot best position
        best_scatter = ax.scatter(best_pos[0], best_pos[1], best_fitness + 3, 
                                c='red', s=200, marker='*', zorder=11)
        
        return scatter, best_scatter, title
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(all_positions), 
                                  interval=200, blit=False)
    
    # Save the animation
    anim.save(f"{func_name.lower()}_optimization_animation.gif", writer='pillow', fps=5, dpi=100)
    
    print(f"Animation saved as {func_name.lower()}_optimization_animation.gif")
    
    # Show the final frame
    plt.show()

if __name__ == "__main__":
    run_3d_visualization()
