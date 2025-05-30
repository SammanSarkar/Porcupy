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

from porcupy.cpo_class import CPO
from porcupy.functions import rastrigin, sphere, ackley
from porcupy.utils.enhanced_visualization import plot_3d_porcupines

def run_3d_visualization():
    """Run the 3D visualization example."""
    print("3D Visualization Example")
    print("=======================")
    
    # Ask the user what they want to visualize
    print("\nChoose an option:")
    print("1. Static 3D visualization with visible defense mechanisms")
    print("2. Optimization animation")
    print("3. Both")
    
    choice = input("Enter your choice (1, 2, or 3): ")
    
    # We'll visualize the Rastrigin function which has an interesting landscape
    func_name = "Rastrigin"
    func = rastrigin
    
    # Define bounds for the search space (2D only for this visualization)
    dimensions = 2
    lb = -5.0 * np.ones(dimensions)
    ub = 5.0 * np.ones(dimensions)
    bounds = (lb, ub)
    
    if choice == '1' or choice == '3':
        print(f"\nVisualizing {func_name} function in 3D...")
        # Create a 3D visualization with improved visibility
        create_3d_visualization(func, func_name, bounds)
    
    if choice == '2' or choice == '3':
        print(f"\nCreating optimization animation for {func_name} function...")
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
    """Create an animation of the optimization process using a simple approach."""
    print("Creating optimization animation...")
    
    # Define parameters
    lb, ub = bounds
    num_frames = 20  # Number of frames to create
    pop_size = 30    # Initial population size
    
    # Store positions and fitness values for animation
    all_positions = []
    all_fitness = []
    all_best_positions = []
    
    # Create a simple simulation of the optimization process
    print("Simulating optimization process...")
    
    # Initialize random population
    positions = np.random.uniform(lb, ub, (pop_size, 2))
    fitness = np.array([func(pos) for pos in positions])
    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_cost = fitness[best_idx]
    
    # Store initial state
    all_positions.append(positions.copy())
    all_fitness.append(fitness.copy())
    all_best_positions.append(best_pos.copy())
    
    print(f"Initial state: Best Cost = {best_cost:.6f}, Population Size = {len(positions)}")
    
    # Simulate optimization steps
    for i in range(1, num_frames):
        # Reduce population size gradually
        current_pop_size = max(15, pop_size - i)
        positions = positions[:current_pop_size]
        
        # Move positions toward better solutions with some randomness
        for j in range(len(positions)):
            # Move toward best position with some randomness
            direction = best_pos - positions[j]
            step_size = 0.1 * np.random.random()
            positions[j] = positions[j] + step_size * direction
            
            # Add some random exploration
            positions[j] = positions[j] + 0.2 * np.random.normal(0, 1, 2)
            
            # Ensure positions stay within bounds
            positions[j] = np.clip(positions[j], lb, ub)
        
        # Evaluate fitness
        fitness = np.array([func(pos) for pos in positions])
        
        # Update best position if better found
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_cost:
            best_pos = positions[current_best_idx].copy()
            best_cost = fitness[current_best_idx]
        
        # Store state
        all_positions.append(positions.copy())
        all_fitness.append(fitness.copy())
        all_best_positions.append(best_pos.copy())
        
        print(f"Frame {i}/{num_frames-1}: Best Cost = {best_cost:.6f}, Population Size = {len(positions)}")
    
    print(f"Simulation completed with {len(all_positions)} frames.")
    
    if len(all_positions) == 0:
        print("No simulation data generated. Cannot create animation.")
        return
    
    if len(all_positions) == 0:
        print("No optimization data captured. Cannot create animation.")
        return
    
    print(f"Creating animation with {len(all_positions)} frames...")
    
    # Create a grid of points for the surface
    resolution = 30  # Reduced resolution for faster rendering
    x = np.linspace(lb[0], ub[0], resolution)
    y = np.linspace(lb[1], ub[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the function at each point
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])
    
    # Create a static visualization of the final state
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with transparency
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Cost')
    
    # Set axis labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Cost')
    
    # Get the final state
    final_positions = all_positions[-1]
    final_fitness = all_fitness[-1]
    final_best_pos = all_best_positions[-1]
    final_best_fitness = func(final_best_pos)
    
    # Plot the final positions
    ax.scatter(final_positions[:, 0], final_positions[:, 1], final_fitness + 2, 
               c='blue', s=100, edgecolors='black', zorder=10)
    
    # Plot the final best position
    ax.scatter(final_best_pos[0], final_best_pos[1], final_best_fitness + 3, 
               c='red', s=200, marker='*', zorder=11)
    
    ax.set_title(f"Final State After Optimization")
    
    # Save the final state visualization
    plt.savefig(f"{func_name.lower()}_final_state.png", dpi=300, bbox_inches='tight')
    print(f"Final state visualization saved as {func_name.lower()}_final_state.png")
    
    # Create an actual animation that can be viewed directly
    print("Creating animation...")
    
    # Create a figure for the animation
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface once (doesn't change during animation)
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Cost')
    
    # Set axis labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Cost')
    
    # Initialize scatter plots
    scatter = ax.scatter([], [], [], c='blue', s=100, edgecolors='black')
    best_scatter = ax.scatter([], [], [], c='red', s=200, marker='*')
    title = ax.set_title("Optimization Progress - Iteration 0")
    
    # Use a simpler approach for animation - recreate the plot each time
    def update(frame):
        # Clear the axis
        ax.clear()
        
        # Re-plot the surface
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        
        # Set axis labels
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Cost')
        ax.set_title(f"Optimization Progress - Iteration {frame}")
        
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
        
        return [surface, scatter, best_scatter]
    
    # Create the animation
    print("Generating animation (this may take a moment)...")
    anim = animation.FuncAnimation(fig, update, frames=len(all_positions), 
                                  interval=300, blit=False)
    
    # Save the animation as a GIF
    animation_path = f"{func_name.lower()}_optimization_animation.gif"
    anim.save(animation_path, writer='pillow', fps=3, dpi=100)
    print(f"Animation saved as {animation_path}")
    
    # Also save individual frames for reference
    frames_dir = "animation_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    print("Saving individual frames...")
    for i, (positions, fitness, best_pos) in enumerate(zip(all_positions, all_fitness, all_best_positions)):
        # Create a new figure for each frame
        frame_fig = plt.figure(figsize=(12, 10))
        frame_ax = frame_fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        frame_surface = frame_ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        
        # Set axis labels and title
        frame_ax.set_xlabel('x1')
        frame_ax.set_ylabel('x2')
        frame_ax.set_zlabel('Cost')
        frame_ax.set_title(f"Optimization Progress - Iteration {i}")
        
        # Plot porcupines
        frame_ax.scatter(positions[:, 0], positions[:, 1], fitness + 2, 
                   c='blue', s=100, edgecolors='black', zorder=10)
        
        # Plot best position
        best_fitness = func(best_pos)
        frame_ax.scatter(best_pos[0], best_pos[1], best_fitness + 3, 
                   c='red', s=200, marker='*', zorder=11)
        
        # Save the frame
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(frame_fig)
    
    print("All frames saved to the 'animation_frames' directory.")
    
    # Display the animation in the window
    plt.show()

if __name__ == "__main__":
    run_3d_visualization()
