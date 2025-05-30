"""
Population dynamics visualization module for the Crested Porcupine Optimizer (CPO).

This module provides specialized visualization tools for the population dynamics
of the CPO algorithm, including cyclic population reduction and diversity visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional, Union, Dict, Any, Callable


def plot_population_reduction_strategies(
    max_iter: int,
    pop_size: int,
    cycles: int,
    strategies: List[str] = ['linear', 'cosine', 'exponential'],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot and compare different population reduction strategies.
    
    Parameters
    ----------
    max_iter : int
        Maximum number of iterations.
    pop_size : int
        Initial population size.
    cycles : int
        Number of cycles.
    strategies : list, optional
        List of reduction strategies to compare (default: ['linear', 'cosine', 'exponential']).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 6)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for different strategies
    strategy_colors = {
        'linear': 'blue',
        'cosine': 'green',
        'exponential': 'red'
    }
    
    # Generate iterations
    iterations = np.arange(max_iter)
    
    # Calculate cycle length
    cycle_length = max_iter // cycles
    
    # Plot each reduction strategy
    for strategy in strategies:
        pop_sizes = []
        
        for i in range(max_iter):
            cycle = i // cycle_length
            cycle_progress = (i % cycle_length) / cycle_length
            
            if strategy == 'linear':
                # Linear reduction
                current_pop = pop_size - (pop_size - 10) * cycle_progress
            elif strategy == 'cosine':
                # Cosine reduction
                current_pop = pop_size - (pop_size - 10) * (1 - np.cos(cycle_progress * np.pi)) / 2
            elif strategy == 'exponential':
                # Exponential reduction
                current_pop = pop_size - (pop_size - 10) * (1 - np.exp(-5 * cycle_progress)) / (1 - np.exp(-5))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            pop_sizes.append(max(10, int(current_pop)))
        
        # Plot the strategy
        ax.plot(iterations, pop_sizes, color=strategy_colors.get(strategy, 'black'), 
               linewidth=2, label=strategy.capitalize())
    
    # Add cycle boundaries
    for i in range(1, cycles):
        cycle_boundary = i * cycle_length
        ax.axvline(x=cycle_boundary, color='gray', linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_title('Population Reduction Strategies Comparison')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Population Size')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Add cycle labels
    for i in range(cycles):
        cycle_middle = i * cycle_length + cycle_length // 2
        ax.text(cycle_middle, pop_size + 2, f"Cycle {i+1}",
               horizontalalignment='center', color='gray')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_population_diversity_map(
    positions_history: List[np.ndarray],
    bounds: Tuple[np.ndarray, np.ndarray],
    sample_iterations: List[int],
    figsize: Tuple[int, int] = (15, 10),
    cmap: str = 'viridis',
    save_path: Optional[str] = None
):
    """
    Create a grid of plots showing population diversity at different iterations.
    
    Parameters
    ----------
    positions_history : list
        List of position arrays at each iteration, each with shape (pop_size, 2).
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    sample_iterations : list
        List of iteration indices to visualize.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (15, 10)).
    cmap : str, optional
        Colormap for the density plot (default: 'viridis').
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if not positions_history:
        raise ValueError("Position history is empty")
    
    if positions_history[0].shape[1] != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    # Determine the grid size
    n_samples = len(sample_iterations)
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    # Set the plot limits based on the bounds
    lb, ub = bounds
    
    # Create a grid for density estimation
    resolution = 50
    x = np.linspace(lb[0], ub[0], resolution)
    y = np.linspace(lb[1], ub[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Plot each sampled iteration
    for i, iter_idx in enumerate(sample_iterations):
        if iter_idx < len(positions_history):
            ax = axes[i]
            positions = positions_history[iter_idx]
            
            # Calculate density
            density = np.zeros((resolution, resolution))
            for pos in positions:
                # Find the closest grid point
                x_idx = np.argmin(np.abs(x - pos[0]))
                y_idx = np.argmin(np.abs(y - pos[1]))
                density[y_idx, x_idx] += 1
            
            # Apply Gaussian blur for smoother density
            from scipy.ndimage import gaussian_filter
            density = gaussian_filter(density, sigma=1.0)
            
            # Plot density
            contour = ax.contourf(X, Y, density, 20, cmap=cmap, alpha=0.8)
            
            # Plot positions
            ax.scatter(positions[:, 0], positions[:, 1], c='white', edgecolors='black', s=30)
            
            # Calculate diversity metrics
            mean_pos = np.mean(positions, axis=0)
            std_pos = np.std(positions, axis=0)
            
            # Set labels and title
            ax.set_title(f'Iteration {iter_idx+1}')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            
            # Add diversity information
            diversity_text = f"Population Size: {len(positions)}\n"
            diversity_text += f"Std Dev x1: {std_pos[0]:.4f}\n"
            diversity_text += f"Std Dev x2: {std_pos[1]:.4f}"
            
            ax.text(0.05, 0.95, diversity_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Set axis limits
            ax.set_xlim(lb[0], ub[0])
            ax.set_ylim(lb[1], ub[1])
    
    # Hide any unused subplots
    for i in range(len(sample_iterations), len(axes)):
        axes[i].axis('off')
    
    # Add a colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(contour, cax=cbar_ax, label='Population Density')
    
    # Set a common title
    fig.suptitle('Population Diversity Map Across Iterations', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def animate_population_cycle(
    positions_history: List[np.ndarray],
    pop_size_history: List[int],
    bounds: Tuple[np.ndarray, np.ndarray],
    max_iter: int,
    cycles: int,
    interval: int = 200,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis',
    save_path: Optional[str] = None,
    dpi: int = 100
):
    """
    Create an animation showing population dynamics throughout cycles.
    
    Parameters
    ----------
    positions_history : list
        List of position arrays at each iteration, each with shape (pop_size, 2).
    pop_size_history : list
        List of population sizes at each iteration.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    max_iter : int
        Maximum number of iterations.
    cycles : int
        Number of cycles.
    interval : int, optional
        Interval between frames in milliseconds (default: 200).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 8)).
    cmap : str, optional
        Colormap for the density plot (default: 'viridis').
    save_path : str, optional
        Path to save the animation. If None, the animation is not saved (default: None).
    dpi : int, optional
        DPI for the saved animation (default: 100).
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The created animation.
    """
    if not positions_history:
        raise ValueError("Position history is empty")
    
    if positions_history[0].shape[1] != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    # Create the figure and axes
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
    
    # Population size plot
    ax_pop = fig.add_subplot(gs[0, 0])
    pop_line, = ax_pop.plot([], [], 'b-', linewidth=2)
    ax_pop.set_title('Population Size')
    ax_pop.set_xlabel('Iterations')
    ax_pop.set_ylabel('Population Size')
    ax_pop.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate cycle length
    cycle_length = max_iter // cycles
    
    # Add cycle boundaries to population plot
    for i in range(1, cycles):
        cycle_boundary = i * cycle_length
        if cycle_boundary < max_iter:
            ax_pop.axvline(x=cycle_boundary, color='r', linestyle='--', alpha=0.7)
    
    # Set axis limits for population plot
    ax_pop.set_xlim(0, max_iter)
    ax_pop.set_ylim(0, max(pop_size_history) * 1.1)
    
    # Add cycle labels
    for i in range(cycles):
        cycle_middle = i * cycle_length + cycle_length // 2
        if cycle_middle < max_iter:
            ax_pop.text(cycle_middle, max(pop_size_history) * 1.05, f"Cycle {i+1}",
                      horizontalalignment='center', color='red')
    
    # Diversity metrics plot
    ax_div = fig.add_subplot(gs[0, 1])
    div_line, = ax_div.plot([], [], 'g-', linewidth=2)
    ax_div.set_title('Diversity')
    ax_div.set_xlabel('Iterations')
    ax_div.set_ylabel('Std Dev')
    ax_div.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits for diversity plot
    ax_div.set_xlim(0, max_iter)
    
    # Positions plot
    ax_pos = fig.add_subplot(gs[1, :])
    scatter = ax_pos.scatter([], [], c='blue', edgecolors='black', s=50)
    
    # Set the plot limits based on the bounds
    lb, ub = bounds
    ax_pos.set_xlim(lb[0], ub[0])
    ax_pos.set_ylim(lb[1], ub[1])
    
    # Set labels and title for positions plot
    ax_pos.set_title('Population Distribution')
    ax_pos.set_xlabel('x1')
    ax_pos.set_ylabel('x2')
    ax_pos.grid(True, linestyle='--', alpha=0.3)
    
    # Text for iteration and cycle information
    info_text = ax_pos.text(0.02, 0.98, '', transform=ax_pos.transAxes,
                          verticalalignment='top', fontsize=10,
                          bbox=dict(facecolor='white', alpha=0.7))
    
    # Calculate diversity history
    diversity_history = []
    for positions in positions_history:
        if len(positions) > 1:
            std_pos = np.std(positions, axis=0)
            diversity = np.mean(std_pos)
        else:
            diversity = 0
        diversity_history.append(diversity)
    
    # Animation update function
    def update(frame):
        # Update population size plot
        iterations = np.arange(frame + 1)
        pop_sizes = pop_size_history[:frame + 1]
        pop_line.set_data(iterations, pop_sizes)
        
        # Update diversity plot
        diversities = diversity_history[:frame + 1]
        div_line.set_data(iterations, diversities)
        
        # Update positions plot
        positions = positions_history[frame]
        scatter.set_offsets(positions)
        
        # Determine current cycle
        current_cycle = frame // cycle_length + 1
        cycle_progress = (frame % cycle_length) / cycle_length * 100
        
        # Update information text
        info_text.set_text(f"Iteration: {frame+1}/{max_iter}\n"
                          f"Cycle: {current_cycle}/{cycles} ({cycle_progress:.1f}%)\n"
                          f"Population Size: {pop_size_history[frame]}\n"
                          f"Diversity: {diversity_history[frame]:.4f}")
        
        # Return all artists that need to be redrawn
        return pop_line, div_line, scatter, info_text
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(positions_history),
                        interval=interval, blit=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the animation if a path is provided
    if save_path:
        anim.save(save_path, dpi=dpi, writer='pillow')
    
    return anim


def plot_exploration_exploitation_balance(
    positions_history: List[np.ndarray],
    best_positions_history: List[np.ndarray],
    bounds: Tuple[np.ndarray, np.ndarray],
    sample_iterations: List[int],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Plot the balance between exploration and exploitation at different iterations.
    
    Parameters
    ----------
    positions_history : list
        List of position arrays at each iteration, each with shape (pop_size, 2).
    best_positions_history : list
        List of best position arrays at each iteration, each with shape (2,).
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    sample_iterations : list
        List of iteration indices to visualize.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (15, 10)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if not positions_history:
        raise ValueError("Position history is empty")
    
    if positions_history[0].shape[1] != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    # Determine the grid size
    n_samples = len(sample_iterations)
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    # Set the plot limits based on the bounds
    lb, ub = bounds
    
    # Plot each sampled iteration
    for i, iter_idx in enumerate(sample_iterations):
        if iter_idx < len(positions_history):
            ax = axes[i]
            positions = positions_history[iter_idx]
            best_pos = best_positions_history[iter_idx]
            
            # Calculate distances to best position
            distances = np.linalg.norm(positions - best_pos, axis=1)
            
            # Normalize distances for coloring
            if np.max(distances) > 0:
                normalized_distances = distances / np.max(distances)
            else:
                normalized_distances = np.zeros_like(distances)
            
            # Create a colormap: blue (close to best) to red (far from best)
            cmap = plt.cm.coolwarm
            colors = cmap(normalized_distances)
            
            # Plot positions with distance-based coloring
            scatter = ax.scatter(positions[:, 0], positions[:, 1], c=colors, 
                               edgecolors='black', s=50)
            
            # Plot best position
            ax.scatter(best_pos[0], best_pos[1], c='yellow', edgecolors='black', 
                      s=150, marker='*', label='Best Position')
            
            # Calculate exploration vs exploitation metrics
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Determine if this iteration is more exploratory or exploitative
            if mean_dist > (ub[0] - lb[0]) * 0.2:  # Arbitrary threshold
                phase = "Exploration"
            else:
                phase = "Exploitation"
            
            # Set labels and title
            ax.set_title(f'Iteration {iter_idx+1}: {phase}')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            
            # Add metrics information
            metrics_text = f"Mean Distance: {mean_dist:.4f}\n"
            metrics_text += f"Std Dev Distance: {std_dist:.4f}\n"
            metrics_text += f"Population Size: {len(positions)}"
            
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.7))
            
            # Set axis limits
            ax.set_xlim(lb[0], ub[0])
            ax.set_ylim(lb[1], ub[1])
            
            # Add legend
            ax.legend(loc='lower right')
    
    # Hide any unused subplots
    for i in range(len(sample_iterations), len(axes)):
        axes[i].axis('off')
    
    # Add a colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Distance to Best Position (Normalized)')
    
    # Set a common title
    fig.suptitle('Exploration vs Exploitation Balance Across Iterations', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_diversity_vs_convergence(
    diversity_history: List[float],
    fitness_history: List[float],
    cycles: int,
    max_iter: int,
    title: str = "Diversity vs Convergence",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot the relationship between population diversity and convergence.
    
    Parameters
    ----------
    diversity_history : list
        List of diversity measures at each iteration.
    fitness_history : list
        List of best fitness values at each iteration.
    cycles : int
        Number of cycles.
    max_iter : int
        Maximum number of iterations.
    title : str, optional
        Title of the plot (default: "Diversity vs Convergence").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 6)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Calculate cycle length
    cycle_length = max_iter // cycles
    
    # Generate iterations
    iterations = np.arange(1, len(diversity_history) + 1)
    
    # Plot diversity
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Diversity', color='blue')
    ax1.plot(iterations, diversity_history, 'b-', linewidth=2, label='Diversity')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a second y-axis for fitness
    ax2 = ax1.twinx()
    ax2.set_ylabel('Best Fitness', color='red')
    ax2.plot(iterations, fitness_history, 'r-', linewidth=2, label='Best Fitness')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add cycle boundaries
    for i in range(1, cycles):
        cycle_boundary = i * cycle_length
        if cycle_boundary < len(iterations):
            ax1.axvline(x=cycle_boundary, color='gray', linestyle='--', alpha=0.7)
    
    # Add cycle labels
    for i in range(cycles):
        cycle_middle = i * cycle_length + cycle_length // 2
        if cycle_middle < len(iterations):
            ax1.text(cycle_middle, max(diversity_history) * 1.05, f"Cycle {i+1}",
                    horizontalalignment='center', color='gray')
    
    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Set title
    plt.title(title)
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate correlation between diversity and fitness improvement
    fitness_improvements = np.zeros_like(fitness_history)
    for i in range(1, len(fitness_history)):
        fitness_improvements[i] = fitness_history[i-1] - fitness_history[i]
    
    correlation = np.corrcoef(diversity_history[1:], fitness_improvements[1:])[0, 1]
    
    # Add correlation information
    correlation_text = f"Correlation between Diversity and Fitness Improvement: {correlation:.4f}"
    plt.figtext(0.5, 0.01, correlation_text, ha='center', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Helper function to calculate population diversity
def calculate_diversity(positions: np.ndarray) -> float:
    """
    Calculate the diversity of a population based on average pairwise distance.
    
    Parameters
    ----------
    positions : ndarray
        Positions of the porcupines, shape (pop_size, dimensions).
    
    Returns
    -------
    float
        Diversity measure.
    """
    n_particles = positions.shape[0]
    
    if n_particles <= 1:
        return 0.0
    
    # Calculate pairwise distances
    distances = np.zeros((n_particles, n_particles))
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            distances[i, j] = np.linalg.norm(positions[i] - positions[j])
            distances[j, i] = distances[i, j]
    
    # Average pairwise distance
    return np.sum(distances) / (n_particles * (n_particles - 1))
