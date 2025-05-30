import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Optional, Callable, Union

"""
Enhanced visualization module for the Crested Porcupine Optimizer (CPO).

This module provides specialized visualization tools for the CPO algorithm,
highlighting its unique features such as the four defense mechanisms and
cyclic population reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from typing import List, Tuple, Optional, Union, Dict, Any, Callable


def plot_defense_mechanisms(
    defense_history: Union[Dict[str, List[int]], List[List[str]]],
    title: str = "Defense Mechanism Activation",
    figsize: Tuple[int, int] = (12, 6),
    colors: Dict[str, str] = None,
    save_path: Optional[str] = None
):
    """
    Plot the activation frequency of each defense mechanism over iterations.
    
    Parameters
    ----------
    defense_history : dict or list
        Either:
        1. Dictionary with keys 'sight', 'sound', 'odor', 'physical' and values
           as lists of counts for each iteration, or
        2. List of lists containing defense mechanisms used by each porcupine at each iteration.
    title : str, optional
        Title of the plot (default: "Defense Mechanism Activation").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 6)).
    colors : dict, optional
        Dictionary mapping defense mechanisms to colors.
        Default: {'sight': 'blue', 'sound': 'green', 'odor': 'orange', 'physical': 'red'}.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if colors is None:
        colors = {
            'sight': 'blue',
            'sound': 'green',
            'odor': 'orange',
            'physical': 'red'
        }
    
    # Convert list-based defense history to dictionary format if needed
    if isinstance(defense_history, list):
        # This is a list of lists with defense types for each porcupine at each iteration
        defense_counts = {
            'sight': [],
            'sound': [],
            'odor': [],
            'physical': []
        }
        
        # Count each defense type for each iteration
        for defenses in defense_history:
            defense_counts['sight'].append(defenses.count('sight'))
            defense_counts['sound'].append(defenses.count('sound'))
            defense_counts['odor'].append(defenses.count('odor'))
            defense_counts['physical'].append(defenses.count('physical'))
            
        defense_history = defense_counts
    
    plt.figure(figsize=figsize)
    
    iterations = np.arange(1, len(list(defense_history.values())[0]) + 1)
    
    for mechanism, counts in defense_history.items():
        plt.plot(iterations, counts, color=colors[mechanism], linewidth=2, label=mechanism.capitalize())
    
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Activation Count")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_population_cycles(
    pop_size_history: List[int],
    cycles: int,
    max_iter: int,
    title: str = "Population Size Cycles",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot the population size history with cycle boundaries highlighted.
    
    Parameters
    ----------
    pop_size_history : list
        List of population sizes at each iteration.
    cycles : int
        Number of cycles used in the optimization.
    max_iter : int
        Maximum number of iterations.
    title : str, optional
        Title of the plot (default: "Population Size Cycles").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 6)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    plt.figure(figsize=figsize)
    
    iterations = np.arange(1, len(pop_size_history) + 1)
    
    # Plot population size
    plt.plot(iterations, pop_size_history, 'b-', linewidth=2, label='Population Size')
    
    # Add cycle boundaries
    cycle_length = max_iter // cycles
    for i in range(1, cycles):
        cycle_boundary = i * cycle_length
        if cycle_boundary < len(iterations):
            plt.axvline(x=cycle_boundary, color='r', linestyle='--', alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Population Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add cycle labels
    for i in range(cycles):
        cycle_middle = i * cycle_length + cycle_length // 2
        if cycle_middle < len(iterations):
            plt.text(cycle_middle, min(pop_size_history) - 2, f"Cycle {i+1}",
                    horizontalalignment='center', color='red')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_diversity_history(
    diversity_history: List[float],
    title: str = "Population Diversity History",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot the diversity history of the population.
    
    Parameters
    ----------
    diversity_history : list
        List of diversity measures at each iteration.
    title : str, optional
        Title of the plot (default: "Population Diversity History").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    plt.figure(figsize=figsize)
    
    iterations = np.arange(1, len(diversity_history) + 1)
    
    plt.plot(iterations, diversity_history, 'g-', linewidth=2)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Diversity Measure")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_2d_porcupines(
    positions: np.ndarray,
    func: Callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    best_pos: Optional[np.ndarray] = None,
    defense_types: Optional[List[str]] = None,
    title: str = "Porcupine Positions",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    contour_levels: int = 20,
    quill_length: float = 0.5,
    save_path: Optional[str] = None
):
    """
    Plot porcupines in 2D search space with quill-like directional indicators.
    
    Parameters
    ----------
    positions : ndarray
        Current positions of the porcupines, shape (pop_size, 2).
    func : callable
        The objective function to visualize.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    best_pos : ndarray, optional
        Global best position, shape (2,).
    defense_types : list, optional
        List of defense mechanisms used by each porcupine.
        Options: 'sight', 'sound', 'odor', 'physical'.
    title : str, optional
        Title of the plot (default: "Porcupine Positions").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 8)).
    cmap : str, optional
        Colormap for the contour plot (default: 'viridis').
    contour_levels : int, optional
        Number of contour levels (default: 20).
    quill_length : float, optional
        Length of the directional quills (default: 0.5).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if len(bounds[0]) != 2 or len(bounds[1]) != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    lb, ub = bounds
    
    # Create a grid of points
    resolution = 100
    x = np.linspace(lb[0], ub[0], resolution)
    y = np.linspace(lb[1], ub[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the function at each point
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot the contour
    contour = plt.contourf(X, Y, Z, contour_levels, cmap=cmap, alpha=0.8)
    plt.colorbar(contour, label='Cost')
    
    # Define colors for different defense mechanisms
    defense_colors = {
        'sight': 'blue',
        'sound': 'green',
        'odor': 'orange',
        'physical': 'red'
    }
    
    # Plot the porcupines with quill-like indicators
    if defense_types is not None:
        for i, (pos, defense) in enumerate(zip(positions, defense_types)):
            color = defense_colors.get(defense, 'white')
            plt.scatter(pos[0], pos[1], c=color, edgecolors='black', s=80)
            
            # Add quills (8 directions)
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx = quill_length * np.cos(angle)
                dy = quill_length * np.sin(angle)
                plt.arrow(pos[0], pos[1], dx, dy, head_width=0.1, head_length=0.1, 
                         fc=color, ec=color, alpha=0.7)
    else:
        plt.scatter(positions[:, 0], positions[:, 1], c='white', edgecolors='black', s=80)
    
    # Plot the best position if provided
    if best_pos is not None:
        plt.scatter(best_pos[0], best_pos[1], c='red', s=150, marker='*', 
                   label='Best Position')
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend for defense mechanisms if used
    if defense_types is not None:
        handles = []
        labels = []
        for defense, color in defense_colors.items():
            if defense in defense_types:
                handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                        markersize=10, label=defense.capitalize()))
                labels.append(defense.capitalize())
        
        if best_pos is not None:
            handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                                     markersize=15, label='Best Position'))
            labels.append('Best Position')
        
        plt.legend(handles=handles, labels=labels)
    else:
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def animate_porcupines_2d(
    position_history: List[np.ndarray],
    func: Callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    defense_history: Optional[List[List[str]]] = None,
    best_pos_history: Optional[List[np.ndarray]] = None,
    best_cost_history: Optional[List[float]] = None,
    interval: int = 200,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = 'viridis',
    contour_levels: int = 20,
    quill_length: float = 0.5,
    save_path: Optional[str] = None,
    dpi: int = 100,
    show_trail: bool = True,
    max_trail_length: int = 20,
    show_convergence: bool = True
):
    """
    Create an enhanced animation of porcupines moving in 2D search space with additional visual feedback.
    
    Parameters
    ----------
    position_history : list
        List of position arrays at each iteration, each with shape (pop_size, 2).
    func : callable
        The objective function to visualize.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    defense_history : list, optional
        List of lists containing defense mechanisms used by each porcupine at each iteration.
    best_pos_history : list, optional
        List of best positions at each iteration, each with shape (2,).
    best_cost_history : list, optional
        List of best costs at each iteration.
    interval : int, optional
        Interval between frames in milliseconds (default: 200).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (14, 10)).
    cmap : str, optional
        Colormap for the contour plot (default: 'viridis').
    contour_levels : int, optional
        Number of contour levels (default: 20).
    quill_length : float, optional
        Length of the directional quills (default: 0.5).
    save_path : str, optional
        Path to save the animation. If None, the animation is not saved (default: None).
    dpi : int, optional
        DPI for the saved animation (default: 100).
    show_trail : bool, optional
        Whether to show the trail of best positions (default: True).
    max_trail_length : int, optional
        Maximum number of positions to show in the trail (default: 20).
    show_convergence : bool, optional
        Whether to show the convergence plot (default: True).
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The created animation.
    """
    if len(bounds[0]) != 2 or len(bounds[1]) != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    lb, ub = bounds
    
    # Create a grid of points
    resolution = 100
    x = np.linspace(lb[0], ub[0], resolution)
    y = np.linspace(lb[1], ub[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the function at each point
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])
    
    # Create the figure with a grid layout
    fig = plt.figure(figsize=figsize)
    
    # Create main plot for porcupines
    if show_convergence and best_cost_history is not None:
        # Create a 2x2 grid with the main plot on the left and convergence plot on top right
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_conv = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[1, 1])
    else:
        # Single plot layout if no convergence plot
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax_main = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])
        ax_conv = None
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Main contour plot
    contour = ax_main.contourf(X, Y, Z, contour_levels, cmap=cmap, alpha=0.8)
    plt.colorbar(contour, ax=ax_main, label='Cost')
    
    # Set up convergence plot if enabled
    if ax_conv is not None:
        ax_conv.set_title('Convergence')
        ax_conv.set_xlabel('Iteration')
        ax_conv.set_ylabel('Best Cost')
        ax_conv.grid(True, linestyle='--', alpha=0.3)
        ax_conv.set_yscale('log')
        convergence_line, = ax_conv.plot([], [], 'b-', linewidth=2)
        current_point = ax_conv.plot([], [], 'ro', markersize=6)[0]
    
    # Set up info panel
    ax_info.axis('off')
    info_text = ax_info.text(0.1, 0.9, '', transform=ax_info.transAxes, 
                           verticalalignment='top', fontsize=10)
    
    # Add progress bar
    progress_bar = ax_info.barh(y=0.8, width=0, height=0.1, color='blue', alpha=0.5, 
                              left=0.1)[0]
    
    # Define colors for different defense mechanisms
    defense_colors = {
        'sight': 'blue',
        'sound': 'green',
        'odor': 'orange',
        'physical': 'red'
    }
    
    # Initialize scatter plots and quills for porcupines
    scatter_porcupines = ax_main.scatter([], [], c='white', edgecolors='black', s=80, zorder=5)
    quills = []
    
    # Create quills for each porcupine (8 directions per porcupine)
    if position_history and len(position_history) > 0:
        n_porcupines = position_history[0].shape[0]
        for i in range(n_porcupines):
            porcupine_quills = []
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                quill, = ax_main.plot([], [], 'k-', lw=1, alpha=0.7, zorder=4)
                porcupine_quills.append(quill)
            quills.append(porcupine_quills)
    
    # Initialize best position marker with a star and pulsing effect
    scatter_best = ax_main.scatter([], [], c='red', s=200, marker='*', 
                                 label='Best Position', zorder=10)
    
    # Initialize best position trail
    if show_trail and best_pos_history:
        trail_length = min(max_trail_length, len(best_pos_history))
        trail_alphas = np.linspace(0.2, 0.8, trail_length)
        trail = ax_main.plot([], [], 'r-', alpha=0.5, linewidth=2, zorder=3)[0]
    
    # Set labels and title for main plot
    ax_main.set_xlabel('x1')
    ax_main.set_ylabel('x2')
    ax_main.set_title('Porcupine Optimization Process')
    ax_main.grid(True, linestyle='--', alpha=0.3)
    
    # Set axis limits with a small margin
    margin_x = 0.1 * (ub[0] - lb[0])
    margin_y = 0.1 * (ub[1] - lb[1])
    ax_main.set_xlim(lb[0] - margin_x, ub[0] + margin_x)   
    ax_main.set_ylim(lb[1] - margin_y, ub[1] + margin_y)
    
    # Text for iteration number
    iteration_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes, 
                                verticalalignment='top', fontsize=10,
                                bbox=dict(facecolor='white', alpha=0.7, pad=2))
    
    # Create legend for defense mechanisms
    if defense_history is not None:
        legend_elements = []
        for defense, color in defense_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, 
                                            label=defense.capitalize()))
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', 
                                        markerfacecolor='red', markersize=15, 
                                        label='Best Position'))
        ax_main.legend(handles=legend_elements, loc='lower right')
    
    # Animation update function
    def update(frame):
        artists = []
        
        # Get current positions
        positions = position_history[frame]
        
        # Update porcupine positions
        scatter_porcupines.set_offsets(positions)
        artists.append(scatter_porcupines)
        
        # Update porcupine colors based on defense mechanisms
        if defense_history is not None and frame < len(defense_history):
            defense_types = defense_history[frame]
            colors = [defense_colors.get(defense, 'white') for defense in defense_types]
            scatter_porcupines.set_color(colors)
        
        # Update quills
        for i, pos in enumerate(positions):
            for j, angle in enumerate(np.linspace(0, 2*np.pi, 8, endpoint=False)):
                dx = quill_length * np.cos(angle)
                dy = quill_length * np.sin(angle)
                quills[i][j].set_data([pos[0], pos[0] + dx], [pos[1], pos[1] + dy])
                
                # Update quill color based on defense mechanism
                if defense_history is not None and frame < len(defense_history):
                    defense = defense_history[frame][i]
                    color = defense_colors.get(defense, 'black')
                    quills[i][j].set_color(color)
                
                artists.append(quills[i][j])
        
        # Update best position with pulsing effect
        if best_pos_history is not None and frame < len(best_pos_history):
            best_pos = best_pos_history[frame]
            scatter_best.set_offsets([best_pos])
            
            # Pulsing effect
            size = 200 + 50 * np.sin(frame * 0.5)
            scatter_best.set_sizes([size])
            artists.append(scatter_best)
            
            # Update trail
            if show_trail and frame > 0:
                start_idx = max(0, frame - max_trail_length + 1)
                trail_x = [p[0] for p in best_pos_history[start_idx:frame+1]]
                trail_y = [p[1] for p in best_pos_history[start_idx:frame+1]]
                trail.set_data(trail_x, trail_y)
                artists.append(trail)
        
        # Update convergence plot
        if ax_conv is not None and best_cost_history is not None and frame < len(best_cost_history):
            # Update convergence line
            iterations = range(frame + 1)
            convergence_line.set_data(iterations, best_cost_history[:frame+1])
            
            # Update current point
            current_point.set_data([frame], [best_cost_history[frame]])
            
            # Adjust axes limits
            ax_conv.relim()
            ax_conv.autoscale_view()
            
            artists.extend([convergence_line, current_point])
        
        # Update info text
        info = []
        info.append(f"Iteration: {frame+1}/{len(position_history)}")
        
        if best_cost_history is not None and frame < len(best_cost_history):
            info.append(f"Best Cost: {best_cost_history[frame]:.4e}")
        
        if best_pos_history is not None and frame < len(best_pos_history):
            best_pos = best_pos_history[frame]
            info.append(f"Best Position: ({best_pos[0]:.6f}, {best_pos[1]:.6f})")
        
        info_text.set_text('\n'.join(info))
        artists.append(info_text)
        
        # Update progress bar
        progress = (frame + 1) / len(position_history)
        progress_bar.set_width(progress)
        
        # Update iteration text
        iteration_text.set_text(f'Iteration: {frame+1}/{len(position_history)}')
        artists.append(iteration_text)
        
        return artists
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(position_history), 
                        interval=interval, blit=True)
    
    # Save the animation if a path is provided
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # Use 'ffmpeg' for better quality if available, otherwise fall back to 'pillow'
        try:
            anim.save(save_path, dpi=dpi, writer='ffmpeg', 
                     fps=1000/interval, bitrate=1800)
        except:
            try:
                anim.save(save_path, dpi=dpi, writer='pillow')
            except Exception as e:
                print(f"Warning: Could not save animation: {e}")
    
    return anim


def plot_3d_porcupines(
    positions: np.ndarray,
    fitness: np.ndarray,
    func: Callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    best_pos: Optional[np.ndarray] = None,
    defense_types: Optional[List[str]] = None,
    title: str = "3D Porcupine Positions",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    alpha: float = 0.7,
    save_path: Optional[str] = None
):
    """
    Plot porcupines in 3D search space.
    
    Parameters
    ----------
    positions : ndarray
        Current positions of the porcupines, shape (pop_size, 2).
    fitness : ndarray
        Fitness values of the porcupines, shape (pop_size,).
    func : callable
        The objective function to visualize.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    best_pos : ndarray, optional
        Global best position, shape (2,).
    defense_types : list, optional
        List of defense mechanisms used by each porcupine.
        Options: 'sight', 'sound', 'odor', 'physical'.
    title : str, optional
        Title of the plot (default: "3D Porcupine Positions").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 10)).
    cmap : str, optional
        Colormap for the surface plot (default: 'viridis').
    alpha : float, optional
        Transparency of the surface (default: 0.7).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if len(bounds[0]) != 2 or len(bounds[1]) != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    lb, ub = bounds
    
    # Create a grid of points
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
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surface = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha)
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Cost')
    
    # Define colors for different defense mechanisms
    defense_colors = {
        'sight': 'blue',
        'sound': 'green',
        'odor': 'orange',
        'physical': 'red'
    }
    
    # Plot the porcupines
    if defense_types is not None:
        for i, (pos, fit, defense) in enumerate(zip(positions, fitness, defense_types)):
            color = defense_colors.get(defense, 'white')
            ax.scatter(pos[0], pos[1], fit, c=color, edgecolors='black', s=80)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], fitness, c='white', edgecolors='black', s=80)
    
    # Plot the best position if provided
    if best_pos is not None:
        best_fitness = func(best_pos)
        ax.scatter(best_pos[0], best_pos[1], best_fitness, c='red', s=150, marker='*', 
                  label='Best Position')
    
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Cost')
    
    # Add legend for defense mechanisms if used
    if defense_types is not None:
        handles = []
        labels = []
        for defense, color in defense_colors.items():
            if defense in defense_types:
                handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                        markersize=10, label=defense.capitalize()))
                labels.append(defense.capitalize())
        
        if best_pos is not None:
            handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                                     markersize=15, label='Best Position'))
            labels.append('Best Position')
        
        ax.legend(handles=handles, labels=labels)
    else:
        ax.legend()
    
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


# Helper function to track defense mechanism usage
def track_defense_mechanisms(
    positions: np.ndarray,
    prev_positions: np.ndarray,
    best_pos: np.ndarray,
    tf: float = 0.8
) -> List[str]:
    """
    Determine which defense mechanism was likely used for each porcupine.
    
    Parameters
    ----------
    positions : ndarray
        Current positions of the porcupines, shape (pop_size, dimensions).
    prev_positions : ndarray
        Previous positions of the porcupines, shape (pop_size, dimensions).
    best_pos : ndarray
        Global best position, shape (dimensions,).
    tf : float, optional
        Tradeoff threshold between third and fourth mechanisms (default: 0.8).
    
    Returns
    -------
    list
        List of defense mechanisms used by each porcupine.
    """
    n_particles = positions.shape[0]
    defense_types = []
    
    for i in range(n_particles):
        # Calculate movement vector
        movement = positions[i] - prev_positions[i]
        
        # Calculate distance to best position
        dist_to_best = np.linalg.norm(positions[i] - best_pos)
        
        # Random threshold for exploration vs exploitation
        random_threshold = np.random.random()
        
        if random_threshold < 0.5:  # Exploration phase
            if np.random.random() < 0.5:
                defense_types.append('sight')
            else:
                defense_types.append('sound')
        else:  # Exploitation phase
            if np.random.random() < tf:
                defense_types.append('odor')
            else:
                defense_types.append('physical')
    
    return defense_types
