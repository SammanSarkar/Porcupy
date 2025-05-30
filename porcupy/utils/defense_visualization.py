"""
Defense mechanism visualization module for the Crested Porcupine Optimizer (CPO).

This module provides specialized visualization tools for the unique defense mechanisms
of the CPO algorithm, including sight, sound, odor, and physical defense visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Arrow
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from typing import List, Tuple, Optional, Union, Dict, Any, Callable


def visualize_defense_territories(
    positions: np.ndarray,
    defense_types: List[str],
    bounds: Tuple[np.ndarray, np.ndarray],
    territory_sizes: Optional[np.ndarray] = None,
    title: str = "Defense Territories",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Visualize the defense territories of porcupines based on their defense mechanisms.
    
    Parameters
    ----------
    positions : ndarray
        Positions of the porcupines, shape (pop_size, 2).
    defense_types : list
        List of defense mechanisms used by each porcupine.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    territory_sizes : ndarray, optional
        Sizes of the territories for each porcupine, shape (pop_size,).
        If None, territories are sized based on defense mechanism.
    title : str, optional
        Title of the plot (default: "Defense Territories").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 8)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if positions.shape[1] != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors and sizes for different defense mechanisms
    defense_colors = {
        'sight': 'blue',
        'sound': 'green',
        'odor': 'orange',
        'physical': 'red'
    }
    
    defense_alphas = {
        'sight': 0.2,
        'sound': 0.3,
        'odor': 0.4,
        'physical': 0.5
    }
    
    if territory_sizes is None:
        defense_sizes = {
            'sight': 1.5,    # Sight has the largest territory
            'sound': 1.2,    # Sound has a medium-large territory
            'odor': 0.8,     # Odor has a medium territory
            'physical': 0.5  # Physical has the smallest territory
        }
        territory_sizes = np.array([defense_sizes[d] for d in defense_types])
    
    # Create territory patches
    patches = []
    for i, (pos, defense, size) in enumerate(zip(positions, defense_types, territory_sizes)):
        color = defense_colors.get(defense, 'gray')
        alpha = defense_alphas.get(defense, 0.3)
        
        # Create a circle patch for the territory
        circle = Circle(pos, size, alpha=alpha, edgecolor=color, facecolor=color, linewidth=1.5)
        patches.append(circle)
        
        # Add a marker for the porcupine itself
        ax.scatter(pos[0], pos[1], c=color, edgecolors='black', s=80, zorder=10)
    
    # Add the territory patches to the plot
    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
    
    # Set the plot limits based on the bounds
    lb, ub = bounds
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    
    # Add a legend
    legend_elements = []
    for defense, color in defense_colors.items():
        if defense in defense_types:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                            markersize=10, label=defense.capitalize()))
    
    ax.legend(handles=legend_elements)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_defense_mechanisms(
    positions: np.ndarray,
    prev_positions: np.ndarray,
    defense_types: List[str],
    bounds: Tuple[np.ndarray, np.ndarray],
    best_position: Optional[np.ndarray] = None,
    title: str = "Defense Mechanisms Visualization",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """
    Visualize the specific defense mechanisms of porcupines with directional indicators.
    
    Parameters
    ----------
    positions : ndarray
        Current positions of the porcupines, shape (pop_size, 2).
    prev_positions : ndarray
        Previous positions of the porcupines, shape (pop_size, 2).
    defense_types : list
        List of defense mechanisms used by each porcupine.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    best_position : ndarray, optional
        Global best position, shape (2,).
    title : str, optional
        Title of the plot (default: "Defense Mechanisms Visualization").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 10)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if positions.shape[1] != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for different defense mechanisms
    defense_colors = {
        'sight': 'blue',
        'sound': 'green',
        'odor': 'orange',
        'physical': 'red'
    }
    
    # Set the plot limits based on the bounds
    lb, ub = bounds
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    
    # Plot the best position if provided
    if best_position is not None:
        ax.scatter(best_position[0], best_position[1], c='red', s=150, marker='*', 
                  label='Best Position', zorder=15)
    
    # Visualize each porcupine and its defense mechanism
    for i, (pos, prev_pos, defense) in enumerate(zip(positions, prev_positions, defense_types)):
        color = defense_colors.get(defense, 'gray')
        
        # Plot the porcupine
        ax.scatter(pos[0], pos[1], c=color, edgecolors='black', s=80, zorder=10)
        
        # Plot the movement vector
        movement = pos - prev_pos
        if np.linalg.norm(movement) > 0:
            ax.arrow(prev_pos[0], prev_pos[1], movement[0], movement[1], 
                    head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.7)
        
        # Visualize the specific defense mechanism
        if defense == 'sight':
            # Sight: Draw a cone of vision
            angle = np.arctan2(movement[1], movement[0]) if np.linalg.norm(movement) > 0 else 0
            wedge = Wedge(pos, 1.0, angle * 180 / np.pi - 30, angle * 180 / np.pi + 30, 
                         alpha=0.2, color=color, width=0.8)
            ax.add_patch(wedge)
            
        elif defense == 'sound':
            # Sound: Draw concentric circles representing sound waves
            for radius in [0.3, 0.6, 0.9]:
                circle = Circle(pos, radius, fill=False, alpha=0.5, color=color, linewidth=1.5)
                ax.add_patch(circle)
            
        elif defense == 'odor':
            # Odor: Draw a diffuse cloud around the porcupine
            circle = Circle(pos, 0.7, alpha=0.3, color=color, linewidth=0)
            ax.add_patch(circle)
            
        elif defense == 'physical':
            # Physical: Draw quills (spikes) around the porcupine
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                dx = 0.3 * np.cos(angle)
                dy = 0.3 * np.sin(angle)
                ax.arrow(pos[0], pos[1], dx, dy, head_width=0.08, head_length=0.08, 
                        fc=color, ec=color, alpha=0.9)
    
    # Add a legend
    legend_elements = []
    for defense, color in defense_colors.items():
        if defense in defense_types:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                            markersize=10, label=defense.capitalize()))
    
    if best_position is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                                        markersize=15, label='Best Position'))
    
    ax.legend(handles=legend_elements)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def animate_defense_mechanisms(
    position_history: List[np.ndarray],
    defense_history: List[List[str]],
    bounds: Tuple[np.ndarray, np.ndarray],
    best_position_history: Optional[List[np.ndarray]] = None,
    interval: int = 200,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    dpi: int = 100
):
    """
    Create an animation of porcupines using different defense mechanisms over time.
    
    Parameters
    ----------
    position_history : list
        List of position arrays at each iteration, each with shape (pop_size, 2).
    defense_history : list
        List of lists containing defense mechanisms used by each porcupine at each iteration.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    best_position_history : list, optional
        List of best positions at each iteration, each with shape (2,).
    interval : int, optional
        Interval between frames in milliseconds (default: 200).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 10)).
    save_path : str, optional
        Path to save the animation. If None, the animation is not saved (default: None).
    dpi : int, optional
        DPI for the saved animation (default: 100).
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The created animation.
    """
    if not position_history:
        raise ValueError("Position history is empty")
    
    if position_history[0].shape[1] != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for different defense mechanisms
    defense_colors = {
        'sight': 'blue',
        'sound': 'green',
        'odor': 'orange',
        'physical': 'red'
    }
    
    # Set the plot limits based on the bounds
    lb, ub = bounds
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    
    # Initialize scatter plot for porcupines
    scatter_porcupines = ax.scatter([], [], c=[], edgecolors='black', s=80, zorder=10)
    
    # Initialize scatter plot for best position
    scatter_best = ax.scatter([], [], c='red', s=150, marker='*', zorder=15)
    
    # Initialize patches for defense mechanisms
    sight_patches = []
    sound_patches = []
    odor_patches = []
    physical_patches = []
    
    # Create initial patches for each porcupine
    n_porcupines = position_history[0].shape[0]
    
    for i in range(n_porcupines):
        # Sight: Wedge for cone of vision
        sight_patch = Wedge((0, 0), 1.0, 0, 60, alpha=0.2, color='blue', width=0.8)
        sight_patches.append(sight_patch)
        ax.add_patch(sight_patch)
        
        # Sound: Concentric circles
        sound_circles = []
        for radius in [0.3, 0.6, 0.9]:
            circle = Circle((0, 0), radius, fill=False, alpha=0.5, color='green', linewidth=1.5)
            sound_circles.append(circle)
            ax.add_patch(circle)
        sound_patches.append(sound_circles)
        
        # Odor: Diffuse cloud
        odor_patch = Circle((0, 0), 0.7, alpha=0.3, color='orange', linewidth=0)
        odor_patches.append(odor_patch)
        ax.add_patch(odor_patch)
        
        # Physical: Quills (will be drawn as arrows in the update function)
        physical_arrows = []
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            arrow = ax.arrow(0, 0, 0, 0, head_width=0.08, head_length=0.08, 
                           fc='red', ec='red', alpha=0.9)
            physical_arrows.append(arrow)
        physical_patches.append(physical_arrows)
    
    # Set labels and title
    ax.set_title('Defense Mechanisms Animation')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add a legend
    legend_elements = []
    for defense, color in defense_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                        markersize=10, label=defense.capitalize()))
    
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                                    markersize=15, label='Best Position'))
    
    ax.legend(handles=legend_elements)
    
    # Text for iteration number
    iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=10)
    
    # Animation update function
    def update(frame):
        positions = position_history[frame]
        defenses = defense_history[frame]
        
        # Update porcupine positions
        scatter_porcupines.set_offsets(positions)
        
        # Update porcupine colors based on defense mechanisms
        colors = [defense_colors.get(defense, 'gray') for defense in defenses]
        scatter_porcupines.set_color(colors)
        
        # Update best position
        if best_position_history is not None:
            best_pos = best_position_history[frame]
            scatter_best.set_offsets([best_pos])
        
        # Calculate movement vectors if not the first frame
        if frame > 0:
            prev_positions = position_history[frame - 1]
        else:
            prev_positions = positions  # No movement for the first frame
        
        # Update defense mechanism visualizations
        for i, (pos, defense) in enumerate(zip(positions, defenses)):
            # Hide all defense mechanisms first
            sight_patches[i].set_visible(False)
            for circle in sound_patches[i]:
                circle.set_visible(False)
            odor_patches[i].set_visible(False)
            for arrow in physical_patches[i]:
                arrow.set_visible(False)
            
            # Show only the active defense mechanism
            if defense == 'sight':
                movement = pos - prev_positions[i]
                angle = np.arctan2(movement[1], movement[0]) if np.linalg.norm(movement) > 0 else 0
                sight_patches[i].set_center(pos)
                sight_patches[i].set_theta1(angle * 180 / np.pi - 30)
                sight_patches[i].set_theta2(angle * 180 / np.pi + 30)
                sight_patches[i].set_visible(True)
                
            elif defense == 'sound':
                for circle in sound_patches[i]:
                    circle.set_center(pos)
                    circle.set_visible(True)
                
            elif defense == 'odor':
                odor_patches[i].set_center(pos)
                odor_patches[i].set_visible(True)
                
            elif defense == 'physical':
                for j, angle in enumerate(np.linspace(0, 2*np.pi, 8, endpoint=False)):
                    dx = 0.3 * np.cos(angle)
                    dy = 0.3 * np.sin(angle)
                    # Remove old arrow and create a new one
                    physical_patches[i][j].remove()
                    physical_patches[i][j] = ax.arrow(pos[0], pos[1], dx, dy, 
                                                    head_width=0.08, head_length=0.08, 
                                                    fc='red', ec='red', alpha=0.9)
        
        # Update iteration text
        iteration_text.set_text(f'Iteration: {frame+1}/{len(position_history)}')
        
        # Return all artists that need to be redrawn
        return [scatter_porcupines, scatter_best, iteration_text]
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(position_history), 
                                 interval=interval, blit=True)
    
    # Save the animation if a path is provided
    if save_path:
        anim.save(save_path, dpi=dpi, writer='pillow')
    
    return anim


def plot_defense_effectiveness(
    defense_history: Dict[str, List[int]],
    fitness_history: List[float],
    title: str = "Defense Mechanism Effectiveness",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot the effectiveness of each defense mechanism in relation to fitness improvement.
    
    Parameters
    ----------
    defense_history : dict
        Dictionary with keys as defense mechanisms and values as lists of activation counts.
    fitness_history : list
        List of best fitness values at each iteration.
    title : str, optional
        Title of the plot (default: "Defense Mechanism Effectiveness").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 8)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    # Create the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                  gridspec_kw={'height_ratios': [2, 1]})
    
    # Define colors for different defense mechanisms
    defense_colors = {
        'sight': 'blue',
        'sound': 'green',
        'odor': 'orange',
        'physical': 'red'
    }
    
    # Plot defense mechanism activation counts
    iterations = np.arange(1, len(list(defense_history.values())[0]) + 1)
    
    for defense, counts in defense_history.items():
        ax1.plot(iterations, counts, color=defense_colors.get(defense, 'gray'), 
                linewidth=2, label=defense.capitalize())
    
    ax1.set_title(title)
    ax1.set_ylabel('Activation Count')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot fitness history
    ax2.plot(iterations, fitness_history, 'k-', linewidth=2)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Best Fitness')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate correlation between defense mechanism usage and fitness improvement
    correlations = {}
    for defense, counts in defense_history.items():
        # Calculate fitness improvements
        fitness_improvements = np.zeros_like(fitness_history)
        for i in range(1, len(fitness_history)):
            fitness_improvements[i] = fitness_history[i-1] - fitness_history[i]
        
        # Calculate correlation
        correlation = np.corrcoef(counts, fitness_improvements)[0, 1]
        correlations[defense] = correlation
    
    # Add correlation information as text
    correlation_text = "Correlation with Fitness Improvement:\n"
    for defense, corr in correlations.items():
        correlation_text += f"{defense.capitalize()}: {corr:.4f}\n"
    
    ax1.text(0.02, 0.02, correlation_text, transform=ax1.transAxes, 
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_quill_directions(
    positions: np.ndarray,
    velocities: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    title: str = "Porcupine Quill Directions",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Visualize porcupines with quills pointing in the direction of their movement.
    
    Parameters
    ----------
    positions : ndarray
        Positions of the porcupines, shape (pop_size, 2).
    velocities : ndarray
        Velocities of the porcupines, shape (pop_size, 2).
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    title : str, optional
        Title of the plot (default: "Porcupine Quill Directions").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 8)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if positions.shape[1] != 2:
        raise ValueError("This function only works for 2D search spaces")
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set the plot limits based on the bounds
    lb, ub = bounds
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    
    # Plot each porcupine with directional quills
    for i, (pos, vel) in enumerate(zip(positions, velocities)):
        # Plot the porcupine
        ax.scatter(pos[0], pos[1], c='brown', edgecolors='black', s=100, zorder=10)
        
        # Normalize velocity for quill direction
        vel_norm = vel / (np.linalg.norm(vel) + 1e-10)  # Avoid division by zero
        
        # Draw quills in the direction of movement and surrounding directions
        main_angle = np.arctan2(vel_norm[1], vel_norm[0])
        
        # Draw main quill (longer) in the direction of movement
        main_length = 0.4
        ax.arrow(pos[0], pos[1], main_length * vel_norm[0], main_length * vel_norm[1],
                head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=11)
        
        # Draw surrounding quills (shorter)
        for angle_offset in np.linspace(-np.pi/2, np.pi/2, 5):
            if angle_offset != 0:  # Skip the main direction
                angle = main_angle + angle_offset
                dx = 0.2 * np.cos(angle)
                dy = 0.2 * np.sin(angle)
                ax.arrow(pos[0], pos[1], dx, dy, head_width=0.05, head_length=0.05,
                        fc='black', ec='black', alpha=0.7, zorder=11)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
