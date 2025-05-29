import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
import matplotlib.cm as cm


def plot_convergence(cost_history: List[float], title: str = "Convergence Curve", 
                    xlabel: str = "Iterations", ylabel: str = "Cost", 
                    figsize: Tuple[int, int] = (10, 6),
                    log_scale: bool = False,
                    save_path: Optional[str] = None):
    """
    Plot the convergence history of the optimization process.
    
    Parameters
    ----------
    cost_history : list
        List of cost values at each iteration.
    title : str, optional
        Title of the plot (default: "Convergence Curve").
    xlabel : str, optional
        Label for the x-axis (default: "Iterations").
    ylabel : str, optional
        Label for the y-axis (default: "Cost").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    log_scale : bool, optional
        Whether to use logarithmic scale for the y-axis (default: False).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    plt.figure(figsize=figsize)
    iterations = np.arange(1, len(cost_history) + 1)
    plt.plot(iterations, cost_history, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if log_scale and np.all(np.array(cost_history) > 0):
        plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_population_size(pop_size_history: List[int], title: str = "Population Size History", 
                         xlabel: str = "Iterations", ylabel: str = "Population Size", 
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None):
    """
    Plot the population size history of the optimization process.
    
    Parameters
    ----------
    pop_size_history : list
        List of population sizes at each iteration.
    title : str, optional
        Title of the plot (default: "Population Size History").
    xlabel : str, optional
        Label for the x-axis (default: "Iterations").
    ylabel : str, optional
        Label for the y-axis (default: "Population Size").
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
    iterations = np.arange(1, len(pop_size_history) + 1)
    plt.plot(iterations, pop_size_history, 'r-', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_2d_search_space(func: Callable, bounds: Tuple[np.ndarray, np.ndarray], 
                         resolution: int = 100, 
                         positions: Optional[np.ndarray] = None,
                         best_pos: Optional[np.ndarray] = None,
                         title: str = "2D Search Space", 
                         figsize: Tuple[int, int] = (10, 8),
                         cmap: str = 'viridis',
                         contour_levels: int = 20,
                         save_path: Optional[str] = None):
    """
    Plot a 2D search space with positions of the porcupines.
    
    Parameters
    ----------
    func : callable
        The objective function to visualize.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    resolution : int, optional
        Resolution of the grid for visualization (default: 100).
    positions : ndarray, optional
        Current positions of the porcupines, shape (pop_size, 2).
    best_pos : ndarray, optional
        Global best position, shape (2,).
    title : str, optional
        Title of the plot (default: "2D Search Space").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 8)).
    cmap : str, optional
        Colormap for the contour plot (default: 'viridis').
    contour_levels : int, optional
        Number of contour levels (default: 20).
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
    
    # Plot the positions if provided
    if positions is not None:
        plt.scatter(positions[:, 0], positions[:, 1], c='white', edgecolors='black', 
                   s=50, label='Porcupines')
    
    # Plot the best position if provided
    if best_pos is not None:
        plt.scatter(best_pos[0], best_pos[1], c='red', s=100, marker='*', 
                   label='Best Position')
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def animate_optimization_2d(position_history: List[np.ndarray], 
                           func: Callable, 
                           bounds: Tuple[np.ndarray, np.ndarray],
                           best_pos_history: Optional[List[np.ndarray]] = None,
                           interval: int = 200, 
                           figsize: Tuple[int, int] = (10, 8),
                           cmap: str = 'viridis',
                           contour_levels: int = 20,
                           save_path: Optional[str] = None,
                           dpi: int = 100):
    """
    Create an animation of the optimization process in 2D.
    
    Parameters
    ----------
    position_history : list
        List of position arrays at each iteration, each with shape (pop_size, 2).
    func : callable
        The objective function to visualize.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    best_pos_history : list, optional
        List of best positions at each iteration, each with shape (2,).
    interval : int, optional
        Interval between frames in milliseconds (default: 200).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 8)).
    cmap : str, optional
        Colormap for the contour plot (default: 'viridis').
    contour_levels : int, optional
        Number of contour levels (default: 20).
    save_path : str, optional
        Path to save the animation. If None, the animation is not saved (default: None).
    dpi : int, optional
        DPI for the saved animation (default: 100).
    
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
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the contour
    contour = ax.contourf(X, Y, Z, contour_levels, cmap=cmap, alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Cost')
    
    # Initialize scatter plots
    scatter_porcupines = ax.scatter([], [], c='white', edgecolors='black', s=50, label='Porcupines')
    scatter_best = ax.scatter([], [], c='red', s=100, marker='*', label='Best Position')
    
    # Set labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Optimization Process')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    # Set axis limits
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    
    # Text for iteration number
    iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=10)
    
    # Animation update function
    def update(frame):
        positions = position_history[frame]
        scatter_porcupines.set_offsets(positions)
        
        if best_pos_history is not None:
            best_pos = best_pos_history[frame]
            scatter_best.set_offsets([best_pos])
        
        iteration_text.set_text(f'Iteration: {frame+1}/{len(position_history)}')
        
        return scatter_porcupines, scatter_best, iteration_text
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(position_history), 
                        interval=interval, blit=True)
    
    # Save the animation if a path is provided
    if save_path:
        anim.save(save_path, dpi=dpi, writer='pillow')
    
    return anim


def plot_multiple_runs(cost_histories: List[List[float]], labels: List[str] = None,
                      title: str = "Comparison of Multiple Runs", 
                      xlabel: str = "Iterations", ylabel: str = "Cost", 
                      figsize: Tuple[int, int] = (10, 6),
                      log_scale: bool = False,
                      save_path: Optional[str] = None):
    """
    Plot the convergence histories of multiple optimization runs.
    
    Parameters
    ----------
    cost_histories : list
        List of cost history lists, each from a different run.
    labels : list, optional
        Labels for each run. If None, runs are labeled as "Run 1", "Run 2", etc.
    title : str, optional
        Title of the plot (default: "Comparison of Multiple Runs").
    xlabel : str, optional
        Label for the x-axis (default: "Iterations").
    ylabel : str, optional
        Label for the y-axis (default: "Cost").
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    log_scale : bool, optional
        Whether to use logarithmic scale for the y-axis (default: False).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    plt.figure(figsize=figsize)
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(cost_histories))]
    
    for i, history in enumerate(cost_histories):
        iterations = np.arange(1, len(history) + 1)
        plt.plot(iterations, history, linewidth=2, label=labels[i])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if log_scale and all(np.all(np.array(history) > 0) for history in cost_histories):
        plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_parameter_sensitivity(parameter_values: List[float], results: List[float],
                              parameter_name: str, result_name: str = "Best Cost",
                              title: str = None, figsize: Tuple[int, int] = (10, 6),
                              save_path: Optional[str] = None):
    """
    Plot the sensitivity of results to a parameter.
    
    Parameters
    ----------
    parameter_values : list
        List of parameter values tested.
    results : list
        List of corresponding results (e.g., best costs).
    parameter_name : str
        Name of the parameter.
    result_name : str, optional
        Name of the result metric (default: "Best Cost").
    title : str, optional
        Title of the plot. If None, a default title is generated.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 6)).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved (default: None).
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if title is None:
        title = f"Sensitivity of {result_name} to {parameter_name}"
    
    plt.figure(figsize=figsize)
    plt.plot(parameter_values, results, 'bo-', linewidth=2, markersize=8)
    plt.title(title)
    plt.xlabel(parameter_name)
    plt.ylabel(result_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data points
    for x, y in zip(parameter_values, results):
        plt.annotate(f"{y:.4g}", (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()
