"""
Interactive visualization module for the Crested Porcupine Optimizer (CPO).

This module provides interactive visualization tools and dashboard components
for monitoring and analyzing the CPO algorithm's performance in real-time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
import time
import threading
from IPython.display import display, clear_output


class OptimizationDashboard:
    """
    Interactive dashboard for monitoring CPO optimization in real-time.
    
    This dashboard displays multiple visualizations simultaneously, including:
    - Convergence history
    - Population size history
    - Diversity metrics
    - Defense mechanism usage
    - Current porcupine positions
    
    The dashboard updates in real-time during optimization.
    """
    
    def __init__(
        self,
        objective_func: Callable,
        bounds: Tuple[np.ndarray, np.ndarray],
        dimensions: int = 2,
        update_interval: float = 0.5,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Initialize the dashboard.
        
        Parameters
        ----------
        objective_func : callable
            The objective function being optimized.
        bounds : tuple
            A tuple (lb, ub) containing the lower and upper bounds.
        dimensions : int, optional
            Number of dimensions in the search space (default: 2).
        update_interval : float, optional
            Time interval between dashboard updates in seconds (default: 0.5).
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (15, 10)).
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.dimensions = dimensions
        self.update_interval = update_interval
        self.figsize = figsize
        
        # Data storage
        self.iterations = []
        self.best_costs = []
        self.pop_sizes = []
        self.diversity_history = []
        self.defense_counts = {'sight': [], 'sound': [], 'odor': [], 'physical': []}
        self.position_history = []
        self.best_position_history = []
        
        # Dashboard state
        self.is_running = False
        self.current_iteration = 0
        
        # Initialize the dashboard figure
        self._init_dashboard()
    
    def _init_dashboard(self):
        """Initialize the dashboard figure and subplots."""
        self.fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(3, 3, figure=self.fig)
        
        # Cost history plot
        self.ax_cost = self.fig.add_subplot(gs[0, 0:2])
        self.cost_line, = self.ax_cost.plot([], [], 'b-', linewidth=2)
        self.ax_cost.set_title('Convergence History')
        self.ax_cost.set_xlabel('Iterations')
        self.ax_cost.set_ylabel('Best Cost')
        self.ax_cost.grid(True, linestyle='--', alpha=0.7)
        
        # Population size plot
        self.ax_pop = self.fig.add_subplot(gs[0, 2])
        self.pop_line, = self.ax_pop.plot([], [], 'g-', linewidth=2)
        self.ax_pop.set_title('Population Size')
        self.ax_pop.set_xlabel('Iterations')
        self.ax_pop.set_ylabel('Population Size')
        self.ax_pop.grid(True, linestyle='--', alpha=0.7)
        
        # Diversity plot
        self.ax_div = self.fig.add_subplot(gs[1, 0])
        self.div_line, = self.ax_div.plot([], [], 'r-', linewidth=2)
        self.ax_div.set_title('Population Diversity')
        self.ax_div.set_xlabel('Iterations')
        self.ax_div.set_ylabel('Diversity')
        self.ax_div.grid(True, linestyle='--', alpha=0.7)
        
        # Defense mechanisms plot
        self.ax_def = self.fig.add_subplot(gs[1, 1])
        self.def_lines = {
            'sight': None,
            'sound': None,
            'odor': None,
            'physical': None
        }
        self.ax_def.set_title('Defense Mechanism Usage')
        self.ax_def.set_xlabel('Iterations')
        self.ax_def.set_ylabel('Count')
        self.ax_def.grid(True, linestyle='--', alpha=0.7)
        
        # Current positions plot (2D only)
        if self.dimensions == 2:
            self.ax_pos = self.fig.add_subplot(gs[1:3, 2])
            self.ax_pos.set_title('Current Positions')
            self.ax_pos.set_xlabel('x1')
            self.ax_pos.set_ylabel('x2')
            self.ax_pos.grid(True, linestyle='--', alpha=0.7)
            
            # Create contour plot for the objective function
            lb, ub = self.bounds
            resolution = 50
            x = np.linspace(lb[0], ub[0], resolution)
            y = np.linspace(lb[1], ub[1], resolution)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            for i in range(resolution):
                for j in range(resolution):
                    Z[i, j] = self.objective_func([X[i, j], Y[i, j]])
            
            self.contour = self.ax_pos.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.8)
            self.fig.colorbar(self.contour, ax=self.ax_pos, label='Cost')
            
            # Initialize scatter plot for porcupine positions
            self.scatter_porcupines = self.ax_pos.scatter([], [], c='white', edgecolors='black', s=80)
            self.scatter_best = self.ax_pos.scatter([], [], c='red', s=150, marker='*', label='Best Position')
            self.ax_pos.legend()
        
        # Status information
        self.ax_status = self.fig.add_subplot(gs[2, 0:2])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0.05, 0.5, '', fontsize=12, transform=self.ax_status.transAxes)
        
        # Adjust layout
        self.fig.tight_layout()
        plt.ion()  # Turn on interactive mode
    
    def update(
        self,
        iteration: int,
        best_cost: float,
        pop_size: int,
        positions: np.ndarray,
        best_position: np.ndarray,
        defense_types: List[str]
    ):
        """
        Update the dashboard with new optimization data.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        best_cost : float
            Current best cost value.
        pop_size : int
            Current population size.
        positions : ndarray
            Current positions of the porcupines, shape (pop_size, dimensions).
        best_position : ndarray
            Current global best position, shape (dimensions,).
        defense_types : list
            List of defense mechanisms used by each porcupine.
        """
        # Update data storage
        self.current_iteration = iteration
        self.iterations.append(iteration)
        self.best_costs.append(best_cost)
        self.pop_sizes.append(pop_size)
        
        # Calculate diversity
        diversity = self._calculate_diversity(positions)
        self.diversity_history.append(diversity)
        
        # Update defense mechanism counts
        for defense in self.defense_counts:
            self.defense_counts[defense].append(defense_types.count(defense))
        
        # Store position history
        self.position_history.append(positions.copy())
        self.best_position_history.append(best_position.copy())
        
        # Update plots
        self._update_plots()
    
    def _update_plots(self):
        """Update all dashboard plots with current data."""
        # Update cost history plot
        self.cost_line.set_data(self.iterations, self.best_costs)
        self.ax_cost.relim()
        self.ax_cost.autoscale_view()
        
        # Update population size plot
        self.pop_line.set_data(self.iterations, self.pop_sizes)
        self.ax_pop.relim()
        self.ax_pop.autoscale_view()
        
        # Update diversity plot
        self.div_line.set_data(self.iterations, self.diversity_history)
        self.ax_div.relim()
        self.ax_div.autoscale_view()
        
        # Update defense mechanisms plot
        colors = {'sight': 'blue', 'sound': 'green', 'odor': 'orange', 'physical': 'red'}
        for defense, color in colors.items():
            if self.def_lines[defense] is None:
                self.def_lines[defense], = self.ax_def.plot(
                    self.iterations, self.defense_counts[defense],
                    color=color, linewidth=2, label=defense.capitalize()
                )
            else:
                self.def_lines[defense].set_data(self.iterations, self.defense_counts[defense])
        
        self.ax_def.relim()
        self.ax_def.autoscale_view()
        self.ax_def.legend()
        
        # Update positions plot (2D only)
        if self.dimensions == 2 and len(self.position_history) > 0:
            current_positions = self.position_history[-1]
            self.scatter_porcupines.set_offsets(current_positions)
            
            # Color porcupines by defense mechanism
            if len(self.defense_counts) > 0:
                defense_types = []
                for defense in ['sight', 'sound', 'odor', 'physical']:
                    count = self.defense_counts[defense][-1]
                    defense_types.extend([defense] * count)
                
                colors = [colors[defense] for defense in defense_types]
                self.scatter_porcupines.set_color(colors)
            
            # Update best position
            if len(self.best_position_history) > 0:
                self.scatter_best.set_offsets([self.best_position_history[-1]])
        
        # Update status text
        status = f"Iteration: {self.current_iteration}\n"
        status += f"Best Cost: {self.best_costs[-1]:.6f}\n"
        status += f"Population Size: {self.pop_sizes[-1]}\n"
        status += f"Diversity: {self.diversity_history[-1]:.6f}\n"
        
        # Add defense mechanism counts
        status += "Defense Mechanisms:\n"
        for defense in self.defense_counts:
            if self.defense_counts[defense]:
                status += f"  {defense.capitalize()}: {self.defense_counts[defense][-1]}\n"
        
        self.status_text.set_text(status)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def _calculate_diversity(self, positions: np.ndarray) -> float:
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
    
    def start_monitoring(self):
        """Start the dashboard monitoring thread."""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the dashboard monitoring thread."""
        self.is_running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Main monitoring loop that updates the dashboard periodically."""
        while self.is_running:
            # Update the dashboard
            if len(self.iterations) > 0:
                self._update_plots()
            
            # Sleep for the update interval
            time.sleep(self.update_interval)
    
    def save_dashboard(self, save_path: str, dpi: int = 300):
        """
        Save the current dashboard state as an image.
        
        Parameters
        ----------
        save_path : str
            Path to save the dashboard image.
        dpi : int, optional
            DPI for the saved image (default: 300).
        """
        self.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    def close(self):
        """Close the dashboard and clean up resources."""
        self.stop_monitoring()
        plt.close(self.fig)


class ParameterTuningDashboard:
    """
    Interactive dashboard for parameter tuning and sensitivity analysis.
    
    This dashboard allows for real-time visualization of how different
    parameter values affect the performance of the CPO algorithm.
    """
    
    def __init__(
        self,
        parameter_name: str,
        parameter_range: List[float],
        result_metric: str = "Best Cost",
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Initialize the parameter tuning dashboard.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter being tuned.
        parameter_range : list
            List of parameter values to test.
        result_metric : str, optional
            Name of the result metric (default: "Best Cost").
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (12, 8)).
        """
        self.parameter_name = parameter_name
        self.parameter_range = parameter_range
        self.result_metric = result_metric
        self.figsize = figsize
        
        # Data storage
        self.results = []
        
        # Initialize the dashboard
        self._init_dashboard()
    
    def _init_dashboard(self):
        """Initialize the dashboard figure and subplots."""
        self.fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(2, 2, figure=self.fig)
        
        # Parameter sensitivity plot
        self.ax_sens = self.fig.add_subplot(gs[0, 0:2])
        self.sens_line, = self.ax_sens.plot([], [], 'b-o', linewidth=2)
        self.ax_sens.set_title(f'{self.result_metric} vs {self.parameter_name}')
        self.ax_sens.set_xlabel(self.parameter_name)
        self.ax_sens.set_ylabel(self.result_metric)
        self.ax_sens.grid(True, linestyle='--', alpha=0.7)
        
        # Convergence comparison plot
        self.ax_conv = self.fig.add_subplot(gs[1, 0])
        self.ax_conv.set_title('Convergence Comparison')
        self.ax_conv.set_xlabel('Iterations')
        self.ax_conv.set_ylabel(self.result_metric)
        self.ax_conv.grid(True, linestyle='--', alpha=0.7)
        
        # Statistics plot
        self.ax_stats = self.fig.add_subplot(gs[1, 1])
        self.ax_stats.set_title('Performance Statistics')
        self.ax_stats.axis('off')
        
        # Adjust layout
        self.fig.tight_layout()
        plt.ion()  # Turn on interactive mode
    
    def update(
        self,
        parameter_value: float,
        result: float,
        convergence_history: Optional[List[float]] = None
    ):
        """
        Update the dashboard with new parameter tuning results.
        
        Parameters
        ----------
        parameter_value : float
            Value of the parameter being tested.
        result : float
            Result metric value (e.g., best cost).
        convergence_history : list, optional
            Convergence history for this parameter value.
        """
        # Store the result
        self.results.append((parameter_value, result, convergence_history))
        
        # Sort results by parameter value
        self.results.sort(key=lambda x: x[0])
        
        # Update parameter sensitivity plot
        param_values = [r[0] for r in self.results]
        result_values = [r[1] for r in self.results]
        
        self.sens_line.set_data(param_values, result_values)
        self.ax_sens.relim()
        self.ax_sens.autoscale_view()
        
        # Update convergence comparison plot
        self.ax_conv.clear()
        self.ax_conv.set_title('Convergence Comparison')
        self.ax_conv.set_xlabel('Iterations')
        self.ax_conv.set_ylabel(self.result_metric)
        self.ax_conv.grid(True, linestyle='--', alpha=0.7)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.results)))
        
        for i, (param, _, conv_hist) in enumerate(self.results):
            if conv_hist is not None:
                iterations = np.arange(1, len(conv_hist) + 1)
                self.ax_conv.plot(iterations, conv_hist, color=colors[i], 
                                 linewidth=2, label=f'{self.parameter_name}={param}')
        
        self.ax_conv.legend()
        
        # Update statistics plot
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        stats_text = "Performance Statistics:\n\n"
        
        if len(self.results) > 0:
            # Find best parameter value
            best_idx = np.argmin(result_values)
            best_param = param_values[best_idx]
            best_result = result_values[best_idx]
            
            stats_text += f"Best {self.parameter_name}: {best_param}\n"
            stats_text += f"Best {self.result_metric}: {best_result:.6f}\n\n"
            
            # Calculate statistics
            mean_result = np.mean(result_values)
            std_result = np.std(result_values)
            
            stats_text += f"Mean {self.result_metric}: {mean_result:.6f}\n"
            stats_text += f"Std Dev: {std_result:.6f}\n\n"
            
            # Parameter range info
            stats_text += f"Parameter Range: [{min(param_values)}, {max(param_values)}]\n"
            stats_text += f"Result Range: [{min(result_values):.6f}, {max(result_values):.6f}]\n"
        
        self.ax_stats.text(0.05, 0.95, stats_text, fontsize=10, 
                          verticalalignment='top', transform=self.ax_stats.transAxes)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def save_dashboard(self, save_path: str, dpi: int = 300):
        """
        Save the current dashboard state as an image.
        
        Parameters
        ----------
        save_path : str
            Path to save the dashboard image.
        dpi : int, optional
            DPI for the saved image (default: 300).
        """
        self.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    def close(self):
        """Close the dashboard and clean up resources."""
        plt.close(self.fig)


def create_interactive_optimization_plot(
    objective_func: Callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    initial_positions: np.ndarray,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Create an interactive plot for exploring the optimization landscape.
    
    Parameters
    ----------
    objective_func : callable
        The objective function to visualize.
    bounds : tuple
        A tuple (lb, ub) containing the lower and upper bounds.
    initial_positions : ndarray
        Initial positions of the porcupines, shape (pop_size, 2).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 8)).
    
    Returns
    -------
    tuple
        A tuple containing (fig, ax, scatter) for further customization.
    """
    if initial_positions.shape[1] != 2:
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
            Z[i, j] = objective_func([X[i, j], Y[i, j]])
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the contour
    contour = ax.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Cost')
    
    # Plot the initial positions
    scatter = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], 
                        c='white', edgecolors='black', s=80)
    
    # Set labels and title
    ax.set_title('Interactive Optimization Landscape')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add interactivity for exploring the landscape
    def on_click(event):
        if event.inaxes == ax:
            # Evaluate the function at the clicked point
            cost = objective_func([event.xdata, event.ydata])
            
            # Display the cost
            ax.set_title(f'Cost at ({event.xdata:.4f}, {event.ydata:.4f}): {cost:.6f}')
            
            # Mark the clicked point
            ax.plot(event.xdata, event.ydata, 'ro', markersize=10)
            
            # Redraw the figure
            fig.canvas.draw_idle()
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    return fig, ax, scatter
