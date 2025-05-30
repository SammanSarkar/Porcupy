"""
Visualization manager for the Crested Porcupine Optimizer (CPO).

This module provides a unified interface for all visualization tools in the Porcupy library.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Union, Dict, Any, Callable

# Import visualization modules
from porcupy.utils.enhanced_visualization import (
    plot_defense_mechanisms,
    plot_population_cycles,
    plot_diversity_history,
    plot_2d_porcupines,
    animate_porcupines_2d,
    plot_3d_porcupines,
    calculate_diversity,
    track_defense_mechanisms
)

from porcupy.utils.interactive_visualization import (
    OptimizationDashboard,
    ParameterTuningDashboard,
    create_interactive_optimization_plot
)

from porcupy.utils.defense_visualization import (
    visualize_defense_territories,
    visualize_defense_mechanisms,
    animate_defense_mechanisms,
    plot_defense_effectiveness,
    visualize_quill_directions
)

from porcupy.utils.population_visualization import (
    plot_population_reduction_strategies,
    plot_population_diversity_map,
    animate_population_cycle,
    plot_exploration_exploitation_balance,
    plot_diversity_vs_convergence
)


class CPOVisualizer:
    """
    Unified interface for all CPO visualization tools.
    
    This class provides access to all visualization capabilities for the
    Crested Porcupine Optimizer algorithm.
    """
    
    def __init__(
        self,
        objective_func: Optional[Callable] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """
        Initialize the CPO visualizer.
        
        Parameters
        ----------
        objective_func : callable, optional
            The objective function being optimized.
        bounds : tuple, optional
            A tuple (lb, ub) containing the lower and upper bounds.
        """
        self.objective_func = objective_func
        self.bounds = bounds
        
        # Data storage
        self.position_history = []
        self.best_position_history = []
        self.fitness_history = []
        self.pop_size_history = []
        self.defense_history = {}
        self.defense_types_history = []
        self.diversity_history = []
    
    def record_iteration(
        self,
        positions: np.ndarray,
        best_position: np.ndarray,
        fitness: np.ndarray,
        pop_size: int,
        defense_types: Optional[List[str]] = None
    ):
        """
        Record data from a single iteration for visualization.
        
        Parameters
        ----------
        positions : numpy.ndarray
            Positions of all porcupines in the current iteration.
        best_position : numpy.ndarray
            Best position found so far.
        fitness : numpy.ndarray
            Fitness values of all porcupines in the current iteration.
        pop_size : int
            Current population size.
        defense_types : list of str, optional
            Types of defense mechanisms used by each porcupine in the current iteration.
        """
        # Store position and fitness data
        self.position_history.append(positions.copy())
        self.best_position_history.append(best_position.copy())
        self.fitness_history.append(fitness.copy())
        
        # Store defense types if provided
        if defense_types is not None:
            # Initialize defense_types_history if it doesn't exist
            if not hasattr(self, 'defense_types_history'):
                self.defense_types_history = []
            self.defense_types_history.append(defense_types)
            
            # Also store in defense_history for backward compatibility
            if 'defense_types' not in self.defense_history:
                self.defense_history['defense_types'] = []
            self.defense_history['defense_types'].append(defense_types)
        
        self.pop_size_history.append(pop_size)
        
        # Calculate and store diversity
        diversity = calculate_diversity(positions)
        self.diversity_history.append(diversity)
        
        # Store defense mechanism data
        if defense_types is not None:
            for defense in ['sight', 'sound', 'odor', 'physical']:
                if defense not in self.defense_history:
                    self.defense_history[defense] = []
                self.defense_history[defense].append(defense_types.count(defense))
    
    def create_dashboard(
        self,
        update_interval: float = 0.5,
        figsize: Tuple[int, int] = (15, 10)
    ) -> OptimizationDashboard:
        """
        Create an interactive dashboard for monitoring optimization.
        
        Parameters
        ----------
        update_interval : float, optional
            Time interval between dashboard updates in seconds (default: 0.5).
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (15, 10)).
        
        Returns
        -------
        OptimizationDashboard
            The created dashboard.
        """
        if self.objective_func is None or self.bounds is None:
            raise ValueError("objective_func and bounds must be provided to create a dashboard")
        
        dimensions = self.position_history[0].shape[1] if self.position_history else 2
        
        dashboard = OptimizationDashboard(
            objective_func=self.objective_func,
            bounds=self.bounds,
            dimensions=dimensions,
            update_interval=update_interval,
            figsize=figsize
        )
        
        return dashboard
    
    def visualize_defense_mechanisms(
        self,
        title: str = "Defense Mechanism Activation",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Visualize the activation of different defense mechanisms over iterations.
        
        Parameters
        ----------
        title : str, optional
            Title of the plot (default: "Defense Mechanism Activation").
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (12, 6)).
        save_path : str, optional
            Path to save the figure. If None, the figure is not saved (default: None).
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if not self.position_history:
            raise ValueError("No optimization data recorded")
            
        # Check if we have defense mechanism data
        # First check if we have defense_types_history directly (from CPO class)
        if hasattr(self, 'defense_types_history') and self.defense_types_history:
            defense_data = self.defense_types_history
            has_direct_data = True
        # Then check if we have it in the defense_history dictionary
        elif self.defense_history and 'defense_types' in self.defense_history and self.defense_history['defense_types']:
            defense_data = self.defense_history['defense_types']
            has_direct_data = True
        else:
            has_direct_data = False
            raise ValueError("No defense mechanism data recorded. Make sure to record defense types during optimization.")
            
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Defense Mechanism Usage Over Time
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Count defense mechanisms at each iteration
        iterations = len(defense_data)
        sight_counts = []
        sound_counts = []
        odor_counts = []
        physical_counts = []
        
        for i in range(iterations):
            defenses = defense_data[i]
            sight_counts.append(defenses.count('sight'))
            sound_counts.append(defenses.count('sound'))
            odor_counts.append(defenses.count('odor'))
            physical_counts.append(defenses.count('physical'))
        
        # Plot usage over time
        x = range(iterations)
        ax1.plot(x, sight_counts, 'b-', label='Sight', linewidth=2)
        ax1.plot(x, sound_counts, 'g-', label='Sound', linewidth=2)
        ax1.plot(x, odor_counts, 'orange', label='Odor', linewidth=2)
        ax1.plot(x, physical_counts, 'r-', label='Physical', linewidth=2)
        
        ax1.set_title('Defense Mechanism Usage')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Exploration vs Exploitation Balance
        ax2 = fig.add_subplot(gs[0, 1])
        
        exploration = [sight + sound for sight, sound in zip(sight_counts, sound_counts)]
        exploitation = [odor + physical for odor, physical in zip(odor_counts, physical_counts)]
        
        ax2.stackplot(x, exploration, exploitation, 
                     labels=['Exploration (Sight/Sound)', 'Exploitation (Odor/Physical)'],
                     colors=['#3498db', '#e74c3c'], alpha=0.7)
        
        ax2.set_title('Exploration-Exploitation Balance')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Fitness Improvement by Defense Mechanism
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate fitness improvement for each defense mechanism
        defense_improvement = {
            'sight': [],
            'sound': [],
            'odor': [],
            'physical': []
        }
        
        # Use best fitness from each iteration
        fitness_values = self.fitness_history
        
        # Skip first iteration as we can't calculate improvement
        for i in range(1, iterations):
            prev_fitness = fitness_values[i-1] if i > 0 else fitness_values[0]
            current_fitness = fitness_values[i]
            # Extract scalar values for comparison
            if isinstance(prev_fitness, np.ndarray) and isinstance(current_fitness, np.ndarray):
                # Take the minimum (best) fitness from each array
                prev_best = np.min(prev_fitness)
                current_best = np.min(current_fitness)
                improvement = prev_best - current_best
            else:
                # Already scalar values
                improvement = prev_fitness - current_fitness
            # Count improvements by defense mechanism
            defenses = defense_data[i-1]
            for defense in ['sight', 'sound', 'odor', 'physical']:
                count = defenses.count(defense)
                if count > 0:
                    defense_improvement[defense].append(improvement / count if improvement > 0 else 0)
                else:
                    defense_improvement[defense].append(0)
        
        # Calculate cumulative improvement
        for defense in defense_improvement:
            defense_improvement[defense] = np.cumsum(defense_improvement[defense])
        
        # Plot cumulative improvement
        ax3.plot(range(1, iterations), defense_improvement['sight'], 'b-', label='Sight', linewidth=2)
        ax3.plot(range(1, iterations), defense_improvement['sound'], 'g-', label='Sound', linewidth=2)
        ax3.plot(range(1, iterations), defense_improvement['odor'], 'orange', label='Odor', linewidth=2)
        ax3.plot(range(1, iterations), defense_improvement['physical'], 'r-', label='Physical', linewidth=2)
        
        ax3.set_title('Cumulative Fitness Improvement')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Cumulative Improvement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Defense Mechanism Effectiveness Pie Chart
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate total improvement by each defense mechanism
        total_improvement = {}
        for defense in defense_improvement:
            if len(defense_improvement[defense]) > 0:
                total_improvement[defense] = defense_improvement[defense][-1]
            else:
                total_improvement[defense] = 0
        
        # Create pie chart
        labels = ['Sight', 'Sound', 'Odor', 'Physical']
        sizes = [total_improvement['sight'], total_improvement['sound'], 
                total_improvement['odor'], total_improvement['physical']]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        # Ensure we don't have negative values for the pie chart
        sizes = [max(0, size) for size in sizes]
        
        # If all sizes are 0, set equal values
        if sum(sizes) == 0:
            sizes = [1, 1, 1, 1]
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               shadow=True, startangle=90)
        ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax4.set_title('Overall Defense Mechanism Effectiveness')
        
        # Set the main title
        fig.suptitle(title, fontsize=16)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
        
    def visualize_population_cycles(
        self,
        cycles: int,
        max_iter: int,
        title: str = "Population Size Cycles",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Visualize the population size changes over cycles.
        
        Parameters
        ----------
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
        if not self.pop_size_history:
            raise ValueError("No population size data recorded")
        
        return plot_population_cycles(
            pop_size_history=self.pop_size_history,
            cycles=cycles,
            max_iter=max_iter,
            title=title,
            figsize=figsize,
            save_path=save_path
        )
    
    def visualize_diversity_history(
        self,
        title: str = "Population Diversity History",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Visualize the diversity history of the population.
        
        Parameters
        ----------
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
        if not self.diversity_history:
            raise ValueError("No diversity data recorded")
        
        return plot_diversity_history(
            diversity_history=self.diversity_history,
            title=title,
            figsize=figsize,
            save_path=save_path
        )
    
    def visualize_porcupines_2d(
        self,
        iteration: int = -1,
        title: str = "Porcupine Positions",
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
        contour_levels: int = 20,
        quill_length: float = 0.5,
        save_path: Optional[str] = None
    ):
        """
        Visualize porcupines in 2D search space at a specific iteration.
        
        Parameters
        ----------
        iteration : int, optional
            Iteration to visualize. Default is -1 (last iteration).
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
        if not self.position_history:
            raise ValueError("No position data recorded")
        
        if self.objective_func is None or self.bounds is None:
            raise ValueError("objective_func and bounds must be provided for this visualization")
        
        # Get positions for the specified iteration
        positions = self.position_history[iteration]
        
        # Get defense types if available
        defense_types = None
        if self.defense_history:
            defense_types = []
            for defense in ['sight', 'sound', 'odor', 'physical']:
                count = self.defense_history[defense][iteration]
                defense_types.extend([defense] * count)
        
        # Get best position if available
        best_pos = self.best_position_history[iteration] if self.best_position_history else None
        
        return plot_2d_porcupines(
            positions=positions,
            func=self.objective_func,
            bounds=self.bounds,
            best_pos=best_pos,
            defense_types=defense_types,
            title=title,
            figsize=figsize,
            cmap=cmap,
            contour_levels=contour_levels,
            quill_length=quill_length,
            save_path=save_path
        )
    
    def animate_optimization(
        self,
        interval: int = 200,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
        contour_levels: int = 20,
        quill_length: float = 0.5,
        save_path: Optional[str] = None,
        dpi: int = 100
    ):
        """
        Create an animation of the optimization process in 2D.
        
        Parameters
        ----------
        interval : int, optional
            Interval between frames in milliseconds (default: 200).
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (10, 8)).
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
        
        Returns
        -------
        matplotlib.animation.FuncAnimation
            The created animation.
        """
        if not self.position_history:
            raise ValueError("No position data recorded")
        
        if self.objective_func is None or self.bounds is None:
            raise ValueError("objective_func and bounds must be provided for this animation")
        
        # Create defense history in the required format
        defense_history = None
        if self.defense_history:
            defense_history = []
            for i in range(len(self.position_history)):
                defenses = []
                for defense in ['sight', 'sound', 'odor', 'physical']:
                    count = self.defense_history[defense][i]
                    defenses.extend([defense] * count)
                defense_history.append(defenses)
        
        return animate_porcupines_2d(
            position_history=self.position_history,
            func=self.objective_func,
            bounds=self.bounds,
            defense_history=defense_history,
            best_pos_history=self.best_position_history,
            interval=interval,
            figsize=figsize,
            cmap=cmap,
            contour_levels=contour_levels,
            quill_length=quill_length,
            save_path=save_path,
            dpi=dpi
        )
    
    def create_animation(
        self,
        positions_history: List[np.ndarray],
        best_position_history: List[np.ndarray],
        title: str = "CPO Optimization Process",
        save_path: Optional[str] = None,
        fps: int = 5,
        defense_types_history: Optional[List[List[str]]] = None,
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 100,
        show_exploration_exploitation: bool = True
    ):
        """
        Create an enhanced animation of the optimization process with defense mechanisms.
        
        Parameters
        ----------
        positions_history : List[np.ndarray]
            List of position arrays for each iteration.
        best_position_history : List[np.ndarray]
            List of best positions for each iteration.
        title : str, optional
            Title of the animation (default: "CPO Optimization Process").
        save_path : str, optional
            Path to save the animation. If None, the animation is not saved (default: None).
        fps : int, optional
            Frames per second for the animation (default: 5).
        defense_types_history : List[List[str]], optional
            List of defense types used by each porcupine at each iteration.
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (12, 10)).
        dpi : int, optional
            DPI for the saved animation (default: 100).
        show_exploration_exploitation : bool, optional
            Whether to show exploration-exploitation balance subplot (default: True).
            
        Returns
        -------
        matplotlib.animation.FuncAnimation
            The created animation.
        """
        if self.objective_func is None or self.bounds is None:
            raise ValueError("objective_func and bounds must be provided for this visualization")
        
        # Create figure with subplots
        if show_exploration_exploitation:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.3)
            ax_main = fig.add_subplot(gs[0, :])
            ax_exp_vs_expl = fig.add_subplot(gs[1, 0])
            ax_defense_pie = fig.add_subplot(gs[1, 1])
        else:
            fig, ax_main = plt.subplots(figsize=figsize)
        
        # Generate meshgrid for contour plot
        lb, ub = self.bounds
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.objective_func(np.array([X[i, j], Y[i, j]]))
        
        # Create contour plot on main axis
        contour = ax_main.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.6)
        fig.colorbar(contour, ax=ax_main, label='Objective Function Value')
        
        # Initialize scatter plots for different defense mechanisms
        sight_scatter = ax_main.scatter([], [], c='blue', s=80, label='Sight', alpha=0.7)
        sound_scatter = ax_main.scatter([], [], c='green', s=80, label='Sound', alpha=0.7)
        odor_scatter = ax_main.scatter([], [], c='orange', s=80, label='Odor', alpha=0.7)
        physical_scatter = ax_main.scatter([], [], c='red', s=80, label='Physical', alpha=0.7)
        best_scatter = ax_main.scatter([], [], c='yellow', s=150, marker='*', 
                                 edgecolors='black', label='Best Position', zorder=10)
        
        # Add trajectory line for best position
        trajectory_line, = ax_main.plot([], [], 'r-', linewidth=1.5, alpha=0.5, label='Best Trajectory')
        
        # Set title and labels for main plot
        ax_main.set_title(title)
        ax_main.set_xlabel('x1')
        ax_main.set_ylabel('x2')
        ax_main.legend(loc='upper right')
        ax_main.grid(True, alpha=0.3)
        
        # Set axis limits
        ax_main.set_xlim(lb[0], ub[0])
        ax_main.set_ylim(lb[1], ub[1])
        
        # Text for iteration counter and population size
        iteration_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes, 
                                 verticalalignment='top', fontsize=10)
        
        # Initialize exploration-exploitation subplot if needed
        if show_exploration_exploitation:
            # Set up exploration vs exploitation subplot
            ax_exp_vs_expl.set_title('Exploration-Exploitation Balance')
            ax_exp_vs_expl.set_xlabel('Iteration')
            ax_exp_vs_expl.set_ylabel('Count')
            ax_exp_vs_expl.grid(True, alpha=0.3)
            
            # Set up defense pie chart
            ax_defense_pie.set_title('Defense Mechanisms')
            ax_defense_pie.axis('equal')
            
            # Initialize exploration-exploitation plot data
            exp_line, = ax_exp_vs_expl.plot([], [], 'b-', label='Exploration (Sight/Sound)')
            expl_line, = ax_exp_vs_expl.plot([], [], 'r-', label='Exploitation (Odor/Physical)')
            ax_exp_vs_expl.legend(loc='upper left', fontsize=8)
            
            # Set initial limits for exploration-exploitation plot
            ax_exp_vs_expl.set_xlim(0, len(positions_history))
            ax_exp_vs_expl.set_ylim(0, max([len(pos) for pos in positions_history]) + 2)
        
        # Update function for animation
        def update(frame):
            # Clear previous points
            sight_scatter.set_offsets(np.empty((0, 2)))
            sound_scatter.set_offsets(np.empty((0, 2)))
            odor_scatter.set_offsets(np.empty((0, 2)))
            physical_scatter.set_offsets(np.empty((0, 2)))
            
            # Get positions for current frame
            positions = positions_history[frame]
            
            # Update best position and trajectory
            best_pos = best_position_history[frame]
            best_scatter.set_offsets([best_pos[0], best_pos[1]])
            
            # Update trajectory line
            if frame > 0:
                traj_x = [pos[0] for pos in best_position_history[:frame+1]]
                traj_y = [pos[1] for pos in best_position_history[:frame+1]]
                trajectory_line.set_data(traj_x, traj_y)
            
            # Update iteration text
            iteration_text.set_text(f'Iteration: {frame+1}/{len(positions_history)}\n'
                                   f'Population Size: {len(positions)}')
            
            # If defense types are provided, color points accordingly
            if defense_types_history is not None:
                defenses = defense_types_history[frame]
                
                sight_pos = []
                sound_pos = []
                odor_pos = []
                physical_pos = []
                
                # Count defenses for exploration-exploitation balance
                sight_count = 0
                sound_count = 0
                odor_count = 0
                physical_count = 0
                
                for i, defense in enumerate(defenses):
                    if i < len(positions):  # Ensure we don't go out of bounds
                        if defense == 'sight':
                            sight_pos.append(positions[i])
                            sight_count += 1
                        elif defense == 'sound':
                            sound_pos.append(positions[i])
                            sound_count += 1
                        elif defense == 'odor':
                            odor_pos.append(positions[i])
                            odor_count += 1
                        elif defense == 'physical':
                            physical_pos.append(positions[i])
                            physical_count += 1
                
                # Update exploration-exploitation plots if enabled
                if show_exploration_exploitation:
                    # Update exploration-exploitation balance plot
                    exploration = [0] * (frame + 1)
                    exploitation = [0] * (frame + 1)
                    
                    # Calculate exploration-exploitation for all frames up to current
                    for i in range(frame + 1):
                        curr_defenses = defense_types_history[i]
                        exp_count = sum(1 for d in curr_defenses if d in ['sight', 'sound'])
                        expl_count = sum(1 for d in curr_defenses if d in ['odor', 'physical'])
                        exploration[i] = exp_count
                        exploitation[i] = expl_count
                    
                    # Update lines
                    exp_line.set_data(range(frame + 1), exploration)
                    expl_line.set_data(range(frame + 1), exploitation)
                    
                    # Update pie chart
                    ax_defense_pie.clear()
                    ax_defense_pie.set_title('Defense Mechanisms')
                    labels = ['Sight', 'Sound', 'Odor', 'Physical']
                    sizes = [sight_count, sound_count, odor_count, physical_count]
                    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
                    
                    # Only create pie if we have non-zero values
                    if sum(sizes) > 0:
                        ax_defense_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                         shadow=True, startangle=90, textprops={'fontsize': 8})
                        ax_defense_pie.axis('equal')
                
                if sight_pos:
                    sight_scatter.set_offsets(sight_pos)
                if sound_pos:
                    sound_scatter.set_offsets(sound_pos)
                if odor_pos:
                    odor_scatter.set_offsets(odor_pos)
                if physical_pos:
                    physical_scatter.set_offsets(physical_pos)
            else:
                # If no defense types, show all points in one color
                sight_scatter.set_offsets(positions)
            
            return sight_scatter, sound_scatter, odor_scatter, physical_scatter, best_scatter, trajectory_line, iteration_text
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(positions_history), interval=1000/fps, blit=True)
        
        # Save animation if path is provided
        if save_path:
            print(f"Generating animation (this may take a moment)...")
            anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
        
        plt.close(fig)
        return anim
    
    def visualize_defense_territories(
        self,
        iteration: int = -1,
        title: str = "Defense Territories",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Visualize the defense territories of porcupines.
        
        Parameters
        ----------
        iteration : int, optional
            Iteration to visualize. Default is -1 (last iteration).
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
        if not self.position_history:
            raise ValueError("No position data recorded")
        
        if self.bounds is None:
            raise ValueError("bounds must be provided for this visualization")
        
        # Get positions for the specified iteration
        positions = self.position_history[iteration]
        
        # Get defense types if available
        defense_types = None
        if self.defense_history:
            defense_types = []
            for defense in ['sight', 'sound', 'odor', 'physical']:
                count = self.defense_history[defense][iteration]
                defense_types.extend([defense] * count)
        else:
            # If no defense data, create random defense types
            defense_options = ['sight', 'sound', 'odor', 'physical']
            defense_types = [defense_options[i % 4] for i in range(len(positions))]
        
        return visualize_defense_territories(
            positions=positions,
            defense_types=defense_types,
            bounds=self.bounds,
            title=title,
            figsize=figsize,
            save_path=save_path
        )
    
    def visualize_exploration_exploitation(
        self,
        sample_iterations: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Visualize the balance between exploration and exploitation.
        
        Parameters
        ----------
        sample_iterations : list, optional
            List of iteration indices to visualize. If None, evenly spaced iterations are selected.
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (15, 10)).
        save_path : str, optional
            Path to save the figure. If None, the figure is not saved (default: None).
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if not self.position_history or not self.best_position_history:
            raise ValueError("No position data recorded")
        
        if self.bounds is None:
            raise ValueError("bounds must be provided for this visualization")
        
        # If no sample iterations provided, select evenly spaced iterations
        if sample_iterations is None:
            n_samples = min(6, len(self.position_history))
            sample_iterations = np.linspace(0, len(self.position_history) - 1, n_samples, dtype=int).tolist()
        
        return plot_exploration_exploitation_balance(
            positions_history=self.position_history,
            best_positions_history=self.best_position_history,
            bounds=self.bounds,
            sample_iterations=sample_iterations,
            figsize=figsize,
            save_path=save_path
        )
    
    def visualize_diversity_vs_convergence(
        self,
        cycles: int,
        max_iter: int,
        title: str = "Diversity vs Convergence",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Visualize the relationship between population diversity and convergence.
        
        Parameters
        ----------
        cycles : int
            Number of cycles used in the optimization.
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
        if not self.diversity_history or not self.fitness_history:
            raise ValueError("No diversity or fitness data recorded")
        
        return plot_diversity_vs_convergence(
            diversity_history=self.diversity_history,
            fitness_history=self.fitness_history,
            cycles=cycles,
            max_iter=max_iter,
            title=title,
            figsize=figsize,
            save_path=save_path
        )
    
    def visualize_defense_effectiveness(
        self,
        title: str = "Defense Mechanism Effectiveness",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Visualize the effectiveness of each defense mechanism.
        
        Parameters
        ----------
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
        if not self.position_history or not self.fitness_history:
            raise ValueError("No optimization data recorded")
            
        # Check if we have defense mechanism data in the expected format
        if not self.defense_history:
            raise ValueError("No defense mechanism data recorded")
            
        # Ensure we have defense_types available
        if 'defense_types' not in self.defense_history:
            # If we have individual defense counts but not defense_types, we can't proceed
            # as we need the specific defense type for each porcupine
            raise ValueError("Defense mechanism types not recorded. This visualization requires defense_types data.")
            
        # Ensure defense_types has data
        if not self.defense_history['defense_types'] or len(self.defense_history['defense_types']) == 0:
            raise ValueError("Defense mechanism types list is empty")
            
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Defense Mechanism Usage Over Time
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Count defense mechanisms at each iteration
        iterations = len(self.defense_history['defense_types'])
        sight_counts = []
        sound_counts = []
        odor_counts = []
        physical_counts = []
        
        for i in range(iterations):
            defenses = self.defense_history['defense_types'][i]
            sight_counts.append(defenses.count('sight'))
            sound_counts.append(defenses.count('sound'))
            odor_counts.append(defenses.count('odor'))
            physical_counts.append(defenses.count('physical'))
        
        # Plot usage over time
        x = range(iterations)
        ax1.plot(x, sight_counts, 'b-', label='Sight', linewidth=2)
        ax1.plot(x, sound_counts, 'g-', label='Sound', linewidth=2)
        ax1.plot(x, odor_counts, 'orange', label='Odor', linewidth=2)
        ax1.plot(x, physical_counts, 'r-', label='Physical', linewidth=2)
        
        ax1.set_title('Defense Mechanism Usage')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Exploration vs Exploitation Balance
        ax2 = fig.add_subplot(gs[0, 1])
        
        exploration = [sight + sound for sight, sound in zip(sight_counts, sound_counts)]
        exploitation = [odor + physical for odor, physical in zip(odor_counts, physical_counts)]
        
        ax2.stackplot(x, exploration, exploitation, 
                     labels=['Exploration (Sight/Sound)', 'Exploitation (Odor/Physical)'],
                     colors=['#3498db', '#e74c3c'], alpha=0.7)
        
        ax2.set_title('Exploration-Exploitation Balance')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Fitness Improvement by Defense Mechanism
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate fitness improvement for each defense mechanism
        defense_improvement = {
            'sight': [],
            'sound': [],
            'odor': [],
            'physical': []
        }
        
        # Use best fitness from each iteration - extract scalar values
        fitness_values = []
        for fitness_array in self.fitness_history:
            # If fitness is an array, take the minimum value (best fitness)
            if isinstance(fitness_array, np.ndarray) and fitness_array.size > 0:
                fitness_values.append(float(np.min(fitness_array)))
            # If it's already a scalar, use it directly
            elif np.isscalar(fitness_array):
                fitness_values.append(float(fitness_array))
            # If it's a list, take the minimum value
            elif isinstance(fitness_array, list) and len(fitness_array) > 0:
                fitness_values.append(float(min(fitness_array)))
            # Default case
            else:
                fitness_values.append(0.0)

        # Skip first iteration as we can't calculate improvement
        if len(fitness_values) <= 1:
            # Not enough data for improvement calculation
            return fig
            
        for i in range(1, min(iterations, len(fitness_values))):
            prev_fitness = fitness_values[i-1] if i > 0 else fitness_values[0]
            current_fitness = fitness_values[i]
            # Extract scalar values for comparison
            if isinstance(prev_fitness, np.ndarray) and isinstance(current_fitness, np.ndarray):
                # Take the minimum (best) fitness from each array
                prev_best = np.min(prev_fitness)
                current_best = np.min(current_fitness)
                improvement = prev_best - current_best
            else:
                # Already scalar values
                improvement = prev_fitness - current_fitness
            
            # Count improvements by defense mechanism
            defenses = self.defense_history['defense_types'][i-1]
            for defense in ['sight', 'sound', 'odor', 'physical']:
                count = defenses.count(defense)
                if count > 0:
                    defense_improvement[defense].append(improvement / count if improvement > 0 else 0)
                else:
                    defense_improvement[defense].append(0)
        
        # Calculate cumulative improvement
        for defense in defense_improvement:
            defense_improvement[defense] = np.cumsum(defense_improvement[defense])
        
        # Plot cumulative improvement
        ax3.plot(range(1, iterations), defense_improvement['sight'], 'b-', label='Sight', linewidth=2)
        ax3.plot(range(1, iterations), defense_improvement['sound'], 'g-', label='Sound', linewidth=2)
        ax3.plot(range(1, iterations), defense_improvement['odor'], 'orange', label='Odor', linewidth=2)
        ax3.plot(range(1, iterations), defense_improvement['physical'], 'r-', label='Physical', linewidth=2)
        
        ax3.set_title('Cumulative Fitness Improvement')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Cumulative Improvement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Defense Mechanism Effectiveness Pie Chart
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate total improvement by each defense mechanism
        total_improvement = {}
        for defense in defense_improvement:
            # Check if the array is not empty
            if len(defense_improvement[defense]) > 0:
                # Get the last value from the array
                total_improvement[defense] = defense_improvement[defense][-1]
            else:
                total_improvement[defense] = 0
        
        # Create pie chart
        labels = ['Sight', 'Sound', 'Odor', 'Physical']
        sizes = [total_improvement['sight'], total_improvement['sound'], 
                total_improvement['odor'], total_improvement['physical']]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        # Ensure we don't have negative values for the pie chart
        sizes = [max(0, size) for size in sizes]
        
        # If all sizes are 0, set equal values
        if sum(sizes) == 0:
            sizes = [1, 1, 1, 1]
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
               shadow=True, startangle=90)
        ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax4.set_title('Overall Defense Mechanism Effectiveness')
        
        # Set the main title
        fig.suptitle(title, fontsize=16)
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
        
    def compare_reduction_strategies(
        self,
        max_iter: int,
        pop_size: int,
        cycles: int,
        strategies: List[str] = ['linear', 'cosine', 'exponential'],
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Compare different population reduction strategies.
        
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
        return plot_population_reduction_strategies(
            max_iter=max_iter,
            pop_size=pop_size,
            cycles=cycles,
            strategies=strategies,
            figsize=figsize,
            save_path=save_path
        )
    
    def record_from_optimizer(self, optimizer):
        """
        Record data from a CPO optimizer instance.
        
        Parameters
        ----------
        optimizer : CPO
            The CPO optimizer instance to record data from.
        """
        # Record position history
        if hasattr(optimizer, 'positions_history') and optimizer.positions_history:
            self.position_history = optimizer.positions_history
            
        # Record best position history
        if hasattr(optimizer, 'best_positions_history') and optimizer.best_positions_history:
            self.best_position_history = optimizer.best_positions_history
            
        # Record fitness history
        if hasattr(optimizer, 'fitness_history') and optimizer.fitness_history:
            self.fitness_history = optimizer.fitness_history
            
        # Record population size history
        if hasattr(optimizer, 'pop_size_history') and optimizer.pop_size_history:
            self.pop_size_history = optimizer.pop_size_history
            
        # Record defense mechanism history
        if hasattr(optimizer, 'defense_types_history') and optimizer.defense_types_history:
            # Store in both places for compatibility
            self.defense_types_history = optimizer.defense_types_history
            if not self.defense_history:
                self.defense_history = {}
            self.defense_history['defense_types'] = optimizer.defense_types_history
    
    def create_parameter_tuning_dashboard(
        self,
        parameter_name: str,
        parameter_range: List[float],
        result_metric: str = "Best Cost",
        figsize: Tuple[int, int] = (12, 8)
    ) -> ParameterTuningDashboard:
        """
        Create a dashboard for parameter tuning and sensitivity analysis.
        
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
        
        Returns
        -------
        ParameterTuningDashboard
            The created dashboard.
        """
        return ParameterTuningDashboard(
            parameter_name=parameter_name,
            parameter_range=parameter_range,
            result_metric=result_metric,
            figsize=figsize
        )
