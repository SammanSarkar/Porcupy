"""
Visualization manager for the Crested Porcupine Optimizer (CPO).

This module provides a unified interface for all visualization tools in the Porcupy library.
"""

import numpy as np
import matplotlib.pyplot as plt
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
        self.diversity_history = []
    
    def record_iteration(
        self,
        iteration: int,
        positions: np.ndarray,
        best_position: np.ndarray,
        best_fitness: float,
        pop_size: int,
        defense_types: Optional[List[str]] = None
    ):
        """
        Record data from an iteration for later visualization.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        positions : ndarray
            Current positions of the porcupines, shape (pop_size, dimensions).
        best_position : ndarray
            Current global best position.
        best_fitness : float
            Current best fitness value.
        pop_size : int
            Current population size.
        defense_types : list, optional
            List of defense mechanisms used by each porcupine.
        """
        # Store position and fitness data
        self.position_history.append(positions.copy())
        self.best_position_history.append(best_position.copy())
        self.fitness_history.append(best_fitness)
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
        if not self.defense_history:
            raise ValueError("No defense mechanism data recorded")
        
        return plot_defense_mechanisms(
            defense_history=self.defense_history,
            title=title,
            figsize=figsize,
            save_path=save_path
        )
    
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
        if not self.defense_history or not self.fitness_history:
            raise ValueError("No defense mechanism or fitness data recorded")
        
        return plot_defense_effectiveness(
            defense_history=self.defense_history,
            fitness_history=self.fitness_history,
            title=title,
            figsize=figsize,
            save_path=save_path
        )
    
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
