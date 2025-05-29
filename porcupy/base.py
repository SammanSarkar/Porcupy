import abc
import numpy as np
from collections import namedtuple
from typing import Callable, Tuple, List, Optional, Dict, Any, Union


class Optimizer(abc.ABC):
    """
    Base class for optimization algorithms in Porcupy.
    
    This abstract base class provides the foundation for all optimization
    algorithms in the Porcupy library. It defines the common interface and
    functionality that all optimizers should implement.
    
    Parameters
    ----------
    dimensions : int
        Number of dimensions in the search space.
    bounds : tuple of numpy.ndarray
        A tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        (dimensions,).
    pop_size : int, optional
        Number of search agents in the population (default: 30).
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    options : dict, optional
        A dictionary containing algorithm-specific parameters.
    ftol : float, optional
        Relative error in objective_func(best_pos) acceptable for
        convergence (default: -np.inf).
    ftol_iter : int, optional
        Number of iterations over which the relative error in
        objective_func(best_pos) is acceptable for convergence (default: 1).
    """
    
    def __init__(
        self,
        dimensions: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        pop_size: int = 30,
        max_iter: int = 100,
        options: Optional[Dict[str, Any]] = None,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
    ):
        # Validate inputs
        if dimensions <= 0:
            raise ValueError("Dimensions must be positive")
        if pop_size <= 0:
            raise ValueError("Population size must be positive")
        if max_iter <= 0:
            raise ValueError("Maximum iterations must be positive")
        if ftol_iter <= 0 or not isinstance(ftol_iter, int):
            raise ValueError("ftol_iter must be a positive integer")
            
        # Initialize primary attributes
        self.dimensions = dimensions
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.options = options if options is not None else {}
        self.ftol = ftol
        self.ftol_iter = ftol_iter
        
        # Initialize history tracking
        self.ToHistory = namedtuple(
            "ToHistory",
            [
                "best_cost",
                "mean_cost",
                "position",
                "population_size"
            ],
        )
        
        # Initialize resettable attributes
        self.reset()
    
    def _populate_history(self, hist):
        """Populate history lists with current iteration data
        
        Parameters
        ----------
        hist : collections.namedtuple
            Must be of the same type as self.ToHistory
        """
        self.cost_history.append(hist.best_cost)
        self.mean_cost_history.append(hist.mean_cost)
        self.position_history.append(hist.position)
        self.pop_size_history.append(hist.population_size)
    
    def reset(self):
        """Reset the attributes of the optimizer
        
        This method reinitializes all history tracking attributes and
        prepares the optimizer for a new optimization run.
        """
        # Initialize history lists
        self.cost_history = []
        self.mean_cost_history = []
        self.position_history = []
        self.pop_size_history = []
        self.best_cost = np.inf
        self.best_pos = None
        
    @abc.abstractmethod
    def optimize(
        self, 
        objective_func: Callable, 
        n_processes: Optional[int] = None, 
        verbose: bool = False, 
        **kwargs
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Optimize the objective function
        
        Parameters
        ----------
        objective_func : callable
            The objective function to be minimized
        n_processes : int, optional
            Number of processes to use for parallel evaluation
        verbose : bool, optional
            Whether to display progress information
        **kwargs
            Additional arguments to pass to the objective function
            
        Returns
        -------
        tuple
            A tuple containing (best_position, best_cost, cost_history)
        """
        raise NotImplementedError("Optimizer::optimize()")
