"""
GPU-accelerated Crested Porcupine Optimizer (CPO).

This module provides a GPU-accelerated implementation of the CPO algorithm
using CuPy for numerical computations on NVIDIA GPUs. It's designed to
automatically fall back to CPU computation if CUDA is not available.

Example:
    >>> from porcupy.gpu_cpo import GPUCPO
    >>> from porcupy.functions import sphere
    >>>
    >>> # Initialize the optimizer
    >>> optimizer = GPUCPO(
    ...     dimensions=10,
    ...     bounds=([-5.12] * 10, [5.12] * 10),
    ...     pop_size=100,
    ...     max_iter=50
    ... )
    >>>
    >>> # Run optimization
    >>> best_pos, best_cost, _ = optimizer.optimize(sphere)
    >>> print(f"Best solution: {best_pos}")
    >>> print(f"Best cost: {best_cost}")

Note:
    For optimal performance, install CuPy with CUDA support:
    ```bash
    pip install cupy-cuda11x  # Choose the right CUDA version
    ```
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any
import warnings

# Try to import CuPy, fall back to NumPy if not available
try:
    import cupy as cp
    from cupy import ndarray as CudaArray
    CUDA_AVAILABLE = True
except ImportError:
    import numpy as cp
    from numpy import ndarray as CudaArray
    CUDA_AVAILABLE = False
    warnings.warn(
        "CuPy not found. Falling back to NumPy. "
        "Install CuPy for GPU acceleration: pip install cupy-cuda11x"
    )

from .cpo_class import CPO

class GPUCPO(CPO):
    """GPU-accelerated Crested Porcupine Optimizer.
    
    This class extends the standard CPO with GPU acceleration using CuPy.
    It's a drop-in replacement for CPO with the same interface but runs
    computations on GPU when available.
    
    Args:
        dimensions (int): Number of dimensions of the search space.
        bounds (tuple): Tuple of (lower_bounds, upper_bounds) for each dimension.
        pop_size (int): Initial population size.
        max_iter (int): Maximum number of iterations.
        min_pop_size (int, optional): Minimum population size. Defaults to 5.
        cycles (int, optional): Number of cycles for population reduction. Defaults to 5.
        alpha (float, optional): Reduction rate. Defaults to 0.95.
        tf (float, optional): Transfer factor. Defaults to 0.8.
        ftol (float, optional): Absolute error for convergence. Defaults to 1e-10.
        ftol_iter (int, optional): Number of iterations to check for convergence. Defaults to 10.
    
    Note:
        The optimizer will automatically detect and use GPU if CuPy with CUDA support
        is installed. Otherwise, it will fall back to CPU computation.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the GPU-accelerated CPO.
        
        Parameters are the same as the base CPO class.
        """
        super().__init__(*args, **kwargs)
        self._device = 'gpu' if CUDA_AVAILABLE else 'cpu'
        self._to_np = cp.asnumpy if CUDA_AVAILABLE else lambda x: x
        self._to_device = cp.asarray if CUDA_AVAILABLE else lambda x: x
        
    def _init_population(self):
        """Initialize the population on the GPU."""
        lb, ub = self.bounds
        
        # Create population on device
        self.positions = cp.random.uniform(
            low=lb[0], high=ub[0], 
            size=(self.pop_size, self.dimensions)
        )
        
        # Initialize fitness and personal best
        self.fitness = cp.full(self.pop_size, cp.inf)
        self.personal_best_pos = self.positions.copy()
        self.best_pos = None
        self.best_cost = cp.inf
    
    def _evaluate_population(self, objective_func: Callable, 
                           f_ieqcons: Optional[Callable] = None, 
                           **kwargs):
        """Evaluate the population on the GPU."""
        # Convert to numpy if objective function doesn't support CuPy arrays
        if not hasattr(objective_func, '__cuda_array_interface__'):
            positions_np = self._to_np(self.positions)
            
            # Evaluate all positions
            fitness = np.array([
                objective_func(pos, **kwargs) 
                if f_ieqcons is None or np.all(f_ieqcons(pos) >= 0)
                else np.inf
                for pos in positions_np
            ])
            
            # Convert back to device
            self.fitness = self._to_device(fitness)
        else:
            # GPU-accelerated objective function
            if f_ieqcons is None:
                self.fitness = cp.array([
                    objective_func(pos, **kwargs) 
                    for pos in self.positions
                ])
            else:
                self.fitness = cp.array([
                    objective_func(pos, **kwargs)
                    if cp.all(f_ieqcons(pos) >= 0)
                    else cp.inf
                    for pos in self.positions
                ])
        
        # Update personal best
        improved = self.fitness < self._to_device(
            [cp.inf if p is None else p for p in self.personal_best_fit]
        )
        if cp.any(improved):
            self.personal_best_pos[improved] = self.positions[improved]
            self.personal_best_fit[improved] = self.fitness[improved]
        
        # Update global best
        min_idx = cp.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_cost:
            self.best_cost = float(self.fitness[min_idx])
            self.best_pos = self._to_np(self.positions[min_idx])
    
    def optimize(self, objective_func: Callable, 
                f_ieqcons: Optional[Callable] = None,
                n_processes: Optional[int] = None,
                verbose: bool = False,
                track_history: bool = True,
                **kwargs) -> Tuple[np.ndarray, float, np.ndarray]:
        """Run the optimization on GPU."""
        # Convert bounds to device
        self.bounds = (
            self._to_device(self.bounds[0]),
            self._to_device(self.bounds[1])
        )
        
        # Run parent optimization
        result = super().optimize(
            objective_func=objective_func,
            f_ieqcons=f_ieqcons,
            n_processes=n_processes,
            verbose=verbose,
            track_history=track_history,
            **kwargs
        )
        
        # Ensure result is in CPU memory
        return (
            self._to_np(result[0]),
            float(result[1]),
            self._to_np(result[2])
        )

def gpu_cpo(fobj, lb, ub, pop_size=30, max_iter=100, **kwargs):
    """GPU-accelerated CPO function interface.
    
    Parameters are the same as the standard cpo() function.
    """
    optimizer = GPUCPO(
        dimensions=len(lb),
        bounds=(np.array(lb), np.array(ub)),
        pop_size=pop_size,
        max_iter=max_iter,
        **{k: v for k, v in kwargs.items() 
           if k in ['min_pop_size', 'cycles', 'alpha', 'tf', 'ftol', 'ftol_iter']}
    )
    
    # Extract constraint function if provided
    f_ieqcons = kwargs.get('f_ieqcons')
    
    return optimizer.optimize(
        objective_func=fobj,
        f_ieqcons=f_ieqcons,
        verbose=kwargs.get('verbose', False)
    )
