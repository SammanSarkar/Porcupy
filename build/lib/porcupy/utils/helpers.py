import numpy as np

def initialize_population(pop_size, dim, lb, ub):
    """
    Initialize population within the specified bounds.

    Generates random positions for the porcupine population using uniform distribution.
    Supports both single bounds (same for all dimensions) and per-dimension bounds.

    Parameters
    ----------
    pop_size : int
        Number of search agents (porcupines).
    dim : int
        Number of dimensions in the search space.
    lb : ndarray
        Lower bounds for each dimension (or single value for all dimensions).
    ub : ndarray
        Upper bounds for each dimension (or single value for all dimensions).

    Returns
    -------
    positions : ndarray
        Initialized population of shape (pop_size, dim).

    Raises
    ------
    ValueError
        If `pop_size` or `dim` is non-positive, or `lb` and `ub` are incompatible.
    """
    if pop_size <= 0 or dim <= 0:
        raise ValueError("Population size and dimension must be positive")
    lb, ub = np.array(lb, dtype=float), np.array(ub, dtype=float)
    if len(lb) != len(ub) or (len(lb) > 1 and len(lb) != dim):
        raise ValueError("Bounds must match dimension or be single values")

    if len(lb) == 1:
        # Single bound for all dimensions
        return np.random.uniform(lb, ub, (pop_size, dim))
    else:
        # Per-dimension bounds
        positions = np.zeros((pop_size, dim))
        for i in range(dim):
            positions[:, i] = np.random.uniform(lb[i], ub[i], pop_size)
        return positions

def clip_to_bounds(positions, lb, ub):
    """
    Clip positions to stay within the specified bounds.

    Ensures all positions remain within the search space by clipping values that
    exceed the lower or upper bounds.

    Parameters
    ----------
    positions : ndarray
        Population positions of shape (pop_size, dim).
    lb : ndarray
        Lower bounds for each dimension (or single value).
    ub : ndarray
        Upper bounds for each dimension (or single value).

    Returns
    -------
    clipped_positions : ndarray
        Positions clipped to stay within bounds.
    """
    lb, ub = np.array(lb), np.array(ub)
    return np.clip(positions, lb, ub)