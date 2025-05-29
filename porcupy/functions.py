import numpy as np
from typing import Callable, Dict, List, Union, Tuple

# ===== Unimodal Functions =====

def sphere(x):
    """
    Sphere function for optimization benchmarking.

    The sphere function is a simple, unimodal function defined as f(x) = sum(x_i^2).
    Its global minimum is 0 at x = [0, 0, ..., 0].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    return np.sum(np.array(x) ** 2)


def rosenbrock(x):
    """
    Rosenbrock function for optimization benchmarking.

    The Rosenbrock function is a non-convex function defined as
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2).
    Its global minimum is 0 at x = [1, 1, ..., 1].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


def schwefel_2_22(x):
    """
    Schwefel 2.22 function for optimization benchmarking.

    This is a unimodal function defined as f(x) = sum(|x_i|) + prod(|x_i|).
    Its global minimum is 0 at x = [0, 0, ..., 0].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def schwefel_1_2(x):
    """
    Schwefel 1.2 function for optimization benchmarking.

    This is a unimodal function defined as f(x) = sum(sum(x_j)^2) for j=1 to i, i=1 to n.
    Its global minimum is 0 at x = [0, 0, ..., 0].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    result = 0
    for i in range(len(x)):
        result += np.sum(x[:i+1])**2
    return result


def schwefel_2_21(x):
    """
    Schwefel 2.21 function for optimization benchmarking.

    This is a unimodal function defined as f(x) = max(|x_i|).
    Its global minimum is 0 at x = [0, 0, ..., 0].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    return np.max(np.abs(x))


def step(x):
    """
    Step function for optimization benchmarking.

    This is a discontinuous function defined as f(x) = sum(floor(x_i + 0.5)^2).
    Its global minimum is 0 at x in [-0.5, 0.5]^n.

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    return np.sum(np.floor(x + 0.5)**2)


def quartic(x):
    """
    Quartic function with noise for optimization benchmarking.

    This is a function defined as f(x) = sum(i * x_i^4) + random[0, 1).
    Its global minimum is close to 0 at x = [0, 0, ..., 0].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    return np.sum(np.arange(1, len(x) + 1) * x**4) + np.random.random()


# ===== Multimodal Functions =====

def rastrigin(x):
    """
    Rastrigin function for optimization benchmarking.

    The Rastrigin function is a multimodal function defined as
    f(x) = 10 * dim + sum(x_i^2 - 10 * cos(2 * pi * x_i)).
    Its global minimum is 0 at x = [0, 0, ..., 0].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def ackley(x):
    """
    Ackley function for optimization benchmarking.

    The Ackley function is a multimodal function defined as
    f(x) = -20 * exp(-0.2 * sqrt(sum(x_i^2) / n)) - exp(sum(cos(2 * pi * x_i)) / n) + 20 + e.
    Its global minimum is 0 at x = [0, 0, ..., 0].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def griewank(x):
    """
    Griewank function for optimization benchmarking.

    The Griewank function is a multimodal function defined as
    f(x) = 1 + sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i))).
    Its global minimum is 0 at x = [0, 0, ..., 0].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    i = np.arange(1, len(x) + 1)
    return 1 + np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(i)))


def schwefel(x):
    """
    Schwefel function for optimization benchmarking.

    The Schwefel function is a multimodal function defined as
    f(x) = 418.9829 * n - sum(x_i * sin(sqrt(|x_i|))).
    Its global minimum is 0 at x = [420.9687, ..., 420.9687].

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def michalewicz(x):
    """
    Michalewicz function for optimization benchmarking.

    The Michalewicz function is a multimodal function with steep ridges and valleys.
    It has n! local minima, and the global minimum value depends on the dimension.

    Parameters
    ----------
    x : ndarray
        Input vector (1D array) of shape (dim,).

    Returns
    -------
    float
        The computed function value.
    """
    x = np.array(x)
    i = np.arange(1, len(x) + 1)
    m = 10  # Steepness parameter
    return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2 * m))


# ===== Function Groups =====

def get_function_by_name(name: str) -> Callable:
    """
    Get a benchmark function by name.

    Parameters
    ----------
    name : str
        Name of the function.

    Returns
    -------
    callable
        The benchmark function.

    Raises
    ------
    ValueError
        If the function name is not recognized.
    """
    function_map = {
        # Unimodal functions
        'sphere': sphere,
        'rosenbrock': rosenbrock,
        'schwefel_2_22': schwefel_2_22,
        'schwefel_1_2': schwefel_1_2,
        'schwefel_2_21': schwefel_2_21,
        'step': step,
        'quartic': quartic,
        
        # Multimodal functions
        'rastrigin': rastrigin,
        'ackley': ackley,
        'griewank': griewank,
        'schwefel': schwefel,
        'michalewicz': michalewicz
    }
    
    if name not in function_map:
        raise ValueError(f"Unknown function name: {name}. Available functions: {list(function_map.keys())}")
    
    return function_map[name]


def get_function_bounds(name: str, dimensions: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the recommended bounds for a benchmark function.

    Parameters
    ----------
    name : str
        Name of the function.
    dimensions : int
        Number of dimensions.

    Returns
    -------
    tuple
        A tuple (lb, ub) containing the lower and upper bounds.

    Raises
    ------
    ValueError
        If the function name is not recognized.
    """
    bounds_map = {
        # Unimodal functions
        'sphere': (-100, 100),
        'rosenbrock': (-30, 30),
        'schwefel_2_22': (-10, 10),
        'schwefel_1_2': (-100, 100),
        'schwefel_2_21': (-100, 100),
        'step': (-100, 100),
        'quartic': (-1.28, 1.28),
        
        # Multimodal functions
        'rastrigin': (-5.12, 5.12),
        'ackley': (-32, 32),
        'griewank': (-600, 600),
        'schwefel': (-500, 500),
        'michalewicz': (0, np.pi)
    }
    
    if name not in bounds_map:
        raise ValueError(f"Unknown function name: {name}. Available functions: {list(bounds_map.keys())}")
    
    lb, ub = bounds_map[name]
    return np.full(dimensions, lb), np.full(dimensions, ub)


def get_function_optimum(name: str, dimensions: int) -> Tuple[np.ndarray, float]:
    """
    Get the global optimum for a benchmark function.

    Parameters
    ----------
    name : str
        Name of the function.
    dimensions : int
        Number of dimensions.

    Returns
    -------
    tuple
        A tuple (x_opt, f_opt) containing the optimal position and value.

    Raises
    ------
    ValueError
        If the function name is not recognized.
    """
    optima_map = {
        # Unimodal functions
        'sphere': (0, 0),
        'rosenbrock': (1, 0),
        'schwefel_2_22': (0, 0),
        'schwefel_1_2': (0, 0),
        'schwefel_2_21': (0, 0),
        'step': (0, 0),  # Approximate, as it's flat in [-0.5, 0.5]^n
        'quartic': (0, 0),  # Approximate due to noise
        
        # Multimodal functions
        'rastrigin': (0, 0),
        'ackley': (0, 0),
        'griewank': (0, 0),
        'schwefel': (420.9687, 0),
        'michalewicz': (None, None)  # Depends on dimension
    }
    
    if name not in optima_map:
        raise ValueError(f"Unknown function name: {name}. Available functions: {list(optima_map.keys())}")
    
    x_opt_val, f_opt = optima_map[name]
    
    if x_opt_val is None:  # Special case for Michalewicz
        return None, None
    
    x_opt = np.full(dimensions, x_opt_val)
    return x_opt, f_opt