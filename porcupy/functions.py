import numpy as np

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