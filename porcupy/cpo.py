import numpy as np

def cpo(fobj, lb, ub, pop_size=30, max_iter=100, f_ieqcons=None, verbose=False):
    """
    Crested Porcupine Optimizer (CPO) for optimization problems.

    This function implements the CPO algorithm, a nature-inspired metaheuristic that mimics
    the defensive behaviors of crested porcupines (sight, sound, odor, physical attack) to
    balance exploration and exploitation, with cyclic population reduction for convergence.

    Parameters
    ----------
    fobj : callable
        Objective function to minimize. Takes a 1D numpy array as input and returns a scalar.
    lb : list or array-like
        Lower bounds for each dimension of the search space.
    ub : list or array-like
        Upper bounds for each dimension of the search space.
    pop_size : int, optional
        Number of search agents (porcupines) in the initial population (default: 30).
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    f_ieqcons : callable, optional
        Constraint function returning a 1D array of inequality constraints (g(x) >= 0).
        Infeasible solutions are assigned infinite fitness (default: None).
    verbose : bool, optional
        If True, print progress information for each iteration (default: False).

    Returns
    -------
    best_pos : ndarray
        Best solution found (1D array of length `dim`).
    best_cost : float
        Best fitness value found.
    cost_history : ndarray
        Best fitness value recorded at each iteration (1D array of length `max_iter`).

    Raises
    ------
    ValueError
        If `lb` and `ub` have different lengths, `pop_size` or `max_iter` is non-positive,
        or `fobj` is not callable.

    Notes
    -----
    The CPO algorithm is based on the paper "Crested Porcupine Optimizer: A new
    nature-inspired metaheuristic" by Mohamed Abdel-Basset et al. It uses four defensive
    mechanisms and cyclic population reduction to optimize complex problems.
    """
    # Validate inputs
    if not callable(fobj):
        raise ValueError("Objective function 'fobj' must be callable")
    lb, ub = np.array(lb, dtype=float), np.array(ub, dtype=float)
    if len(lb) != len(ub):
        raise ValueError("Lower bounds 'lb' and upper bounds 'ub' must have the same length")
    if pop_size <= 0:
        raise ValueError("Population size 'pop_size' must be positive")
    if max_iter <= 0:
        raise ValueError("Maximum iterations 'max_iter' must be positive")
    if f_ieqcons is not None and not callable(f_ieqcons):
        raise ValueError("Constraint function 'f_ieqcons' must be callable")
    
    dim = len(lb)  # Dimension of the search space
    min_pop_size = max(10, pop_size // 2)  # Minimum population size, adjusted for small populations
    cycles = 2  # Number of cycles for population reduction (from MATLAB)
    alpha = 0.2  # Convergence rate for fourth defense mechanism (from MATLAB)
    tf = 0.8  # Tradeoff threshold between third and fourth mechanisms (from MATLAB)

    # Initialize population
    if len(lb) == 1:
        # Single bound for all dimensions
        positions = np.random.uniform(lb, ub, (pop_size, dim))
    else:
        # Different bounds for each dimension
        positions = np.zeros((pop_size, dim))
        for i in range(dim):
            positions[:, i] = np.random.uniform(lb[i], ub[i], pop_size)

    # Initialize fitness and best solution
    fitness = np.full(pop_size, np.inf)
    for i in range(pop_size):
        if f_ieqcons is None or np.all(np.array(f_ieqcons(positions[i])) >= 0):
            fitness[i] = fobj(positions[i])
    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_cost = fitness[best_idx]
    personal_best_pos = positions.copy()  # Personal best positions
    cost_history = np.zeros(max_iter)  # Convergence history

    # Optimization loop
    current_pop_size = pop_size
    for t in range(max_iter):
        # Update population size cyclically
        cycle_progress = (t % (max_iter // cycles)) / (max_iter // cycles)
        current_pop_size = max(1, int(min_pop_size + (pop_size - min_pop_size) * (1 - cycle_progress)))

        # Ensure population size does not exceed current array size
        current_pop_size = min(current_pop_size, positions.shape[0])

        for i in range(current_pop_size):
            # Random binary mask for position updates
            u1 = np.random.random(dim) > np.random.random()

            if np.random.random() < np.random.random():  # Exploration phase
                if np.random.random() < np.random.random():  # First defense mechanism (sight)
                    # Midpoint between current and random position
                    y = (positions[i] + positions[np.random.randint(current_pop_size)]) / 2
                    # Update with random perturbation toward global best
                    positions[i] += np.random.randn() * np.abs(2 * np.random.random() * best_pos - y)
                else:  # Second defense mechanism (sound)
                    # Midpoint with another random position
                    y = (positions[i] + positions[np.random.randint(current_pop_size)]) / 2
                    # Update using binary mask and random differences
                    rand_diff = positions[np.random.randint(current_pop_size)] - positions[np.random.randint(current_pop_size)]
                    positions[i] = u1 * positions[i] + (1 - u1) * (y + np.random.random() * rand_diff)
            else:  # Exploitation phase
                # Time-dependent factor for scaling updates
                yt = 2 * np.random.random() * (1 - t / max_iter) ** (t / max_iter)
                # Random direction vector
                u2 = (np.random.random(dim) < 0.5) * 2 - 1
                s = np.random.random() * u2

                if np.random.random() < tf:  # Third defense mechanism (odor)
                    # Exponential scaling based on relative fitness
                    fitness_sum = np.sum(fitness[:current_pop_size])
                    if np.isfinite(fitness[i]) and fitness_sum > 0:
                        st = np.exp(fitness[i] / (fitness_sum + np.finfo(float).eps))
                    else:
                        st = 1.0  # Default scaling if fitness is invalid
                    s = s * yt * st
                    # Update with random position and scaled difference
                    rand_diff = positions[np.random.randint(current_pop_size)] - positions[np.random.randint(current_pop_size)]
                    positions[i] = (1 - u1) * positions[i] + u1 * (positions[np.random.randint(current_pop_size)] + st * rand_diff - s)
                else:  # Fourth defense mechanism (physical attack)
                    # Exponential movement factor
                    fitness_sum = np.sum(fitness[:current_pop_size])
                    if np.isfinite(fitness[i]) and fitness_sum > 0:
                        mt = np.exp(fitness[i] / (fitness_sum + np.finfo(float).eps))
                    else:
                        mt = 1.0  # Default scaling if fitness is invalid
                    vt = positions[i]
                    vtp = positions[np.random.randint(current_pop_size)]
                    ft = np.random.random(dim) * (mt * (-vt + vtp))
                    s = s * yt * ft
                    # Update toward global best with convergence rate
                    r2 = np.random.random()
                    positions[i] = best_pos + (alpha * (1 - r2) + r2) * (u2 * best_pos - positions[i]) - s

            # Clip positions to bounds
            positions[i] = np.clip(positions[i], lb, ub)

            # Evaluate new fitness
            new_fitness = np.inf
            if f_ieqcons is None or np.all(np.array(f_ieqcons(positions[i])) >= 0):
                new_fitness = fobj(positions[i])

            # Update personal and global bests
            if new_fitness < fitness[i]:
                personal_best_pos[i] = positions[i].copy()
                fitness[i] = new_fitness
                if fitness[i] < best_cost:
                    best_pos = positions[i].copy()
                    best_cost = fitness[i]
            else:
                positions[i] = personal_best_pos[i].copy()

        # Trim population if size reduced
        if current_pop_size < positions.shape[0]:
            positions = positions[:current_pop_size]
            personal_best_pos = personal_best_pos[:current_pop_size]
            fitness = fitness[:current_pop_size]

        # Update convergence history
        cost_history[t] = best_cost

        # Print progress if verbose
        if verbose:
            print(f"Iteration {t+1}/{max_iter}: Best Cost = {best_cost:.6f}, Population Size = {current_pop_size}")

    return best_pos, best_cost, cost_history