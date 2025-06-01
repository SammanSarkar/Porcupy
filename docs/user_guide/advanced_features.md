# Advanced Features

This guide covers advanced features and customization options in Porcupy for solving complex optimization problems.

## Custom Defense Mechanisms

Porcupy allows you to implement custom defense mechanisms by subclassing the `DefenseMechanisms` class:

```python
from porcupy.porcupines import DefenseMechanisms
import numpy as np

class CustomDefense(DefenseMechanisms):
    def __init__(self, dimensions, bounds, **kwargs):
        super().__init__(dimensions, bounds, **kwargs)
        # Add custom initialization
        self.custom_param = kwargs.get('custom_param', 1.0)
    
    def apply_defense(self, positions, fitness, iteration):
        """Custom defense mechanism implementation."""
        # Example: Add random perturbation based on iteration
        perturbation = self.custom_param * np.random.normal(
            scale=1.0/(iteration+1),
            size=positions.shape
        )
        return positions + perturbation

# Usage
optimizer = CPO(
    dimensions=10,
    bounds=([-5]*10, [5]*10),
    defense_mechanism=CustomDefense,
    defense_params={'custom_param': 0.5}
)
```

## Hybrid Optimization

Combine CPO with local search or other optimization methods:

```python
from scipy.optimize import minimize

def hybrid_optimization(objective_func, dimensions, bounds, max_iter=100):
    # Global search with CPO
    optimizer = CPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=50,
        max_iter=max_iter//2
    )
    
    # Get best solution from CPO
    x0, best_fitness, _ = optimizer.optimize(objective_func)
    
    # Local refinement with L-BFGS-B
    result = minimize(
        objective_func,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter//2}
    )
    
    return result.x, result.fun
```

## Constraint Handling

### Penalty Function Approach

```python
def constrained_optimization():
    def objective(x):
        return x[0]**2 + x[1]**2  # Objective function
    
    def constraint(x):
        return x[0] + x[1] - 1  # x + y >= 1
    
    def penalty_function(x, penalty=1e6):
        violation = max(0, -constraint(x))
        return objective(x) + penalty * violation**2
    
    optimizer = CPO(dimensions=2, bounds=([0, 0], [5, 5]))
    best_solution, best_fitness, _ = optimizer.optimize(penalty_function)
    return best_solution, best_fitness
```

### Feasible Solutions Only

```python
def feasible_optimization():
    def is_feasible(x):
        return x[0] + x[1] >= 1  # Feasibility condition
    
    def constrained_objective(x):
        if not is_feasible(x):
            return float('inf')  # Reject infeasible solutions
        return x[0]**2 + x[1]**2
    
    optimizer = CPO(dimensions=2, bounds=([0, 0], [5, 5]))
    best_solution, best_fitness, _ = optimizer.optimize(constrained_objective)
    return best_solution, best_fitness
```

## Multi-objective Optimization

Implement a simple multi-objective optimization using a weighted sum approach:

```python
def multi_objective_optimization():
    def objective1(x):
        return np.sum(x**2)  # First objective: minimize sum of squares
    
    def objective2(x):
        return np.sum((x - 2)**2)  # Second objective: minimize distance from [2, 2, ...]
    
    def weighted_sum(x, weights=(0.5, 0.5)):
        return weights[0] * objective1(x) + weights[1] * objective2(x)
    
    # Get Pareto front by varying weights
    pareto_front = []
    for w in np.linspace(0, 1, 10):
        optimizer = CPO(dimensions=3, bounds=([-5]*3, [5]*3))
        x, f, _ = optimizer.optimize(lambda x: weighted_sum(x, [w, 1-w]))
        pareto_front.append((objective1(x), objective2(x)))
    
    return np.array(pareto_front)
```

## Dynamic Optimization

Optimize problems where the objective function changes over time:

```python
def dynamic_optimization():
    class DynamicObjective:
        def __init__(self):
            self.time = 0
            self.period = 100  # Change every 100 iterations
        
        def __call__(self, x):
            # Objective function that changes over time
            t = self.time // self.period
            return np.sum((x - t)**2)  # Target moves over time
        
        def update(self):
            self.time += 1
    
    # Initialize
    obj = DynamicObjective()
    optimizer = CPO(dimensions=2, bounds=([-10]*2, [10]*2), max_iter=1000)
    
    # Track best solutions over time
    solutions = []
    
    # Run optimization with dynamic updates
    for _ in range(10):  # 10 dynamic updates
        x, f, _ = optimizer.optimize(obj)
        solutions.append(x.copy())
        obj.update()  # Update the objective
        
        # Optional: Re-initialize population around current best
        optimizer.initialize_population_around(x, radius=1.0)
    
    return np.array(solutions)
```

## Custom Selection Strategies

Implement custom selection mechanisms:

```python
from porcupy.utils.population import SelectionStrategies

class TournamentSelection(SelectionStrategies):
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size
    
    def select(self, population, fitness, num_parents):
        selected = []
        for _ in range(num_parents):
            # Random tournament
            idx = np.random.choice(len(population), self.tournament_size, replace=False)
            # Select best in tournament
            best_idx = idx[np.argmin(fitness[idx])]
            selected.append(population[best_idx])
        return np.array(selected)

# Usage
optimizer = CPO(
    dimensions=10,
    bounds=([-5]*10, [5]*10),
    selection_strategy=TournamentSelection(tournament_size=5)
)
```

## Parallel Evaluation

Speed up evaluation of expensive objective functions using parallel processing:

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_optimization():
    def expensive_objective(x):
        # Simulate expensive computation
        import time
        time.sleep(0.01)
        return np.sum(x**2)
    
    # Initialize optimizer
    optimizer = CPO(dimensions=10, bounds=([-5]*10, [5]*10), pop_size=100)
    
    # Create process pool
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Run optimization with parallel evaluation
        best_solution, best_fitness, _ = optimizer.optimize(
            expensive_objective,
            executor=executor,
            chunksize=10  # Number of solutions to evaluate in each batch
        )
    
    return best_solution, best_fitness
```

## Custom Initialization

Implement custom population initialization strategies:

```python
def custom_initialization():
    def latin_hypercube_sampling(bounds, n_samples):
        """Generate samples using Latin Hypercube Sampling."""
        n_dims = len(bounds[0])
        samples = np.zeros((n_samples, n_dims))
        
        # Generate samples
        for i in range(n_dims):
            # Create intervals
            edges = np.linspace(bounds[0][i], bounds[1][i], n_samples + 1)
            # Randomly sample within each interval
            samples[:, i] = np.random.uniform(edges[:-1], edges[1:])
            # Shuffle the samples
            np.random.shuffle(samples[:, i])
            
        return samples
    
    # Generate initial population
    bounds = ([-5]*10, [5]*10)
    initial_population = latin_hypercube_sampling(bounds, n_samples=100)
    
    # Initialize optimizer with custom population
    optimizer = CPO(dimensions=10, bounds=bounds)
    optimizer.initialize_population(initial_population)
    
    # Run optimization
    return optimizer.optimize(lambda x: np.sum(x**2))
```

## Surrogate Models

Use surrogate models for expensive objective functions:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def surrogate_optimization():
    # Initialize surrogate model
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    surrogate = GaussianProcessRegressor(kernel=kernel)
    
    # Initial samples
    X_train = np.random.uniform(-5, 5, (20, 2))
    y_train = np.array([objective(x) for x in X_train])
    
    # Main optimization loop
    n_iterations = 10
    for i in range(n_iterations):
        # Update surrogate model
        surrogate.fit(X_train, y_train)
        
        # Optimize acquisition function (expected improvement)
        def acquisition(x):
            x = np.array(x).reshape(1, -1)
            mu, sigma = surrogate.predict(x, return_std=True)
            return -(mu - 0.1 * sigma)  # Exploration-exploitation trade-off
        
        optimizer = CPO(dimensions=2, bounds=([-5, -5], [5, 5]))
        x_next, _, _ = optimizer.optimize(acquisition)
        
        # Evaluate true objective
        y_next = objective(x_next)
        
        # Update training data
        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)
    
    # Return best solution found
    best_idx = np.argmin(y_train)
    return X_train[best_idx], y_train[best_idx]
```

## Custom Logging and Callbacks

Implement custom logging and callbacks during optimization:

```python
def optimization_with_callbacks():
    # Custom callback function
    def log_callback(iteration, fitness, positions, **kwargs):
        if iteration % 10 == 0:
            best_fitness = np.min(fitness)
            avg_fitness = np.mean(fitness)
            print(f"Iteration {iteration}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
    
    # Custom logger class
    class OptimizationLogger:
        def __init__(self):
            self.history = []
        
        def __call__(self, iteration, fitness, positions, **kwargs):
            entry = {
                'iteration': iteration,
                'best_fitness': float(np.min(fitness)),
                'avg_fitness': float(np.mean(fitness)),
                'std_fitness': float(np.std(fitness)),
                'best_solution': positions[np.argmin(fitness)].copy()
            }
            self.history.append(entry)
    
    # Initialize logger and optimizer
    logger = OptimizationLogger()
    optimizer = CPO(
        dimensions=5,
        bounds=([-5]*5, [5]*5),
        callbacks=[log_callback, logger],
        verbose=False
    )
    
    # Run optimization
    best_solution, best_fitness, _ = optimizer.optimize(lambda x: np.sum(x**2))
    
    return best_solution, best_fitness, logger.history
```

## Custom Termination Criteria

Implement custom termination criteria for the optimization process:

```python
def custom_termination():
    class CustomTermination:
        def __init__(self, patience=10, tol=1e-6):
            self.patience = patience
            self.tol = tol
            self.best_fitness = float('inf')
            self.no_improve = 0
        
        def __call__(self, iteration, fitness, **kwargs):
            current_best = np.min(fitness)
            
            if current_best < self.best_fitness - self.tol:
                self.best_fitness = current_best
                self.no_improve = 0
            else:
                self.no_improve += 1
            
            # Stop if no improvement for 'patience' iterations
            return self.no_improve >= self.patience
    
    # Initialize with custom termination
    termination = CustomTermination(patience=20, tol=1e-6)
    
    optimizer = CPO(
        dimensions=10,
        bounds=([-5]*10, [5]*10),
        termination_condition=termination
    )
    
    return optimizer.optimize(lambda x: np.sum(x**2))
```

These advanced features provide powerful tools for customizing and extending Porcupy to handle a wide range of optimization problems. The library's modular design makes it easy to implement custom components and integrate with other scientific Python libraries.
