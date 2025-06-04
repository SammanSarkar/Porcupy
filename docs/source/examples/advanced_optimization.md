# Advanced Optimization Examples

This guide demonstrates advanced optimization techniques using Porcupy, including multi-objective optimization, dynamic optimization, and hybrid approaches.

## Table of Contents
- [Multi-Objective Optimization](#multi-objective-optimization)
- [Dynamic Optimization](#dynamic-optimization)
- [Hybrid Optimization](#hybrid-optimization)
- [Surrogate-Assisted Optimization](#surrogate-assisted-optimization)
- [Distributed Optimization](#distributed-optimization)
- [Custom Optimization Loop](#custom-optimization-loop)

## Multi-Objective Optimization

### Weighted Sum Approach

```python
import numpy as np
from porcupy import CPO

def multi_objective_optimization():
    # Define two competing objectives
    def objective1(x):
        return np.sum(x**2)  # Minimize distance from origin
    
    def objective2(x):
        return np.sum((x - 2)**2)  # Minimize distance from [2, 2, ...]
    
    # Generate Pareto front by varying weights
    n_weights = 10
    pareto_front = []
    
    for w in np.linspace(0, 1, n_weights):
        def weighted_objective(x):
            return w * objective1(x) + (1 - w) * objective2(x)
        
        optimizer = CPO(dimensions=3, bounds=([-5]*3, [5]*3), max_iter=100)
        best_solution, best_fitness, _ = optimizer.optimize(weighted_objective)
        
        # Store both objectives
        pareto_front.append((objective1(best_solution), objective2(best_solution)))
    
    return np.array(pareto_front)
```

### Constraint-Based Multi-Objective

```python
def constrained_multi_objective():
    def objective1(x):
        return np.sum(x**2)
    
    def objective2(x):
        return np.sum((x - 1)**2)
    
    def constraint(x):
        return np.sum(x) - 1  # sum(x) >= 1
    
    # Optimize first objective with constraint
    optimizer = CPO(dimensions=3, bounds=([0]*3, [1]*3))
    x1, f1, _ = optimizer.optimize(
        objective_func=objective1,
        f_ieqcons=constraint
    )
    
    # Optimize second objective with constraint
    x2, f2, _ = optimizer.optimize(
        objective_func=objective2,
        f_ieqcons=constraint
    )
    
    return {"obj1": (x1, f1), "obj2": (x2, f2)}
```

## Dynamic Optimization

### Time-Varying Objective

```python
def dynamic_optimization():
    class DynamicObjective:
        def __init__(self):
            self.time = 0
            self.period = 50  # Change every 50 iterations
        
        def __call__(self, x):
            phase = 2 * np.pi * (self.time // self.period)
            offset = 5 * np.sin(phase)  # Moving target
            return np.sum((x - offset)**2)
        
        def update(self):
            self.time += 1
    
    # Initialize
    obj = DynamicObjective()
    optimizer = CPO(dimensions=2, bounds=([-10]*2, [10]*2), max_iter=200)
    
    # Track best solutions over time
    solutions = []
    
    # Run optimization with dynamic updates
    for _ in range(4):  # 4 dynamic updates
        x, f, _ = optimizer.optimize(obj)
        solutions.append({
            'time': obj.time,
            'solution': x.copy(),
            'fitness': f
        })
        
        # Update the objective
        obj.update()
        
        # Re-initialize population around current best
        optimizer.initialize_population_around(x, radius=2.0)
    
    return solutions
```

## Hybrid Optimization

### CPO with Local Search

```python
from scipy.optimize import minimize

def hybrid_cpo_local_search():
    # Define objective (Rastrigin function)
    def rastrigin(x):
        n = len(x)
        return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    # Global search with CPO
    optimizer = CPO(
        dimensions=5,
        bounds=([-5.12]*5, [5.12]*5),
        pop_size=50,
        max_iter=50
    )
    x0, f0, _ = optimizer.optimize(rastrigin)
    
    # Local refinement with L-BFGS-B
    result = minimize(
        rastrigin,
        x0,
        method='L-BFGS-B',
        bounds=[(-5.12, 5.12)]*5,
        options={'maxiter': 100}
    )
    
    return {
        'cpo_solution': (x0, f0),
        'refined_solution': (result.x, result.fun)
    }
```

## Surrogate-Assisted Optimization

### Using Gaussian Process Surrogate

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def surrogate_assisted_optimization():
    # Initialize surrogate model
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    surrogate = GaussianProcessRegressor(kernel=kernel)
    
    # Initial samples (Latin Hypercube Sampling)
    n_initial = 20
    X_train = np.random.uniform(-5, 5, (n_initial, 2))
    y_train = np.array([ackley(x) for x in X_train])
    
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
        y_next = ackley(x_next)
        
        # Update training data
        X_train = np.vstack([X_train, x_next])
        y_train = np.append(y_train, y_next)
    
    # Return best solution found
    best_idx = np.argmin(y_train)
    return X_train[best_idx], y_train[best_idx]
```

## Distributed Optimization

### Parallel Evaluation with Dask

```python
def distributed_optimization():
    from dask.distributed import Client
    
    # Start a local Dask cluster
    client = Client()
    
    # Define objective function
    def expensive_objective(x):
        import time
        time.sleep(0.1)  # Simulate expensive computation
        return np.sum(x**2)
    
    # Scatter data to workers
    x_scattered = client.scatter([np.random.rand(10) for _ in range(100)])
    
    # Define parallel evaluation function
    def evaluate_population(positions):
        futures = [client.submit(expensive_objective, x) for x in positions]
        return np.array(client.gather(futures))
    
    # Initialize optimizer
    optimizer = CPO(
        dimensions=10,
        bounds=([0]*10, [1]*10),
        pop_size=100,
        max_iter=20
    )
    
    # Run optimization with parallel evaluation
    best_solution, best_fitness, _ = optimizer.optimize(
        objective_func=None,  # Will use parallel evaluation
        parallel_func=evaluate_population
    )
    
    client.close()
    return best_solution, best_fitness
```

## Custom Optimization Loop

### Manual Iteration Control

```python
def custom_optimization_loop():
    # Initialize optimizer
    optimizer = CPO(
        dimensions=5,
        bounds=([-5]*5, [5]*5),
        pop_size=30,
        max_iter=100
    )
    
    # Manual iteration loop
    best_fitness_history = []
    
    # Initialize population
    optimizer.initialize_population()
    
    for iteration in range(optimizer.max_iter):
        # Evaluate current population
        fitness = np.array([ackley(ind) for ind in optimizer.positions])
        
        # Update best solution
        best_idx = np.argmin(fitness)
        best_fitness_history.append(fitness[best_idx])
        
        # Print progress
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best fitness = {fitness[best_idx]:.6f}")
        
        # Update population
        optimizer.step(fitness)
        
        # Custom termination condition
        if iteration > 10 and (best_fitness_history[-11] - best_fitness_history[-1]) < 1e-6:
            print(f"Early stopping at iteration {iteration}")
            break
    
    # Get final results
    best_idx = np.argmin([ackley(ind) for ind in optimizer.positions])
    best_solution = optimizer.positions[best_idx]
    best_fitness = ackley(best_solution)
    
    return best_solution, best_fitness, best_fitness_history
```

## Advanced Constraint Handling

### Adaptive Constraint Handling

```python
def adaptive_constraint_handling():
    # Define objective and constraints
    def objective(x):
        return x[0]**2 + x[1]**2 + x[2]**2
    
    def constraint1(x):
        return x[0] + x[1] + x[2] - 1  # x + y + z >= 1
    
    def constraint2(x):
        return 2 - x[0] - x[1]  # x + y <= 2
    
    # Initialize optimizer with adaptive penalty
    class AdaptivePenalty:
        def __init__(self, initial_penalty=1.0, max_penalty=1e6):
            self.penalty = initial_penalty
            self.max_penalty = max_penalty
            self.best_feasible = None
            self.best_violation = float('inf')
        
        def __call__(self, x):
            # Calculate constraint violations
            c1 = max(0, -constraint1(x))  # c1 >= 0
            c2 = max(0, constraint2(x))    # c2 <= 0
            violation = c1 + c2
            
            # Update best feasible solution
            if violation < 1e-6:  # Feasible solution
                if self.best_feasible is None or objective(x) < objective(self.best_feasible):
                    self.best_feasible = x.copy()
            
            # Update best violation
            if violation > 0 and violation < self.best_violation:
                self.best_violation = violation
                self.best_infeasible = x.copy()
            
            # Apply penalty
            return objective(x) + self.penalty * violation**2
        
        def update_penalty(self):
            # Increase penalty if no feasible solution found
            if self.best_feasible is None:
                self.penalty = min(self.penalty * 10, self.max_penalty)
    
    # Initialize
    penalty_func = AdaptivePenalty()
    optimizer = CPO(dimensions=3, bounds=([0]*3, [5]*3), max_iter=200)
    
    # Main optimization loop
    for _ in range(5):  # 5 penalty updates
        x, f, _ = optimizer.optimize(penalty_func)
        print(f"Penalty: {penalty_func.penalty}, Best feasible: {penalty_func.best_feasible is not None}")
        penalty_func.update_penalty()
        
        # Re-initialize population
        if penalty_func.best_feasible is not None:
            optimizer.initialize_population_around(penalty_func.best_feasible, radius=0.5)
        else:
            optimizer.initialize_population_around(penalty_func.best_infeasible, radius=0.5)
    
    # Return best solution found
    if penalty_func.best_feasible is not None:
        return penalty_func.best_feasible, objective(penalty_func.best_feasible)
    else:
        return penalty_func.best_infeasible, objective(penalty_func.best_infeasible)
```

## Performance Tips for Advanced Usage

1. **For High-Dimensional Problems**:
   - Use GPU acceleration for dimensions > 20
   - Consider using dimension reduction techniques
   - Increase population size with dimension

2. **For Noisy Objectives**:
   - Use multiple evaluations per solution
   - Implement resampling strategies
   - Consider surrogate modeling

3. **For Constrained Problems**:
   - Use adaptive penalty methods
   - Consider feasibility rules
   - Implement repair operators for infeasible solutions

4. **For Multi-Modal Problems**:
   - Use niching or crowding techniques
   - Implement restart strategies
   - Consider island models for parallel exploration

5. **For Expensive Objectives**:
   - Use surrogate models
   - Implement efficient parallel evaluation
   - Consider early stopping for poor solutions
