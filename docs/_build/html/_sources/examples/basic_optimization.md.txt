# Basic Optimization Examples

This guide provides practical examples of using Porcupy for various optimization problems.

## Table of Contents
- [Minimizing a Simple Function](#minimizing-a-simple-function)
- [Optimizing with Constraints](#optimizing-with-constraints)
- [Using Different Benchmark Functions](#using-different-benchmark-functions)
- [Visualizing the Optimization Process](#visualizing-the-optimization-process)
- [Saving and Loading Results](#saving-and-loading-results)

## Minimizing a Simple Function

Let's start with a basic example of minimizing the Sphere function:

```python
import numpy as np
from porcupy import CPO
from porcupy.functions import sphere

# Initialize the optimizer
optimizer = CPO(
    dimensions=5,  # 5-dimensional problem
    bounds=([-5.12] * 5, [5.12] * 5),  # Search space bounds
    pop_size=30,    # Number of candidate solutions
    max_iter=100,   # Maximum number of iterations
    ftol=1e-6,      # Function tolerance for early stopping
    verbose=True    # Print progress
)

# Run optimization
best_solution, best_fitness, history = optimizer.optimize(sphere)

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

## Optimizing with Constraints

Porcupy supports both equality and inequality constraints. Here's an example with an inequality constraint:

```python
def constrained_optimization():
    # Define the objective function (Rosenbrock function)
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
    # Define constraint: sum of variables must be >= 1
    def constraint(x):
        return np.sum(x) - 1  # sum(x) >= 1
    
    # Initialize optimizer
    optimizer = CPO(
        dimensions=5,
        bounds=([-5] * 5, [5] * 5),
        pop_size=50,
        max_iter=200
    )
    
    # Run optimization with constraint
    best_solution, best_fitness, _ = optimizer.optimize(
        objective_func=rosenbrock,
        f_ieqcons=constraint  # Inequality constraint (must be >= 0)
    )
    
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print(f"Constraint value: {constraint(best_solution) + 1}")  # Should be >= 1
    
    return best_solution, best_fitness
```

## Using Different Benchmark Functions

Porcupy provides several benchmark functions in the `porcupy.functions` module:

```python
from porcupy.functions import (
    ackley,        # Ackley function
    rastrigin,     # Rastrigin function
    rosenbrock,    # Rosenbrock function
    schwefel,      # Schwefel function
    griewank,      # Griewank function
    michalewicz,   # Michalewicz function
    sphere,        # Sphere function
    schwefel_1_2,  # Schwefel 1.2 function
    schwefel_2_21, # Schwefel 2.21 function
    schwefel_2_22, # Schwefel 2.22 function
    step,          # Step function
    quartic        # Quartic function with noise
)

# Example using the Ackley function
def optimize_ackley():
    optimizer = CPO(
        dimensions=10,
        bounds=([-32.768] * 10, [32.768] * 10),  # Standard bounds for Ackley
        pop_size=50,
        max_iter=200
    )
    
    best_solution, best_fitness, _ = optimizer.optimize(ackley)
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    return best_solution, best_fitness
```

## Visualizing the Optimization Process

Visualization is crucial for understanding how the optimization progresses:

```python
import matplotlib.pyplot as plt

def visualize_optimization():
    # Run optimization with history tracking
    optimizer = CPO(dimensions=2, bounds=([-5, -5], [5, 5]), max_iter=50)
    best_solution, best_fitness, history = optimizer.optimize(ackley)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot convergence
    ax1.plot(history['best_fitness'])
    ax1.set_title('Convergence')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Fitness')
    ax1.grid(True)
    
    # Plot search space exploration
    all_positions = np.vstack(history['population'])
    ax2.hist2d(all_positions[:, 0], all_positions[:, 1], 
               bins=30, range=[[-5, 5], [-5, 5]], cmap='viridis')
    ax2.set_title('Search Space Exploration')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(ax2.collections[0], ax=ax2, label='Visit Count')
    
    plt.tight_layout()
    plt.show()
    
    return best_solution, best_fitness
```

## Saving and Loading Results

For long-running optimizations, it's important to save your results:

```python
import pickle
import json
import numpy as np

def save_optimization_results(filename='optimization_results.pkl'):
    # Run optimization
    optimizer = CPO(dimensions=5, bounds=([-5]*5, [5]*5), max_iter=100)
    best_solution, best_fitness, history = optimizer.optimize(rosenbrock)
    
    # Save results using pickle
    results = {
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'history': history,
        'parameters': {
            'dimensions': 5,
            'bounds': ([-5]*5, [5]*5),
            'max_iter': 100
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    # Also save a human-readable version
    with open('optimization_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'best_solution': best_solution.tolist(),
            'best_fitness': float(best_fitness),
            'parameters': results['parameters']
        }
        json.dump(json_results, f, indent=2)
    
    return results

def load_optimization_results(filename='optimization_results.pkl'):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results
```

## Real-world Example: Hyperparameter Tuning

Here's an example of using Porcupy for hyperparameter tuning of a machine learning model:

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def tune_hyperparameters():
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_classes=2,
        random_state=42
    )
    
    # Define the objective function
    def objective(params):
        """Objective function to minimize (1 - accuracy)"""
        n_estimators = int(params[0])
        max_depth = int(params[1]) if params[1] > 1 else None
        min_samples_split = int(params[2])
        min_samples_leaf = int(params[3])
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        # Use cross-validation for robust evaluation
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        return 1 - np.mean(scores)  # Minimize (1 - accuracy)
    
    # Define parameter bounds
    bounds = [
        (10, 200),      # n_estimators
        (1, 20),        # max_depth
        (2, 20),        # min_samples_split
        (1, 20)         # min_samples_leaf
    ]
    
    # Initialize and run optimizer
    optimizer = CPO(
        dimensions=len(bounds),
        bounds=bounds,
        pop_size=30,
        max_iter=20,
        verbose=True
    )
    
    best_params, best_score, _ = optimizer.optimize(objective)
    
    # Print results
    print("\nBest parameters:")
    print(f"n_estimators: {int(best_params[0])}")
    print(f"max_depth: {int(best_params[1]) if best_params[1] > 1 else None}")
    print(f"min_samples_split: {int(best_params[2])}")
    print(f"min_samples_leaf: {int(best_params[3])}")
    print(f"Best accuracy: {1 - best_score:.4f}")
    
    return best_params, 1 - best_score
```

## Tips for Better Performance

1. **Population Size**:
   - Start with 20-50 for simple problems
   - Use 100-500 for complex or high-dimensional problems
   - Larger populations explore more but are computationally expensive

2. **Iterations**:
   - Start with 100 iterations and increase if needed
   - Monitor convergence to determine the optimal number

3. **Bounds**:
   - Set realistic bounds based on your problem domain
   - Tighter bounds can significantly improve convergence

4. **Parallelization**:
   - Use the `pool` parameter for parallel evaluation
   - Particularly useful for expensive objective functions

5. **GPU Acceleration**:
   - For high-dimensional problems, consider using `GPUCPO`
   - Ensure your problem size justifies the GPU overhead

## Next Steps

- Learn about [advanced optimization techniques](advanced_optimization.md)
- Explore [GPU acceleration](gpu_acceleration.md) for large-scale problems
- Check out [real-world applications](real_world_applications.md) of Porcupy
