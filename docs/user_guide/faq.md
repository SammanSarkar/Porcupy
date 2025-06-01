# Frequently Asked Questions (FAQ)

## General Questions

### What is Porcupy?
Porcupy is a Python implementation of the Crested Porcupine Optimizer (CPO), a nature-inspired metaheuristic optimization algorithm. It's designed for solving complex optimization problems efficiently.

### How is CPO different from other optimization algorithms?
CPO is inspired by the defense mechanisms of crested porcupines and uses four main strategies (sight, sound, odor, and physical defense) to explore and exploit the search space. It's particularly effective for high-dimensional problems and can handle both continuous and discrete optimization.

### What types of optimization problems can Porcupy solve?
Porcupy can solve:
- Unconstrained optimization problems
- Constrained optimization problems
- Single-objective optimization
- High-dimensional problems
- Both convex and non-convex problems

## Installation

### How do I install Porcupy with GPU support?
```bash
# For CUDA 11.x
pip install porcupy[cuda11x]

# For CUDA 12.x
pip install porcupy[cuda12x]
```

### What are the system requirements?
- Python 3.7+
- NumPy
- SciPy
- Matplotlib (for visualization)
- CuPy (for GPU support)
- NVIDIA GPU with CUDA (for GPU acceleration)

### I'm getting import errors. What should I do?
1. Make sure you've installed all required dependencies
2. Check your Python version (3.7+ required)
3. Verify that your PYTHONPATH includes the directory containing Porcupy
4. If using GPU, ensure CUDA and CuPy are properly installed

## Usage

### How do I define my objective function?
Your objective function should take a numpy array as input and return a scalar value:

```python
def my_objective(x):
    # x is a numpy array of decision variables
    return np.sum(x**2)  # Example: minimize sum of squares
```

### How do I handle constraints?
You can handle constraints using penalty functions or by implementing a custom constraint handling mechanism:

```python
# Penalty function approach
def constrained_objective(x):
    # Objective
    obj = np.sum(x**2)
    
    # Constraint: x[0] + x[1] >= 1
    penalty = max(0, 1 - (x[0] + x[1]))**2
    
    return obj + 1e6 * penalty  # Large penalty for constraint violation
```

### How do I choose the right parameters?
- **Population size**: Start with 20-50 for simple problems, 100+ for complex ones
- **Iterations**: 100-1000 depending on problem complexity
- **Bounds**: Set based on your problem's domain knowledge
- **Defense weights**: Default values work well for most problems

## Performance

### How can I speed up optimization?
1. Use GPU acceleration for large problems
2. Reduce population size (but not too small)
3. Use fewer iterations with proper convergence criteria
4. Implement parallel evaluation of the objective function
5. Use a faster objective function implementation

### When should I use GPU acceleration?
GPU acceleration is most beneficial when:
- Problem dimension is high (>20 dimensions)
- Population size is large (>1000 individuals)
- Objective function evaluations are parallelizable
- You have a CUDA-compatible NVIDIA GPU

### How do I know if my optimization has converged?
Monitor the best fitness value over iterations. If it stops improving significantly (based on `ftol`), the optimization has likely converged. You can also plot the convergence history to visualize progress.

## Troubleshooting

### The optimizer is stuck in a local minimum. What can I do?
1. Increase population size for better exploration
2. Adjust defense mechanism weights
3. Try different initial population strategies
4. Use a hybrid approach with local search
5. Implement a restart mechanism

### I'm getting NaN or infinite values. What's wrong?
This can happen due to:
1. Numerical instability in the objective function
2. Poorly scaled variables (try normalizing your inputs)
3. Division by zero in your objective function
4. Extreme parameter values

### The optimization is too slow. How can I profile it?
Use Python's built-in profiler:

```python
import cProfile

def run_optimization():
    # Your optimization code here
    pass

cProfile.run('run_optimization()', sort='cumtime')
```

## Advanced Topics

### How can I implement custom defense mechanisms?
Subclass the `DefenseMechanisms` class and implement your custom logic. See the [Advanced Features](advanced_features.md) guide for examples.

### Can I use Porcupy for multi-objective optimization?
Porcupy is primarily designed for single-objective optimization. For multi-objective problems, you can:
1. Use a weighted sum approach
2. Implement a custom multi-objective extension
3. Use Porcupy as part of a larger multi-objective optimization framework

### How do I save and load optimization results?
Use Python's `pickle` module to save and load the optimizer state:

```python
import pickle

# Save
with open('optimizer_state.pkl', 'wb') as f:
    pickle.dump(optimizer, f)

# Load
with open('optimizer_state.pkl', 'rb') as f:
    optimizer = pickle.load(f)
```

## Contributing

### How can I contribute to Porcupy?
1. Fork the repository
2. Create a new branch for your feature
3. Write tests for your changes
4. Submit a pull request

### What coding standards should I follow?
- Follow PEP 8 style guide
- Write docstrings for all public functions and classes
- Include type hints for better code clarity
- Write unit tests for new features

### How do I run the test suite?
```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run a specific test file
pytest tests/test_cpo.py -v
```

## Getting Help

### Where can I ask questions or report issues?
- [GitHub Issues](https://github.com/SammanSarkar/Porcupy/issues) for bug reports and feature requests
- [GitHub Discussions](https://github.com/SammanSarkar/Porcupy/discussions) for questions and discussions

### How can I cite Porcupy?
If you use Porcupy in your research, please cite the original CPO paper and include a link to the GitHub repository.

## Performance Tips

### For CPU Optimization
- Use NumPy vectorized operations
- Avoid Python loops in the objective function
- Use appropriate data types (e.g., `float32` instead of `float64` if precision allows)
- Set `OMP_NUM_THREADS` environment variable to control parallelization

### For GPU Optimization
- Use powers of 2 for population size
- Minimize CPU-GPU data transfers
- Use `float32` instead of `float64` when possible
- Monitor GPU memory usage

### For Large-Scale Problems
- Use batch processing for objective function evaluations
- Implement checkpointing for long-running optimizations
- Consider using a distributed computing framework for very large problems

## Common Error Messages

### "ModuleNotFoundError: No module named 'porcupy'"
This means Porcupy is not installed or not in your Python path. Try:
1. Reinstalling the package: `pip install -e .`
2. Checking your Python environment
3. Verifying your PYTHONPATH

### "CUDA error: out of memory"
Reduce the population size or problem dimension, or free up GPU memory.

### "Objective function returned NaN"
Check your objective function for numerical stability issues, divisions by zero, or invalid operations.

### "Optimization did not converge"
Try increasing the maximum number of iterations or adjusting the convergence tolerance.
