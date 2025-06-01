# Porcupy Documentation

![Porcupy Logo](logo.png)

**Porcupy** is a powerful Python library implementing the Crested Porcupine Optimizer (CPO) algorithm, featuring GPU acceleration and advanced optimization capabilities. Based on the paper "Crested Porcupine Optimizer: A new nature-inspired metaheuristic" by Mohamed Abdel-Basset et al., Porcupy is designed for researchers and developers working on complex optimization tasks.

## Key Features

- **GPU Acceleration** - Optimize with the power of CUDA-enabled GPUs
- **Multiple Optimization Modes** - Support for constrained and unconstrained problems
- **Advanced Visualization** - Built-in tools for analyzing optimization progress
- **High Performance** - Optimized for both speed and accuracy
- **Extensible Architecture** - Easy to customize and extend

## Quick Start

```python
from porcupy import CPO
from porcupy.functions import rastrigin

# Initialize the optimizer
optimizer = CPO(
    dimensions=10,
    bounds=([-5.12] * 10, [5.12] * 10),
    pop_size=50,
    max_iter=100
)

# Run optimization
best_solution, best_fitness, _ = optimizer.optimize(rastrigin)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

## Documentation Contents

### Getting Started
- [Installation](getting_started/installation.md) - Set up Porcupy with CPU or GPU support
- [Quick Start](getting_started/quickstart.md) - Run your first optimization in minutes
- [Configuration](getting_started/configuration.md) - Customize Porcupy's behavior

### User Guide
- [Basic Usage](user_guide/basic_usage.md) - Core concepts and basic optimization
- [GPU Acceleration](user_guide/gpu_acceleration.md) - Speed up optimization with GPUs
- [Visualization](user_guide/visualization.md) - Tools for analyzing results
- [Advanced Features](user_guide/advanced_features.md) - Advanced usage patterns

### Examples
- [Basic Optimization](examples/basic_optimization.md) - Simple optimization problems
- [Custom Defense Mechanisms](examples/custom_defense.md) - Implementing custom behaviors
- [Parallel Processing](examples/parallel_processing.md) - Scaling up with parallel execution

### API Reference
- [Core API](api_reference/core.md) - Main optimization classes and functions
- [GPU Module](api_reference/gpu.md) - GPU-accelerated optimization
- [Utilities](api_reference/utils.md) - Helper functions and visualization tools

## Community

- [GitHub Issues](https://github.com/SammanSarkar/Porcupy/issues) - Report bugs or request features
- [Discussions](https://github.com/SammanSarkar/Porcupy/discussions) - Ask questions and share ideas
- [Contributing](development/contributing.md) - Contribute to Porcupy

## License

Porcupy is released under the MIT License. See [LICENSE](https://github.com/SammanSarkar/Porcupy/blob/main/LICENSE) for details.