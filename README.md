# Porcupy: Crested Porcupine Optimizer

![CPO](docs/full_logo.png)


[![PyPI version](https://badge.fury.io/py/porcupy.svg)](https://badge.fury.io/py/porcupy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

## Overview

Porcupy is a Python library that implements the Crested Porcupine Optimizer (CPO) algorithm, a nature-inspired metaheuristic optimization technique. The algorithm mimics the defensive behaviors of crested porcupines (sight, sound, odor, and physical attack) to balance exploration and exploitation, with cyclic population reduction for convergence.

This library provides both object-oriented and procedural interfaces for the CPO algorithm, along with visualization tools, benchmark functions, and population management utilities.

## Features

- **Object-oriented implementation** of the Crested Porcupine Optimizer
- **Procedural API** for backward compatibility and simplicity
- **Parallel processing** support for faster optimization
- **Constraint handling** for constrained optimization problems
- **Cyclic population reduction** strategies (linear, cosine, exponential)
- **Visualization tools** for convergence history and search space
- **Extensive benchmark functions** for testing and comparison
- **Comprehensive documentation** and examples

## Installation

```bash
pip install porcupy
```

For visualization support, install with the plotting extras:

```bash
pip install porcupy[plotting]
```

For development, install with the dev extras:

```bash
pip install porcupy[dev]
```

## Quick Start

```python
import numpy as np
from porcupy import CrestPorcupineOptimizer
from porcupy.functions import sphere, get_function_bounds

# Define the problem
dimensions = 10
bounds = get_function_bounds('sphere', dimensions)

# Create the optimizer
optimizer = CrestPorcupineOptimizer(
    dimensions=dimensions,
    bounds=bounds,
    pop_size=30,
    max_iter=100
)

# Run the optimization
best_pos, best_cost, cost_history = optimizer.optimize(
    objective_func=sphere,
    verbose=True
)

print(f"Best position: {best_pos}")
print(f"Best cost: {best_cost}")
```

## Documentation

For detailed documentation, see the [API Reference](docs/api_reference.md) and [User Guide](docs/user_guide.md).

## Algorithm

The Crested Porcupine Optimizer (CPO) algorithm is based on the defensive behaviors of crested porcupines:

1. **Sight Defense**: Exploration mechanism using visual cues
2. **Sound Defense**: Exploration mechanism using auditory signals
3. **Odor Defense**: Exploitation mechanism using olfactory signals
4. **Physical Attack**: Exploitation mechanism using direct interaction

These mechanisms are combined with a cyclic population reduction strategy to balance exploration and exploitation throughout the optimization process.

## Citing

If you use Porcupy in your research, please cite the original paper:

```
@article{article,
author = {Abdel-Basset, Mohamed and Mohamed, Reda and Abouhawwash, Mohamed},
year = {2023},
month = {12},
pages = {111257},
title = {Crested Porcupine Optimizer: A new nature-inspired metaheuristic},
volume = {284},
journal = {Knowledge-Based Systems},
doi = {10.1016/j.knosys.2023.111257}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Development and Testing

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/SammanSarkar/Porcupy.git
cd Porcupy

# Install in development mode with all extras
pip install -e .[all]
```

### Running Tests

To run the tests, use:

```bash
# Recommended way to run tests
python -m pytest tests/

# For verbose output
python -m pytest tests/ -v

# For test coverage report
python -m pytest tests/ --cov=porcupy
```

Note: Using `python -m pytest` is recommended over just `pytest` as it ensures the current directory is in the Python path, which helps with imports.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
