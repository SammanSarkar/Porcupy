# GPU Acceleration

This guide explains how to leverage GPU acceleration in Porcupy for faster optimization.

## Prerequisites

Before using GPU acceleration, ensure you have:

1. NVIDIA GPU with CUDA support
2. CUDA Toolkit installed
3. Appropriate version of CuPy for your CUDA version

## Enabling GPU Acceleration

### Installation

Install Porcupy with GPU support:

```bash
# For CUDA 11.x
pip install porcupy[cuda11x]

# For CUDA 12.x
pip install porcupy[cuda12x]
```

### Basic Usage

To use GPU acceleration, simply import `GPUCPO` instead of `CPO`:

```python
from porcupy import GPUCPO
from porcupy.functions import rastrigin

# Initialize GPU-accelerated optimizer
optimizer = GPUCPO(
    dimensions=50,  # Larger dimensions benefit more from GPU
    bounds=([-5.12] * 50, [5.12] * 50),
    pop_size=1024,  # Larger populations benefit more from GPU
    max_iter=100
)

# Run optimization (same API as CPO)
best_solution, best_fitness, _ = optimizer.optimize(rastrigin)
```

## Performance Considerations

### When to Use GPU

GPU acceleration is most beneficial when:
- Problem dimension is high (>20 dimensions)
- Population size is large (>1000 individuals)
- Objective function is computationally expensive
- Multiple optimizations need to be run in parallel

### Performance Tips

1. **Batch Size**: Use powers of 2 for population size (e.g., 256, 512, 1024)
2. **Data Type**: Use `float32` instead of `float64` for better performance
3. **Memory Management**: Monitor GPU memory usage for large problems
4. **CPU-GPU Transfer**: Minimize data transfer between CPU and GPU

## Advanced GPU Configuration

### Memory Management

Control GPU memory usage:

```python
optimizer = GPUCPO(
    # ... other parameters ...
    gpu_params={
        'dtype': 'float32',  # 'float32' or 'float64'
        'device_id': 0,      # GPU device ID
        'memory_fraction': 0.8,  # Max fraction of GPU memory to use
        'pinned_memory': True  # Use pinned memory for faster transfers
    }
)
```

### Multi-GPU Support

To use multiple GPUs, you can distribute evaluations across devices:

```python
import cupy as cp
from porcupy import GPUCPO

# Get number of available GPUs
num_gpus = cp.cuda.runtime.getDeviceCount()

# Run separate optimizations on each GPU
results = []
for device_id in range(num_gpus):
    with cp.cuda.Device(device_id):
        optimizer = GPUCPO(
            dimensions=100,
            bounds=([-5]*100, [5]*100),
            pop_size=1024,
            gpu_params={'device_id': device_id}
        )
        result = optimizer.optimize(rastrigin)
        results.append(result)

# Find best solution across all GPUs
best_solution, best_fitness, _ = min(results, key=lambda x: x[1])
```

## Benchmarking GPU vs CPU

Compare performance between CPU and GPU:

```python
import time
from porcupy import CPO, GPUCPO

def benchmark(optimizer_cls, name, **kwargs):
    start = time.time()
    optimizer = optimizer_cls(**kwargs)
    optimizer.optimize(rastrigin)
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f} seconds")
    return elapsed

# Parameters
dim = 100
pop_size = 1024
max_iter = 50
bounds = ([-5.12] * dim, [5.12] * dim)

# Run benchmarks
cpu_time = benchmark(CPO, "CPU", dimensions=dim, bounds=bounds, 
                   pop_size=pop_size, max_iter=max_iter)
gpu_time = benchmark(GPUCPO, "GPU", dimensions=dim, bounds=bounds,
                   pop_size=pop_size, max_iter=max_iter)

print(f"\nSpeedup: {cpu_time/gpu_time:.2f}x")
```

## Troubleshooting

### Common Issues

1. **CUDA Errors**
   - Verify CUDA and CuPy versions are compatible
   - Check GPU memory usage
   - Try reducing population size or problem dimension

2. **Slow Performance**
   - Ensure you're using the correct GPU (not integrated graphics)
   - Check for CPU-GPU data transfer bottlenecks
   - Try increasing population size for better GPU utilization

3. **Memory Errors**
   - Reduce population size or problem dimension
   - Lower `memory_fraction` parameter
   - Use smaller data types (float32 instead of float64)

## Best Practices

1. **Warm-up Runs**
   - Run a few warm-up iterations before timing
   - GPU performance improves after initial compilation

2. **Batch Processing**
   - Process multiple solutions in batches
   - Use vectorized operations when possible

3. **Memory Management**
   - Clear GPU cache between runs
   - Use context managers for GPU resources

4. **Mixed Precision**
   - Consider using mixed precision training for deep learning applications
   - Combine float16 and float32 where appropriate

## Example: Large-Scale Optimization

```python
import numpy as np
from porcupy import GPUCPO

def complex_objective(x):
    # Complex, computationally expensive function
    return np.sum(x**2) + np.sum(np.sin(x)) + np.sum(np.exp(-x**2))

# High-dimensional optimization
optimizer = GPUCPO(
    dimensions=500,
    bounds=([-10]*500, [10]*500),
    pop_size=2048,
    max_iter=200,
    gpu_params={
        'dtype': 'float32',
        'memory_fraction': 0.9
    }
)

# Run optimization
best_solution, best_fitness, _ = optimizer.optimize(complex_objective)
print(f"Best fitness: {best_fitness:.6f}")
```

This example demonstrates GPU-accelerated optimization of a high-dimensional problem with a complex objective function. The GPU acceleration significantly reduces the computation time compared to CPU.
