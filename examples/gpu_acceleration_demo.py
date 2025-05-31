"""
GPU Acceleration Demo for Porcupy

This example demonstrates how to use the GPU-accelerated Crested Porcupine Optimizer (CPO)
to solve optimization problems. The example compares the performance between CPU and GPU
implementations on a standard benchmark function.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from porcupy import CPO
try:
    from porcupy.gpu_cpo import GPUCPO
except ImportError:
    GPUCPO = None
    print("GPU acceleration not available. Install CuPy for GPU support.")
from porcupy.functions import rastrigin

def run_optimization(optimizer_class, objective_func, dimensions=10, pop_size=100, max_iter=100):
    """Run optimization with the specified optimizer class."""
    # Define problem bounds
    lb = [-5.12] * dimensions
    ub = [5.12] * dimensions
    
    # Create and run optimizer
    optimizer = optimizer_class(
        dimensions=dimensions,
        bounds=(np.array(lb), np.array(ub)),
        pop_size=pop_size,
        max_iter=max_iter
    )
    
    start_time = time.time()
    best_pos, best_cost, cost_history = optimizer.optimize(
        objective_func=objective_func,
        verbose=True
    )
    elapsed = time.time() - start_time
    
    return best_pos, best_cost, cost_history, elapsed

def main():
    """Run the GPU acceleration demo."""
    print("GPU Acceleration Demo for Porcupy")
    print("=" * 50)
    
    # Problem parameters
    dimensions = 30
    pop_size = 1000
    max_iter = 100
    
    print(f"Problem: Rastrigin function ({dimensions}D)")
    print(f"Population size: {pop_size}")
    print(f"Max iterations: {max_iter}")
    
    # Run CPU version
    print("\nRunning CPU version...")
    _, _, cpu_history, cpu_time = run_optimization(
        CPO, rastrigin, dimensions, pop_size, max_iter
    )
    print(f"CPU time: {cpu_time:.2f} seconds")
    
    # Run GPU version
    print("\nRunning GPU version...")
    try:
        _, _, gpu_history, gpu_time = run_optimization(
            GPUCPO, rastrigin, dimensions, pop_size, max_iter
        )
        print(f"GPU time: {gpu_time:.2f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(cpu_history, label='CPU')
        plt.plot(gpu_history, label='GPU')
        plt.title('Convergence Comparison (CPU vs GPU)')
        plt.xlabel('Iteration')
        plt.ylabel('Best Cost')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Error running GPU version: {e}")
        print("Make sure you have CuPy installed with CUDA support.")
        print("Install with: pip install cupy-cuda11x")

if __name__ == "__main__":
    main()
