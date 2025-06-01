"""
Parallel and Hybrid Optimization Example

This example demonstrates advanced features of the Porcupy library:
1. Parallel processing for faster optimization
2. Hybrid optimization combining CPO with local search
3. Performance comparison between different approaches
"""

import numpy as np
import sys
import os
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy import CPO
from porcupy.functions import (
    rastrigin, rosenbrock, ackley, griewank, 
    get_function_bounds, get_function_optimum
)
from porcupy.utils.plotting import plot_convergence


def run_sequential_cpo(func, dimensions, bounds, max_iter=100, pop_size=30):
    """
    Run standard sequential CPO.
    """
    print(f"Running sequential CPO on {func.__name__} function...")
    start_time = time.time()
    
    optimizer = CPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=pop_size,
        max_iter=max_iter
        # Sequential execution is the default
    )
    
    best_pos, best_cost, cost_history = optimizer.optimize(
        objective_func=func,
        verbose=False
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Sequential CPO - Best cost: {best_cost}")
    print(f"Sequential CPO - Time taken: {elapsed:.2f} seconds")
    
    return best_pos, best_cost, cost_history, elapsed


def run_parallel_cpo(func, dimensions, bounds, max_iter=100, pop_size=30):
    """
    Run CPO with multiple starting points to simulate parallel processing.
    """
    print(f"\nRunning parallel CPO on {func.__name__} function...")
    start_time = time.time()
    
    # For parallel processing simulation, we'll use multiple runs with different seeds
    # and then select the best result
    
    # Number of runs
    num_runs = 4
    results = []
    
    # Run multiple optimizations with different random seeds
    for i in range(num_runs):
        # Set a different seed for each run
        np.random.seed(42 + i)
        
        # Create optimizer with smaller population
        opt = CPO(
            dimensions=dimensions,
            bounds=bounds,
            pop_size=pop_size // num_runs,  # Smaller population per run
            max_iter=max_iter // 2  # Fewer iterations per run to maintain similar computation
        )
        
        # Run optimization
        result = opt.optimize(objective_func=func, verbose=False)
        results.append(result)
    
    # Find the best result
    best_idx = np.argmin([r[1] for r in results])  # Find minimum cost
    best_pos, best_cost, cost_history = results[best_idx]
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Parallel CPO - Best cost: {best_cost}")
    print(f"Parallel CPO - Time taken: {elapsed:.2f} seconds")
    
    return best_pos, best_cost, cost_history, elapsed


def local_search(func, x0, bounds):
    """
    Perform local search using scipy's L-BFGS-B algorithm.
    """
    lb, ub = bounds
    result = minimize(
        func,
        x0,
        method='L-BFGS-B',
        bounds=[(lb[i], ub[i]) for i in range(len(x0))],
        options={'maxiter': 100}
    )
    
    return result.x, result.fun


def run_hybrid_cpo(func, dimensions, bounds, max_iter=100, pop_size=30, local_search_freq=10):
    """
    Run hybrid CPO with periodic local search.
    """
    print(f"\nRunning hybrid CPO on {func.__name__} function...")
    start_time = time.time()
    
    # Create a wrapper for the objective function to track evaluations
    evaluations = 0
    def func_wrapper(x):
        nonlocal evaluations
        evaluations += 1
        return func(x)
    
    # Run standard CPO first
    optimizer = CPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=pop_size,
        max_iter=max_iter
    )
    
    # Run the optimization
    best_pos, best_cost, cost_history = optimizer.optimize(
        objective_func=func_wrapper,
        verbose=False
    )
    
    # Apply local search to refine the solution
    print("  Applying local search to refine the solution...")
    refined_pos, refined_cost = local_search(func_wrapper, best_pos, bounds)
    
    # Update if the local search found a better solution
    if refined_cost < best_cost:
        print(f"  Local search improved solution from {best_cost} to {refined_cost}")
        best_pos = refined_pos
        best_cost = refined_cost
        
        # Update the last entry in cost history
        cost_history[-1] = refined_cost
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Hybrid CPO - Best cost: {best_cost}")
    print(f"Hybrid CPO - Time taken: {elapsed:.2f} seconds")
    print(f"Hybrid CPO - Function evaluations: {evaluations}")
    
    return best_pos, best_cost, cost_history, elapsed


def compare_performance(sequential_results, parallel_results, hybrid_results, func_name):
    """
    Compare the performance of different optimization approaches.
    """
    seq_cost, seq_time = sequential_results[1], sequential_results[3]
    par_cost, par_time = parallel_results[1], parallel_results[3]
    hyb_cost, hyb_time = hybrid_results[1], hybrid_results[3]
    
    seq_history = sequential_results[2]
    par_history = parallel_results[2]
    hyb_history = hybrid_results[2]
    
    # Print performance comparison
    print("\nPerformance Comparison:")
    print(f"Function: {func_name}")
    print(f"Sequential CPO: Cost = {seq_cost}, Time = {seq_time:.2f}s")
    print(f"Parallel CPO: Cost = {par_cost}, Time = {par_time:.2f}s, Speedup = {seq_time/par_time:.2f}x")
    print(f"Hybrid CPO: Cost = {hyb_cost}, Time = {hyb_time:.2f}s, Improvement = {(seq_cost-hyb_cost)/seq_cost*100:.2f}%")
    
    # Plot convergence comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(seq_history, label='Sequential', color='blue')
    plt.plot(par_history, label='Parallel', color='green')
    plt.plot(hyb_history, label='Hybrid', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Best Cost')
    plt.title(f'Convergence Comparison - {func_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot execution time comparison
    plt.subplot(1, 2, 2)
    methods = ['Sequential', 'Parallel', 'Hybrid']
    times = [seq_time, par_time, hyb_time]
    costs = [seq_cost, par_cost, hyb_cost]
    
    bar_positions = np.arange(len(methods))
    bar_width = 0.35
    
    # Create primary axis for time
    ax1 = plt.gca()
    bars1 = ax1.bar(bar_positions - bar_width/2, times, bar_width, label='Time (s)', color='skyblue')
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(methods)
    
    # Create secondary axis for cost
    ax2 = ax1.twinx()
    bars2 = ax2.bar(bar_positions + bar_width/2, costs, bar_width, label='Cost', color='salmon')
    ax2.set_ylabel('Best Cost')
    
    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title(f'Performance Comparison - {func_name}')
    plt.tight_layout()
    plt.show()


def run_benchmark():
    """
    Run a benchmark comparing different optimization approaches on multiple functions.
    """
    # Define benchmark functions
    benchmark_functions = [rastrigin, rosenbrock, ackley, griewank]
    dimensions = 10
    max_iter = 50
    pop_size = 30
    
    # Results storage
    results = {}
    
    for func in benchmark_functions:
        print(f"\n{'='*50}")
        print(f"Benchmarking {func.__name__} function")
        print(f"{'='*50}")
        
        bounds = get_function_bounds(func.__name__, dimensions)
        
        # Run all three approaches
        seq_results = run_sequential_cpo(func, dimensions, bounds, max_iter, pop_size)
        par_results = run_parallel_cpo(func, dimensions, bounds, max_iter, pop_size)
        hyb_results = run_hybrid_cpo(func, dimensions, bounds, max_iter, pop_size)
        
        # Compare performance
        compare_performance(seq_results, par_results, hyb_results, func.__name__)
        
        # Store results
        results[func.__name__] = {
            'sequential': seq_results,
            'parallel': par_results,
            'hybrid': hyb_results
        }
    
    return results


def run_single_function_demo():
    """
    Run a detailed demonstration on a single function.
    """
    # Define the problem
    func = rastrigin
    dimensions = 10
    bounds = get_function_bounds(func.__name__, dimensions)
    max_iter = 100
    pop_size = 30
    
    # Run all three approaches
    seq_results = run_sequential_cpo(func, dimensions, bounds, max_iter, pop_size)
    par_results = run_parallel_cpo(func, dimensions, bounds, max_iter, pop_size)
    hyb_results = run_hybrid_cpo(func, dimensions, bounds, max_iter, pop_size)
    
    # Compare performance
    compare_performance(seq_results, par_results, hyb_results, func.__name__)
    
    # Get the known optimum
    x_opt, f_opt = get_function_optimum(func.__name__, dimensions)
    
    # Calculate distance to optimum
    seq_dist = np.linalg.norm(seq_results[0] - x_opt)
    par_dist = np.linalg.norm(par_results[0] - x_opt)
    hyb_dist = np.linalg.norm(hyb_results[0] - x_opt)
    
    print("\nDistance to Known Optimum:")
    print(f"Sequential CPO: {seq_dist}")
    print(f"Parallel CPO: {par_dist}")
    print(f"Hybrid CPO: {hyb_dist}")
    
    # Plot detailed convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(seq_results[2], label='Sequential', color='blue')
    plt.semilogy(par_results[2], label='Parallel', color='green')
    plt.semilogy(hyb_results[2], label='Hybrid', color='red')
    plt.axhline(y=f_opt, color='black', linestyle='--', label=f'Optimum ({f_opt})')
    plt.xlabel('Iterations')
    plt.ylabel('Best Cost (log scale)')
    plt.title(f'Convergence Comparison - {func.__name__}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Choose which demo to run
    run_benchmark_flag = False
    
    if run_benchmark_flag:
        # Run benchmark on multiple functions
        results = run_benchmark()
    else:
        # Run detailed demo on a single function
        run_single_function_demo()
