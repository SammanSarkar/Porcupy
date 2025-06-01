"""
Backward Compatibility Example

This example demonstrates how to use the Porcupy library with both the new
object-oriented interface and the legacy procedural interface, showing how
to achieve the same results with both approaches.
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

# Import both interfaces
from porcupy import CPO  # Object-oriented interface
from porcupy.cpo import cpo  # Procedural interface
from porcupy.functions import rastrigin, get_function_bounds
from porcupy.utils.plotting import plot_convergence


def run_oo_interface():
    """
    Run optimization using the object-oriented interface.
    """
    print("Running optimization with object-oriented interface...")
    
    # Define the problem
    dimensions = 10
    bounds = get_function_bounds('rastrigin', dimensions)
    
    # Create the optimizer with specific parameters
    start_time = time.time()
    optimizer = CPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=30,
        max_iter=100,
        alpha=0.2,
        tf=0.8,
        cycles=2,
        min_pop_size=10
    )
    
    # Run the optimization
    best_pos, best_cost, cost_history = optimizer.optimize(
        objective_func=rastrigin,
        verbose=True
    )
    end_time = time.time()
    
    print(f"OO Interface - Best cost: {best_cost}")
    print(f"OO Interface - Time taken: {end_time - start_time:.2f} seconds")
    
    return best_pos, best_cost, cost_history


def run_procedural_interface():
    """
    Run optimization with the procedural interface.
    """
    print("Running optimization with procedural interface...")
    start_time = time.time()
    
    # Define the problem
    dimensions = 10
    bounds = get_function_bounds('rastrigin', dimensions)
    
    # Convert bounds to lb and ub lists for procedural interface
    lb = bounds[0].tolist()
    ub = bounds[1].tolist()
    
    # Run the optimization using the procedural interface
    best_pos, best_cost, cost_history = cpo(
        fobj=rastrigin,
        lb=lb,
        ub=ub,
        pop_size=30,
        max_iter=100,
        verbose=True
    )
    end_time = time.time()
    
    print(f"Procedural Interface - Best cost: {best_cost}")
    print(f"Procedural Interface - Time taken: {end_time - start_time:.2f} seconds")
    
    return best_pos, best_cost, cost_history


def compare_results(oo_results, proc_results):
    """
    Compare the results from both interfaces.
    """
    _, oo_cost, oo_history = oo_results
    _, proc_cost, proc_history = proc_results
    
    print("\nResults Comparison:")
    print(f"OO Interface Best Cost: {oo_cost}")
    print(f"Procedural Interface Best Cost: {proc_cost}")
    print(f"Cost Difference: {abs(oo_cost - proc_cost)}")
    
    # Plot convergence comparison
    plt.figure(figsize=(10, 6))
    plt.plot(oo_history, label='Object-Oriented Interface', color='blue')
    plt.plot(proc_history, label='Procedural Interface', color='red', linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Best Cost')
    plt.title('Convergence Comparison: OO vs Procedural Interface')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def migrate_from_procedural_to_oo():
    """
    Demonstrate how to migrate from procedural to object-oriented interface.
    """
    print("\nMigrating from procedural to object-oriented interface...")
    
    # Step 1: Start with procedural code
    print("Step 1: Original procedural code")
    dimensions = 2
    lb = [-5.12] * dimensions
    ub = [5.12] * dimensions
    
    # Step 2: Convert to object-oriented equivalent
    print("Step 2: Equivalent object-oriented code")
    bounds = (np.array(lb), np.array(ub))
    
    # Show the conversion
    print("Procedural parameters:")
    print(f"  lb = {lb}")
    print(f"  ub = {ub}")
    print(f"  pop_size = 30")
    print(f"  max_iter = 50")
    print(f"  reduction_strategy = 'linear'")
    print(f"  min_pop_size = 5")
    
    print("\nObject-oriented equivalent:")
    print("  optimizer = CPO(")
    print(f"      dimensions={dimensions},")
    print(f"      bounds={bounds},")
    print(f"      pop_size=30,")
    print(f"      max_iter=50,")
    print(f"      options={{")
    print(f"          'reduction_strategy': 'linear',")
    print(f"          'min_pop_size': 5")
    print(f"      }}")
    print("  )")
    
    # Step 3: Show how to run the optimization
    print("\nStep 3: Running the optimization")
    print("Procedural:")
    print("  best_pos, best_cost, cost_history = cpo(")
    print("      objective_func=rastrigin,")
    print("      lb=lb,")
    print("      ub=ub,")
    print("      pop_size=30,")
    print("      max_iter=50,")
    print("      verbose=True")
    print("  )")
    
    print("\nObject-oriented:")
    print("  best_pos, best_cost, cost_history = optimizer.optimize(")
    print("      objective_func=rastrigin,")
    print("      verbose=True")
    print("  )")


if __name__ == "__main__":
    # Run both interfaces and compare results
    oo_results = run_oo_interface()
    proc_results = run_procedural_interface()
    compare_results(oo_results, proc_results)
    
    # Show migration path
    migrate_from_procedural_to_oo()
