"""
Enhanced Visualization Demo for the Crested Porcupine Optimizer

This example demonstrates the enhanced visualization capabilities of the Porcupy library,
including the improved best position tracking, convergence plot, and information panel.
"""

import numpy as np
import matplotlib.pyplot as plt
from porcupy.cpo_class import CPO
from porcupy.functions import rastrigin, ackley, rosenbrock
from porcupy.utils.enhanced_visualization import animate_porcupines_2d
import os

def run_demo(function_name="rastrigin", dimensions=2, population_size=30, max_iterations=50):
    """
    Run the enhanced visualization demo with the specified function.
    
    Parameters
    ----------
    function_name : str, optional
        Name of the objective function to use (default: "rastrigin").
    dimensions : int, optional
        Number of dimensions for the optimization problem (default: 2).
    population_size : int, optional
        Number of porcupines in the population (default: 30).
    max_iterations : int, optional
        Maximum number of iterations (default: 50).
    """
    # Select the objective function
    function_map = {
        "rastrigin": rastrigin,
        "ackley": ackley,
        "rosenbrock": rosenbrock
    }
    
    if function_name not in function_map:
        raise ValueError(f"Unknown function: {function_name}. Available functions: {list(function_map.keys())}")
    
    objective_func = function_map[function_name]
    
    # Define search space bounds
    if function_name == "rosenbrock":
        lb = -5.0 * np.ones(dimensions)
        ub = 10.0 * np.ones(dimensions)
    else:
        lb = -5.12 * np.ones(dimensions)
        ub = 5.12 * np.ones(dimensions)
    
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer
    optimizer = CPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=population_size,
        max_iter=max_iterations,
        cycles=5
    )
    
    print(f"Running optimization on {function_name} function ({dimensions}D) with {population_size} porcupines...")
    
    # Run the optimization with history tracking enabled
    best_position, best_cost, cost_history = optimizer.optimize(
        objective_func,
        verbose=True,
        track_history=True
    )
    
    print(f"\nOptimization finished!")
    print(f"Best position: {best_position}")
    print(f"Best cost: {best_cost:.6f}")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the enhanced animation
    print("\nCreating visualization...")
    
    # First, create the animation without saving to test if it works
    try:
        # Create a test figure with the animation
        test_fig = plt.figure(figsize=(14, 10))
        anim = animate_porcupines_2d(
            position_history=optimizer.positions_history,
            func=objective_func,
            bounds=bounds,
            defense_history=optimizer.defense_types_history,
            best_pos_history=optimizer.best_positions_history,
            best_cost_history=cost_history,
            interval=200,
            figsize=(14, 10),
            show_trail=True,
            max_trail_length=20,
            show_convergence=True,
            save_path=None  # Don't save yet
        )
        
        # If we get here, the animation works, so try to save it
        try:
            # Try saving as MP4
            save_path = os.path.join(output_dir, f"{function_name}_optimization.mp4")
            print(f"Saving animation as MP4: {os.path.abspath(save_path)}")
            anim.save(save_path, writer='ffmpeg', fps=5, dpi=100)
            print("Successfully saved MP4 animation!")
        except Exception as e:
            print(f"Could not save MP4: {e}")
            try:
                # Fallback to GIF
                save_path = os.path.join(output_dir, f"{function_name}_optimization.gif")
                print(f"Trying to save as GIF instead: {os.path.abspath(save_path)}")
                anim.save(save_path, writer='pillow', fps=5, dpi=100)
                print("Successfully saved GIF animation!")
            except Exception as e2:
                print(f"Could not save GIF either: {e2}")
                save_path = None
        
        # Show the animation
        try:
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display animation: {e}")
            if save_path:
                print(f"Animation saved to: {os.path.abspath(save_path)}")
            else:
                print("Could not save or display the animation.")
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        print("Please check if you have all required dependencies installed.")
        print("You may need to install 'ffmpeg' for MP4 support or 'pillow' for GIF support.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Visualization Demo for CPO")
    parser.add_argument("--function", type=str, default="rastrigin",
                        help="Objective function (rastrigin, ackley, rosenbrock)")
    parser.add_argument("--dimensions", type=int, default=2,
                        help="Number of dimensions (default: 2)")
    parser.add_argument("--population", type=int, default=30,
                        help="Population size (default: 30)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Maximum number of iterations (default: 50)")
    
    args = parser.parse_args()
    
    run_demo(
        function_name=args.function.lower(),
        dimensions=args.dimensions,
        population_size=args.population,
        max_iterations=args.iterations
    )
