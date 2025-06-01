"""
Custom Defense Mechanisms Example

This example demonstrates how to customize the defense mechanisms in the
Crested Porcupine Optimizer (CPO) algorithm by creating a custom DefenseMechanisms
class and integrating it with the optimizer.
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy import CPO
from porcupy.porcupines import DefenseMechanisms, PorcupinePopulation
from porcupy.functions import rastrigin, get_function_bounds
from porcupy.utils.visualization import plot_2d_search_space, animate_optimization_2d


class CustomDefenseMechanisms(DefenseMechanisms):
    """
    Custom implementation of defense mechanisms with modified behaviors.
    
    This class overrides the standard defense mechanisms to demonstrate
    how to customize the algorithm's behavior.
    """
    
    def __init__(self, alpha=0.2, beta=0.4, gamma=0.6):
        """
        Initialize with custom parameters.
        
        Parameters
        ----------
        alpha : float
            Control parameter for sight defense (default: 0.2)
        beta : float
            Control parameter for sound defense (default: 0.4)
        gamma : float
            Control parameter for odor defense (default: 0.6)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def sight_defense(self, position, other_position, best_position):
        """
        Custom sight defense mechanism with enhanced exploration.
        
        This version increases the randomness to promote wider exploration.
        """
        r1 = np.random.random(position.shape)
        r2 = np.random.random(position.shape)
        
        # Enhanced randomness for better exploration
        random_position = position + self.alpha * r1 * (other_position - position)
        random_position += self.alpha * r2 * (np.random.random(position.shape) - 0.5) * 2
        
        return random_position
    
    def sound_defense(self, position, other_positions, weights):
        """
        Custom sound defense mechanism with weighted influence.
        
        This version uses a weighted average of other positions based on their fitness.
        """
        if len(other_positions) == 0:
            return position
        
        # Create a weighted average based on normalized weights
        normalized_weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)
        
        # Compute the weighted center
        center = np.zeros_like(position)
        for i, pos in enumerate(other_positions):
            center += normalized_weights[i] * pos
            
        # Move toward the weighted center
        r = np.random.random(position.shape)
        return position + self.beta * r * (center - position)
    
    def odor_defense(self, position, best_position):
        """
        Custom odor defense mechanism with adaptive step size.
        
        This version adjusts the step size based on the distance to the best position.
        """
        distance = np.linalg.norm(best_position - position)
        adaptive_gamma = self.gamma * np.exp(-0.1 * distance)  # Smaller steps when far away
        
        r = np.random.random(position.shape)
        return position + adaptive_gamma * r * (best_position - position)
    
    def physical_attack(self, position, best_position):
        """
        Custom physical attack mechanism with momentum.
        
        This version adds a momentum term to help escape local optima.
        """
        if not hasattr(self, 'previous_direction'):
            self.previous_direction = np.zeros_like(position)
        
        # Calculate new direction with momentum
        current_direction = best_position - position
        momentum = 0.2 * self.previous_direction
        combined_direction = current_direction + momentum
        
        # Update for next iteration
        self.previous_direction = combined_direction
        
        r = np.random.random(position.shape)
        return position + r * combined_direction


def create_custom_cpo(dimensions, bounds, pop_size, max_iter):
    """
    Create a CPO optimizer with custom defense mechanisms.
    """
    # Create a subclass of CPO that uses our custom defense mechanisms
    class CustomCPO(CPO):
        def _create_defense_mechanisms(self):
            return CustomDefenseMechanisms(alpha=self.alpha, tf=self.tf)
    
    # Create the custom optimizer
    optimizer = CustomCPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=pop_size,
        max_iter=max_iter,
        alpha=0.3,  # Custom alpha value
        tf=0.8      # Standard tf value
    )
    
    return optimizer


def run_standard_cpo(dimensions, bounds, pop_size, max_iter, objective_func):
    """
    Run the standard CPO algorithm.
    """
    print("Running standard CPO...")
    
    optimizer = CPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=pop_size,
        max_iter=max_iter
    )
    
    best_pos, best_cost, cost_history = optimizer.optimize(
        objective_func=objective_func,
        verbose=True
    )
    
    print(f"Standard CPO - Best cost: {best_cost}")
    return best_pos, best_cost, cost_history, optimizer.position_history


def run_custom_cpo(dimensions, bounds, pop_size, max_iter, objective_func):
    """
    Run CPO with custom defense mechanisms.
    """
    print("\nRunning CPO with custom defense mechanisms...")
    start_time = time.time()
    
    # Create a CPO optimizer with custom defense mechanisms
    optimizer = create_custom_cpo(dimensions, bounds, pop_size, max_iter)
    
    # Run the optimization
    best_pos, best_cost, cost_history = optimizer.optimize(
        objective_func=objective_func,
        verbose=True
    )
    
    print(f"Custom CPO - Best cost: {best_cost}")
    return best_pos, best_cost, cost_history, optimizer.position_history


def compare_results(standard_results, custom_results):
    """
    Compare the results from standard and custom CPO.
    """
    _, std_cost, std_history, _ = standard_results
    _, custom_cost, custom_history, _ = custom_results
    
    print("\nResults Comparison:")
    print(f"Standard CPO Best Cost: {std_cost}")
    print(f"Custom CPO Best Cost: {custom_cost}")
    print(f"Improvement: {(std_cost - custom_cost) / std_cost * 100:.2f}%")
    
    # Plot convergence comparison
    plt.figure(figsize=(10, 6))
    plt.plot(std_history, label='Standard CPO', color='blue')
    plt.plot(custom_history, label='Custom CPO', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Best Cost')
    plt.title('Convergence Comparison: Standard vs Custom CPO')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def visualize_search_behavior(dimensions, bounds, standard_history, custom_history, objective_func):
    """
    Visualize the search behavior of both algorithms.
    """
    if dimensions != 2:
        print("Search behavior visualization only available for 2D problems.")
        return
    
    # Create animation of the optimization process
    print("\nCreating animations of the optimization process...")
    
    # Extract best positions from history for both algorithms
    std_best_pos_history = []
    custom_best_pos_history = []
    
    # For each frame, find the best position (lowest cost)
    for positions in standard_history:
        costs = np.array([objective_func(pos) for pos in positions])
        best_idx = np.argmin(costs)
        std_best_pos_history.append(positions[best_idx])
    
    for positions in custom_history:
        costs = np.array([objective_func(pos) for pos in positions])
        best_idx = np.argmin(costs)
        custom_best_pos_history.append(positions[best_idx])
    
    # Standard CPO animation
    std_anim = animate_optimization_2d(
        standard_history,
        objective_func,
        bounds,
        best_pos_history=std_best_pos_history,
        interval=100,
        contour_levels=20
    )
    
    # Custom CPO animation
    custom_anim = animate_optimization_2d(
        custom_history,
        objective_func,
        bounds,
        best_pos_history=custom_best_pos_history,
        interval=100,
        contour_levels=20
    )
    
    # Display the animations
    plt.show()


if __name__ == "__main__":
    # Define the problem
    dimensions = 2  # Use 2D for visualization
    bounds = get_function_bounds('rastrigin', dimensions)
    pop_size = 30
    max_iter = 100
    objective_func = rastrigin
    
    # Run both algorithms
    standard_results = run_standard_cpo(dimensions, bounds, pop_size, max_iter, objective_func)
    custom_results = run_custom_cpo(dimensions, bounds, pop_size, max_iter, objective_func)
    
    # Compare results
    compare_results(standard_results, custom_results)
    
    # Visualize search behavior
    visualize_search_behavior(
        dimensions, 
        bounds, 
        standard_results[3], 
        custom_results[3], 
        objective_func
    )
