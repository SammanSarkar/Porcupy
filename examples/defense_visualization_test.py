"""
Defense Mechanism Visualization Test

This script tests the fixed visualization functionality for defense mechanisms
in the Crested Porcupine Optimizer.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy.cpo_class import CPO
from porcupy.utils.visualization_manager import CPOVisualizer
from porcupy.functions import rastrigin

def test_defense_visualization():
    """Test the defense mechanism visualization functionality."""
    print("Testing defense mechanism visualization...")
    
    # Define a 2D test function
    def func(x):
        return rastrigin(x)
    
    # Define bounds for the search space
    dimensions = 2
    lb = -5.12 * np.ones(dimensions)
    ub = 5.12 * np.ones(dimensions)
    bounds = (lb, ub)
    
    # Initialize the CPO optimizer
    optimizer = CPO(
        dimensions=dimensions,
        bounds=bounds,
        pop_size=20,
        max_iter=30,
        cycles=3
    )
    
    # Run the optimization
    print("Running optimization...")
    best_pos, best_cost, _ = optimizer.optimize(
        objective_func=func,
        verbose=True,
        track_history=True  # Make sure to track history for visualization
    )
    
    print(f"Optimization completed. Best cost: {best_cost}")
    
    # Verify that defense types were recorded
    if hasattr(optimizer, 'defense_types_history') and optimizer.defense_types_history:
        print(f"Defense types history length: {len(optimizer.defense_types_history)}")
        
        # Initialize the visualizer
        visualizer = CPOVisualizer()
        
        # Record data from the optimizer
        visualizer.record_from_optimizer(optimizer)
        
        # Visualize defense mechanisms directly using the defense_types_history
        print("Creating defense mechanism visualization...")
        
        # Convert the list of defense types to a dictionary of counts for direct visualization
        defense_counts = {
            'sight': [],
            'sound': [],
            'odor': [],
            'physical': []
        }
        
        for defenses in optimizer.defense_types_history:
            defense_counts['sight'].append(defenses.count('sight'))
            defense_counts['sound'].append(defenses.count('sound'))
            defense_counts['odor'].append(defenses.count('odor'))
            defense_counts['physical'].append(defenses.count('physical'))
        
        # Use the visualizer to create the visualizations
        fig1 = visualizer.visualize_defense_mechanisms(
            title="Defense Mechanism Activation Test",
            save_path="defense_mechanisms_test.png"
        )
        
        # Visualize defense effectiveness
        print("Creating defense effectiveness visualization...")
        fig2 = visualizer.visualize_defense_effectiveness(
            title="Defense Mechanism Effectiveness Test",
            save_path="defense_effectiveness_test.png"
        )
        
        # Show the plots
        plt.show()
        
        print("Test completed successfully!")
        print("Visualization images saved as 'defense_mechanisms_test.png' and 'defense_effectiveness_test.png'")
    else:
        print("No defense types history recorded! Check the CPO implementation.")

if __name__ == "__main__":
    test_defense_visualization()
