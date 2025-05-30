"""
Simple Defense Mechanism Visualization Test
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from porcupy.cpo_class import CPO
from porcupy.utils.enhanced_visualization import plot_defense_mechanisms
from porcupy.functions import rastrigin

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
    max_iter=20,
    cycles=2
)

# Run the optimization
print("Running optimization...")
best_pos, best_cost, _ = optimizer.optimize(
    objective_func=func,
    verbose=True,
    track_history=True
)

print(f"Optimization completed. Best cost: {best_cost}")

# Verify that defense types were recorded
if hasattr(optimizer, 'defense_types_history') and optimizer.defense_types_history:
    print(f"Defense types history length: {len(optimizer.defense_types_history)}")
    
    # Convert the list of defense types to a dictionary of counts
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
    
    # Plot using the direct function
    fig = plot_defense_mechanisms(
        defense_history=defense_counts,
        title="Defense Mechanism Visualization",
        save_path="defense_mechanisms.png"
    )
    
    plt.show()
else:
    print("No defense types history recorded!")
