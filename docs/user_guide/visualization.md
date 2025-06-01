# Visualization Guide

This guide covers the visualization capabilities of Porcupy for analyzing and understanding optimization results.

## Basic Plotting

### Convergence Plot

Plot the optimization progress to visualize how the best fitness improves over iterations:

```python
import matplotlib.pyplot as plt
from porcupy import CPO
from porcupy.functions import ackley

# Run optimization
optimizer = CPO(dimensions=2, bounds=([-5, -5], [5, 5]), max_iter=50)
best_solution, best_fitness, history = optimizer.optimize(ackley)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(history['best_fitness'], 'b-', linewidth=2)
plt.title('Optimization Convergence')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()
```

### Population Distribution

Visualize how the population evolves over time:

```python
import numpy as np

def plot_population(positions, iteration, ax):
    ax.clear()
    ax.scatter(positions[:, 0], positions[:, 1], alpha=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(f'Population Distribution (Iteration {iteration})')
    ax.grid(True)

# Create animation of population evolution
fig, ax = plt.subplots(figsize=(8, 8))
for i, pop in enumerate(history['population'][::5]):  # Plot every 5th iteration
    plot_population(pop, i*5, ax)
    plt.pause(0.1)
```

## Advanced Visualization

### 3D Surface Plot

For 2D optimization problems, you can visualize the objective function surface:

```python
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_surface(func, bounds, points=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid
    x = np.linspace(bounds[0][0], bounds[1][0], 100)
    y = np.linspace(bounds[0][1], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Plot points if provided
    if points is not None:
        z = np.array([func(p) for p in points])
        ax.scatter(points[:, 0], points[:, 1], z, c='red', s=50)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    plt.colorbar(surf)
    plt.show()

# Example usage
bounds = ([-5, -5], [5, 5])
plot_3d_surface(ackley, bounds, history['population'][-1])
```

### Heatmap of Search Space

Visualize where the population is exploring in the search space:

```python
def plot_search_heatmap(history, bounds, n_bins=20):
    # Combine all positions from all iterations
    all_positions = np.vstack(history['population'])
    
    # Create 2D histogram
    plt.figure(figsize=(10, 8))
    plt.hist2d(all_positions[:, 0], all_positions[:, 1], 
               bins=n_bins, range=[bounds[0], bounds[1]], 
               cmap='viridis')
    plt.colorbar(label='Visit Count')
    plt.title('Search Space Exploration')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

plot_search_heatmap(history, bounds=([-5, -5], [5, 5]))
```

## Interactive Visualization

### Real-time Optimization Dashboard

Create an interactive dashboard to monitor optimization progress:

```python
import ipywidgets as widgets
from IPython.display import display, clear_output

def interactive_optimization():
    # Set up widgets
    start_button = widgets.Button(description="Start Optimization")
    progress = widgets.FloatProgress(value=0, max=100, description='Progress:')
    output = widgets.Output()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    def on_button_clicked(b):
        with output:
            clear_output(wait=True)
            
            # Initialize optimizer
            optimizer = CPO(dimensions=2, bounds=([-5, -5], [5, 5]), max_iter=100)
            
            # Run optimization with callback
            def callback(iteration, fitness, positions):
                progress.value = iteration
                
                # Update plots
                ax1.clear()
                ax1.plot(history['best_fitness'][:iteration+1])
                ax1.set_title('Convergence')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Best Fitness')
                
                ax2.clear()
                ax2.scatter(positions[:, 0], positions[:, 1], alpha=0.5)
                ax2.set_xlim(-5, 5)
                ax2.set_ylim(-5, 5)
                ax2.set_title(f'Population (Iteration {iteration})')
                
                display(fig)
                clear_output(wait=True)
            
            # Store history for callback
            optimizer.history = {'best_fitness': []}
            
            # Run optimization
            best_solution, best_fitness, _ = optimizer.optimize(
                ackley,
                callback=callback
            )
    
    # Display widgets
    display(start_button, progress, output)
    start_button.on_click(on_button_clicked)

# Run in Jupyter notebook
# interactive_optimization()
```

## Custom Visualization

### Parallel Coordinates Plot

Visualize high-dimensional solutions using parallel coordinates:

```python
def parallel_coordinates(solutions, fitness_values, n_best=10):
    # Sort solutions by fitness
    idx = np.argsort(fitness_values)
    best_solutions = solutions[idx[:n_best]]
    
    # Create parallel coordinates plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each solution
    for i in range(min(n_best, len(best_solutions))):
        ax.plot(best_solutions[i], 'o-', alpha=0.7, label=f'Rank {i+1}')
    
    ax.set_xticks(range(len(best_solutions[0])))
    ax.set_xticklabels([f'Dim {i+1}' for i in range(len(best_solutions[0]))])
    ax.set_ylabel('Value')
    ax.set_title('Best Solutions (Parallel Coordinates)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

# Example usage
parallel_coordinates(history['population'][-1], history['best_fitness'][-1])
```

## Exporting Visualizations

Save your visualizations for reports or presentations:

```python
# Save figure in various formats
plt.figure(figsize=(10, 6))
plt.plot(history['best_fitness'])
plt.title('Optimization Convergence')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')

# Save as high-resolution PNG
plt.savefig('convergence.png', dpi=300, bbox_inches='tight')

# Save as vector graphic (for publications)
plt.savefig('convergence.pdf', format='pdf', bbox_inches='tight')

# Save as interactive HTML (requires plotly)
# fig = px.line(y=history['best_fitness'])
# fig.write_html('convergence.html')
```

## Tips for Effective Visualization

1. **Choose Appropriate Scales**
   - Use log scale for fitness values spanning multiple orders of magnitude
   - Set appropriate axis limits to focus on relevant regions

2. **Highlight Important Information**
   - Mark the best solution found
   - Annotate significant events (e.g., restarts, convergence)

3. **Compare Multiple Runs**
   - Overlay convergence plots from different configurations
   - Use different colors/markers for clarity

4. **Interactive Exploration**
   - Use tools like Plotly or Bokeh for interactive visualizations
   - Enable zooming and panning for detailed inspection

5. **Performance Considerations**
   - Downsample data for large histories
   - Use vector graphics for publications
   - Optimize rendering for real-time visualization
