import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(cost_history, title="Convergence Curve", save_path=None):
    """
    Plot the convergence curve of the optimization process.

    Parameters
    ----------
    cost_history : ndarray
        Array of best fitness values over iterations.
    title : str, optional
        Title of the plot (default: "Convergence Curve").
    save_path : str, optional
        Path to save the plot (e.g., "convergence.png"). If None, displays the plot.

    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(np.arange(1, len(cost_history) + 1), cost_history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()