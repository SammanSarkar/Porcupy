import numpy as np
from porcupy.cpo import cpo
from porcupy.functions import sphere, rosenbrock
from porcupy.utils.plotting import plot_convergence

def run_unconstrained_example():
    """
    Demonstrate CPO on the unconstrained sphere function.
    """
    print("Running unconstrained CPO example (sphere function)")
    lb = [-5, -5]
    ub = [5, 5]
    best_pos, best_cost, cost_history = cpo(
        sphere,
        lb=lb,
        ub=ub,
        pop_size=30,
        max_iter=100,
        verbose=True
    )
    print(f"Best Position: {best_pos}")
    print(f"Best Cost: {best_cost}")
    plot_convergence(cost_history, title="Sphere Function Convergence")

def run_constrained_example():
    """
    Demonstrate CPO with constraints on the Rosenbrock function.
    """
    print("\nRunning constrained CPO example (Rosenbrock function)")
    def constraint(x):
        # Constraint: x[0] + x[1] >= 0
        return np.array([x[0] + x[1]])
    
    lb = [-2, -2]
    ub = [2, 2]
    best_pos, best_cost, cost_history = cpo(
        rosenbrock,
        lb=lb,
        ub=ub,
        pop_size=30,
        max_iter=100,
        f_ieqcons=constraint,
        verbose=True
    )
    print(f"Best Position: {best_pos}")
    print(f"Best Cost: {best_cost}")
    print(f"Constraint Satisfied: {np.all(constraint(best_pos) >= 0)}")
    plot_convergence(cost_history, title="Rosenbrock Function Convergence")

if __name__ == "__main__":
    run_unconstrained_example()
    run_constrained_example()