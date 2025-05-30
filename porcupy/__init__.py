from .cpo import cpo
from .cpo_class import CPO
from .base import Optimizer
from .porcupines import PorcupinePopulation, DefenseMechanisms, PopulationManager
from .utils.population import PopulationCycle, SelectionStrategies
from .functions import (
    # Unimodal functions
    sphere, rosenbrock, schwefel_2_22, schwefel_1_2, schwefel_2_21, step, quartic,
    # Multimodal functions
    rastrigin, ackley, griewank, schwefel, michalewicz,
    # Function utilities
    get_function_by_name, get_function_bounds, get_function_optimum
)

__version__ = "0.1.0"
__author__ = "Samman Sarkar"