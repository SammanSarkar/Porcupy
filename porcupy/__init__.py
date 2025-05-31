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

# Try to import GPU-accelerated version if CuPy is available
try:
    from .gpu_cpo import GPUCPO, gpu_cpo
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    # Create dummy classes to avoid import errors
    class GPUCPO:
        """Dummy class when GPU acceleration is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GPU acceleration requires CuPy. "
                "Install with: pip install cupy-cuda11x"
            )
    
    def gpu_cpo(*args, **kwargs):
        """Dummy function when GPU acceleration is not available."""
        raise ImportError(
            "GPU acceleration requires CuPy. "
            "Install with: pip install cupy-cuda11x"
        )

__version__ = "0.1.1"  # Updated for GPU support
__author__ = "Samman Sarkar"
__all__ = [
    'CPO', 'GPUCPO', 'Optimizer', 'PorcupinePopulation', 'DefenseMechanisms',
    'PopulationManager', 'PopulationCycle', 'SelectionStrategies', 'cpo', 'gpu_cpo',
    # Functions
    'sphere', 'rosenbrock', 'schwefel_2_22', 'schwefel_1_2', 'schwefel_2_21',
    'step', 'quartic', 'rastrigin', 'ackley', 'griewank', 'schwefel', 'michalewicz',
    # Function utilities
    'get_function_by_name', 'get_function_bounds', 'get_function_optimum',
    # Constants
    'GPU_AVAILABLE'
]