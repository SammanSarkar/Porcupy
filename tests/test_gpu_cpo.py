"""
Tests for GPU-accelerated CPO implementation.

This module contains tests for the GPU-accelerated Crested Porcupine Optimizer.
It includes unit tests, integration tests, and performance tests.
"""

import numpy as np
import pytest
import time
from porcupy import CPO
from porcupy.functions import sphere, rosenbrock, rastrigin

# Skip all tests if GPU is not available
pytest.importorskip("cupy")

class TestGPUCPO:
    """Test cases for GPU-accelerated CPO."""
    
    def test_import_and_initialization(self):
        """Test that GPUCPO can be imported and initialized."""
        from porcupy.gpu_cpo import GPUCPO
        
        # Test basic initialization
        optimizer = GPUCPO(
            dimensions=2,
            bounds=([-5, -5], [5, 5]),
            pop_size=10,
            max_iter=5
        )
        assert optimizer is not None
        assert hasattr(optimizer, 'optimize')
    
    def test_basic_optimization(self):
        """Test basic GPU optimization on a simple function."""
        from porcupy.gpu_cpo import GPUCPO
        
        optimizer = GPUCPO(
            dimensions=2,
            bounds=([-5, -5], [5, 5]),
            pop_size=20,
            max_iter=10
        )
        
        best_pos, best_cost, _ = optimizer.optimize(sphere)
        
        # Verify solution is reasonable
        assert len(best_pos) == 2
        assert np.all(np.abs(best_pos) < 1.0)  # Should be close to [0, 0]
        assert best_cost < 1.0  # Should be close to 0


class TestGPUCPOIntegration:
    """Integration tests for GPU-accelerated CPO."""
    
    def test_cpu_gpu_consistency(self):
        """Test that GPU and CPU versions produce similar results."""
        from porcupy.gpu_cpo import GPUCPO
        
        # Test parameters
        dimensions = 3
        pop_size = 50
        max_iter = 20
        bounds = ([-5] * dimensions, [5] * dimensions)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run CPU version
        cpu_optimizer = CPO(
            dimensions=dimensions,
            bounds=bounds,
            pop_size=pop_size,
            max_iter=max_iter
        )
        cpu_pos, cpu_cost, _ = cpu_optimizer.optimize(rosenbrock)
        
        # Set same random seed
        np.random.seed(42)
        
        # Run GPU version
        gpu_optimizer = GPUCPO(
            dimensions=dimensions,
            bounds=bounds,
            pop_size=pop_size,
            max_iter=max_iter
        )
        gpu_pos, gpu_cost, _ = gpu_optimizer.optimize(rosenbrock)
        
        # Check that results are similar (within 5% relative tolerance)
        assert np.allclose(cpu_pos, gpu_pos, rtol=0.05)
        assert np.isclose(cpu_cost, gpu_cost, rtol=0.05)
    
    def test_large_scale_optimization(self):
        """Test GPU performance with larger problem size."""
        from porcupy.gpu_cpo import GPUCPO
        
        # Larger problem size
        dimensions = 50
        pop_size = 500
        max_iter = 5  # Keep iterations low for test speed
        bounds = ([-5.12] * dimensions, [5.12] * dimensions)
        
        optimizer = GPUCPO(
            dimensions=dimensions,
            bounds=bounds,
            pop_size=pop_size,
            max_iter=max_iter
        )
        
        # Time the optimization
        start_time = time.time()
        best_pos, best_cost, _ = optimizer.optimize(rastrigin)
        gpu_time = time.time() - start_time
        
        # Basic validation
        assert len(best_pos) == dimensions
        assert best_cost < 100.0  # Should find a reasonable solution
        
        # Just log the time, don't fail the test based on performance
        print(f"\nGPU Optimization Time: {gpu_time:.2f} seconds")
        print(f"Best Cost: {best_cost}")


class TestGPUResourceManagement:
    """Tests for GPU resource handling and cleanup."""
    
    def test_memory_cleanup(self):
        """Test that GPU memory is properly cleaned up."""
        import cupy as cp
        from porcupy.gpu_cpo import GPUCPO
        
        # Get initial memory usage
        mempool = cp.get_default_memory_pool()
        initial_mem = mempool.used_bytes()
        
        # Create and run optimizer in a separate scope
        def run_optimization():
            optimizer = GPUCPO(
                dimensions=5,
                bounds=([-5]*5, [5]*5),
                pop_size=100,
                max_iter=5
            )
            optimizer.optimize(sphere)
            return None  # Explicitly return None to ensure optimizer is deleted
        
        run_optimization()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory was freed (within 10MB of initial)
        final_mem = mempool.used_bytes()
        mem_diff = abs(final_mem - initial_mem) / (1024 * 1024)  # Convert to MB
        assert mem_diff < 10, f"Memory not properly freed. Difference: {mem_diff:.2f}MB"
