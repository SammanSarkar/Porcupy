"""
Basic import tests for visualization modules.

This module tests that all visualization modules can be imported correctly.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_enhanced_visualization_imports():
    """Test that enhanced_visualization module can be imported."""
    from porcupy.utils.enhanced_visualization import (
        plot_defense_mechanisms,
        plot_population_cycles,
        plot_diversity_history,
        plot_2d_porcupines,
        animate_porcupines_2d,
        plot_3d_porcupines,
        calculate_diversity,
        track_defense_mechanisms
    )
    assert callable(plot_defense_mechanisms)
    assert callable(plot_population_cycles)
    assert callable(plot_diversity_history)
    assert callable(plot_2d_porcupines)
    assert callable(animate_porcupines_2d)
    assert callable(plot_3d_porcupines)
    assert callable(calculate_diversity)
    assert callable(track_defense_mechanisms)


def test_defense_visualization_imports():
    """Test that defense_visualization module can be imported."""
    from porcupy.utils.defense_visualization import (
        visualize_defense_territories,
        visualize_defense_mechanisms,
        animate_defense_mechanisms,
        plot_defense_effectiveness,
        visualize_quill_directions
    )
    assert callable(visualize_defense_territories)
    assert callable(visualize_defense_mechanisms)
    assert callable(animate_defense_mechanisms)
    assert callable(plot_defense_effectiveness)
    assert callable(visualize_quill_directions)


def test_population_visualization_imports():
    """Test that population_visualization module can be imported."""
    from porcupy.utils.population_visualization import (
        plot_population_reduction_strategies,
        plot_population_diversity_map,
        animate_population_cycle,
        plot_exploration_exploitation_balance,
        plot_diversity_vs_convergence,
        calculate_diversity
    )
    assert callable(plot_population_reduction_strategies)
    assert callable(plot_population_diversity_map)
    assert callable(animate_population_cycle)
    assert callable(plot_exploration_exploitation_balance)
    assert callable(plot_diversity_vs_convergence)
    assert callable(calculate_diversity)


def test_interactive_visualization_imports():
    """Test that interactive_visualization module can be imported."""
    from porcupy.utils.interactive_visualization import (
        OptimizationDashboard,
        ParameterTuningDashboard,
        create_interactive_optimization_plot
    )
    assert OptimizationDashboard is not None
    assert ParameterTuningDashboard is not None
    assert callable(create_interactive_optimization_plot)


def test_visualization_manager_imports():
    """Test that visualization_manager module can be imported."""
    from porcupy.utils.visualization_manager import CPOVisualizer
    assert CPOVisualizer is not None
