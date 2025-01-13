"""Tests for pattern evolution system."""

import array
import time
import unittest
from typing import Dict, Optional

import numpy as np
import pytest

from ALPHA.core.memory.space import MemoryBlock, MemoryMetrics, MemoryOrganizer
from ALPHA.core.patterns.pattern_evolution import (
    BloomEnvironment,
    NaturalPattern,
    PatternEvolution,
    TimeWarp,
)


def generate_observation_sequence(base_interval: float = 0.1, steps: int = 8) -> np.ndarray:
    """Generate observation times using mathematical sequences.

    Creates a sequence combining:
    - Fibonacci for natural growth patterns
    - φ (golden ratio) for harmonic intervals
    - e for natural exponential growth
    - π for cyclic observation points
    """
    # Fibonacci sequence normalized to our base interval
    fib = np.array([base_interval * fibonacci(i) for i in range(steps)])

    # Golden ratio sequence
    phi = np.array([base_interval * (((1 + np.sqrt(5)) / 2) ** i) for i in range(steps)])

    # Natural exponential sequence
    exp = np.array([base_interval * (np.e ** (i / steps)) for i in range(steps)])

    # Pi-based cyclic sequence
    pi_seq = np.array([base_interval * (1 + np.sin(i * np.pi / steps)) for i in range(steps)])

    # Combine sequences with weights favoring natural growth
    sequence = 0.4 * fib + 0.3 * phi + 0.2 * exp + 0.1 * pi_seq
    return np.sort(sequence)


def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def analyze_harmonic_relationships(pattern: bytes, stage: str) -> Dict[str, float]:
    """Analyze mathematical harmonies in pattern structure."""
    pattern_array = np.frombuffer(pattern, dtype=np.uint8)

    # Calculate various mathematical relationships
    transitions = np.sum(pattern_array[1:] != pattern_array[:-1])
    density = np.mean(pattern_array) / 255
    entropy = -np.sum(
        (np.bincount(pattern_array) / len(pattern_array))
        * np.log2(np.bincount(pattern_array) / len(pattern_array) + 1e-10)
    )

    # Fourier analysis for frequency patterns
    fft = np.abs(np.fft.fft(pattern_array))
    dominant_freq = np.argmax(fft[1:]) + 1  # Skip DC component
    freq_ratio = dominant_freq / len(pattern_array)

    # Calculate harmonic relationships
    phi_ratio = transitions / len(pattern_array)
    pi_relation = (transitions / len(pattern_array)) / 0.318  # Normalized to π/10
    e_relation = entropy / np.e

    # Detect symmetry patterns
    symmetry = 1 - np.mean(np.abs(pattern_array - pattern_array[::-1]) / 255)

    return {
        "phi_ratio": phi_ratio,
        "phi_alignment": abs(1.618 - phi_ratio),
        "pi_relation": pi_relation,
        "e_relation": e_relation,
        "symmetry": symmetry,
        "freq_ratio": freq_ratio,
        "entropy": entropy / 8,  # Normalized to [0,1]
        "density": density,
    }


def visualize_pattern_state(
    pattern: bytes, metrics: MemoryMetrics, stage: str, harmonics: Optional[Dict[str, float]] = None
) -> None:
    """Visualize pattern state with detailed metrics and natural relationships."""
    pattern_array = np.frombuffer(pattern, dtype=np.uint8)

    # Pattern visualization with enhanced detail
    pattern_viz = "".join("█" if b > 127 else "░" for b in pattern_array[:32])

    print(f"\n{'=' * 60}")
    print(f"Pattern State: {stage}")
    print(f"{'=' * 60}")

    # Core pattern properties
    print("\nPattern Structure:")
    print(f"Binary Form:  {pattern_viz}")
    print(f"Density:      {'█' * int(harmonics['density'] * 32) if harmonics else ''}")
    print(f"Symmetry:     {'█' * int(harmonics['symmetry'] * 32) if harmonics else ''}")
    print(f"Entropy:      {'█' * int(harmonics['entropy'] * 32) if harmonics else ''}")

    # Experience and wonder metrics
    print("\nExperiential Properties:")
    print(
        f"Experience:   {'█' * int(metrics.experience_depth * 32):<32} {metrics.experience_depth:.3f}"
    )
    print(
        f"Wonder:       {'█' * int(metrics.wonder_potential * 32):<32} {metrics.wonder_potential:.3f}"
    )
    print(
        f"Resonance:    {'█' * int(metrics.resonance_stability * 32):<32} {metrics.resonance_stability:.3f}"
    )

    if harmonics:
        print("\nHarmonic Analysis:")
        print(f"φ ratio:      {harmonics['phi_ratio']:.3f}")
        print(f"φ alignment:  {harmonics['phi_alignment']:.3f} deviation")
        print(f"π relation:   {harmonics['pi_relation']:.3f}")
        print(f"e relation:   {harmonics['e_relation']:.3f}")
        print(f"Freq ratio:   {harmonics['freq_ratio']:.3f}")

    if metrics.variation_history:
        print("\nVariation Resonances:")
        for var_id, res in metrics.variation_history.items():
            print(f"  {var_id[-8:]}: {'█' * int(res * 32):<32} {res:.3f}")

    print(f"\n{'=' * 60}")


class TestPatternEvolution(unittest.TestCase):
    """Test cases for pattern evolution system."""

    def setUp(self):
        """Set up test environment."""
        self.evolution = PatternEvolution()

        # Create test pattern
        self.test_data = array.array(
            "B",
            [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,  # Gradual increase
                70,
                60,
                50,
                40,
                30,
                20,
                10,
                0,  # Gradual decrease (symmetry)
                0,
                0,
                255,
                255,
                0,
                0,
                255,
                255,  # Repeating pattern
            ],
        )

        # Create test context
        self.test_context = {
            "expected_role": "processor",
            "expected_behavior": {"regularity": 0.8, "symmetry": 0.7, "complexity": 0.5},
            "performance_targets": {"efficiency": 0.7, "reliability": 0.8, "accuracy": 0.9},
        }

    def test_pattern_metrics(self):
        """Test pattern metrics calculation."""
        metrics = self.evolution._calculate_pattern_metrics(self.test_data, self.test_context)

        # Verify all expected metrics are present
        expected_metrics = {
            "success_rate",
            "adaptation_rate",
            "improvement_rate",
            "stability",
            "effectiveness",
            "complexity",
        }
        self.assertEqual(set(metrics.keys()), expected_metrics)

        # Verify metric values are in valid range [0, 1]
        for metric, value in metrics.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_pattern_evolution(self):
        """Test pattern evolution over multiple iterations."""
        pattern_id = "test_pattern"

        # Simulate pattern evolution over multiple iterations
        for i in range(5):
            # Evolve pattern with slight modifications
            evolved_data = array.array(
                "B",
                np.clip(
                    np.frombuffer(self.test_data, dtype=np.uint8)
                    + np.random.normal(0, 10, len(self.test_data)),
                    0,
                    255,
                ).astype(np.uint8),
            )

            # Record evolution
            self.evolution.pattern_history[pattern_id] = []
            self.evolution.pattern_history[pattern_id].append(
                {
                    "pattern_data": evolved_data,
                    "metrics": self.evolution._calculate_pattern_metrics(
                        evolved_data, self.test_context
                    ),
                }
            )

        # Calculate adaptation rate
        adaptation_rate = self.evolution._calculate_adaptation_rate(pattern_id, self.test_data)

        # Verify adaptation rate is in valid range
        self.assertGreaterEqual(adaptation_rate, 0.0)
        self.assertLessEqual(adaptation_rate, 1.0)

    def test_pattern_analysis(self):
        """Test pattern analysis capabilities."""
        # Test entropy calculation
        entropy = self.evolution._calculate_entropy(np.frombuffer(self.test_data, dtype=np.uint8))
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)

        # Test symmetry calculation
        symmetry = self.evolution._calculate_symmetry(np.frombuffer(self.test_data, dtype=np.uint8))
        self.assertGreaterEqual(symmetry, 0.0)
        self.assertLessEqual(symmetry, 1.0)

        # Test consistency calculation
        consistency = self.evolution._calculate_consistency(
            np.frombuffer(self.test_data, dtype=np.uint8)
        )
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)

    def test_error_resistance(self):
        """Test error resistance capabilities."""
        # Test error resistance calculation
        resistance = self.evolution._calculate_error_resistance(
            np.frombuffer(self.test_data, dtype=np.uint8)
        )
        self.assertGreaterEqual(resistance, 0.0)
        self.assertLessEqual(resistance, 1.0)

        # Test correction potential
        potential = self.evolution._calculate_correction_potential(
            np.frombuffer(self.test_data, dtype=np.uint8)
        )
        self.assertGreaterEqual(potential, 0.0)
        self.assertLessEqual(potential, 1.0)

    def test_timewarp_evolution(self):
        """Test TimeWarp evolution and quantum state handling."""
        # Create test pattern with stable resonance
        pattern = NaturalPattern(
            name="test_pattern",
            confidence=0.8,
            ratio=1.618,  # Golden ratio
            resonance_frequency=0.5,
        )
        pattern.properties["stability"] = 0.7

        # Create test environment
        environment = BloomEnvironment()
        environment.environmental_rhythm = 0.3

        # Create TimeWarp instance
        time_warp = TimeWarp(base_frequency=1.0)

        # Simulate pattern evolution over time
        for _ in range(10):  # Multiple iterations to allow patterns to emerge
            time_warp.update_time_dilation(pattern, environment)

            # Verify quantum states
            self.assertTrue(len(time_warp.quantum_states) >= 3)  # At least base states
            self.assertTrue(all(0 <= prob <= 1 for _, prob in time_warp.quantum_states))
            self.assertAlmostEqual(sum(prob for _, prob in time_warp.quantum_states), 1.0, places=5)

            # Verify coherence
            self.assertTrue(0 <= time_warp.quantum_coherence <= 1)

            # Check temporal state
            self.assertIn("quantum_coherence", time_warp.temporal_state)
            self.assertIn("crystallization_count", time_warp.temporal_state)
            self.assertIn("resonance_stability", time_warp.temporal_state)

        # Verify crystallization points emerged naturally
        self.assertTrue(len(time_warp.crystallization_points) > 0)

        # Verify nodal points at crystallization
        self.assertTrue(len(time_warp.nodal_points) > 0)

        # Verify resonance memory
        self.assertTrue(any(len(hist) > 5 for hist in time_warp.resonance_memory.values()))

    def test_pattern_experience_and_evolution(self):
        """Test how patterns naturally gain experience and evolve."""
        block = MemoryBlock()

        # Create initial pattern from hardware state
        initial_data = np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
        ref = "test_pattern"

        print("\n[Step 1] Creating initial pattern from hardware state...")
        assert block.write(initial_data, ref)
        metrics = block.get_metrics(ref)
        harmonics = analyze_harmonic_relationships(initial_data, "Initial")
        visualize_pattern_state(initial_data, metrics, "Initial", harmonics)

        # Generate observation sequence
        observation_times = generate_observation_sequence(base_interval=0.1, steps=8)
        print(f"\nObservation sequence (seconds): {observation_times}")

        # Let pattern gain experience through multiple interactions
        print("\n[Step 2] Pattern gaining experience through interactions...")
        for i, wait_time in enumerate(observation_times):
            time.sleep(wait_time)  # Wait for harmonic interval
            metrics.update_experience(metrics.resonance_stability, np.random.random())
            harmonics = analyze_harmonic_relationships(initial_data, f"Experience {i+1}")
            visualize_pattern_state(initial_data, metrics, f"Experience {i+1}", harmonics)

        # Check if pattern can dream
        print("\n[Step 3] Checking pattern's ability to dream...")
        assert metrics.can_dream(), "Pattern should have gained enough experience to dream"

        # Generate and observe variations
        print("\n[Step 4] Observing pattern variations...")
        variations = block.dream_variations(ref)
        for i, var in enumerate(variations):
            var_metrics = MemoryMetrics()
            var_metrics.update_experience(metrics.resonance_stability, np.random.random())
            harmonics = analyze_harmonic_relationships(var, f"Variation {i+1}")
            visualize_pattern_state(var, var_metrics, f"Variation {i+1}", harmonics)

        # Analyze overall evolution
        print("\n[Step 5] Analyzing pattern evolution...")
        all_harmonics = [
            analyze_harmonic_relationships(var, f"Variation {i}")
            for i, var in enumerate(variations)
        ]

        print("\nEvolution Summary:")
        print(f"Total Variations: {len(variations)}")
        print(f"Mean φ ratio: {np.mean([h['phi_ratio'] for h in all_harmonics]):.3f}")
        print(f"φ stability: {np.std([h['phi_ratio'] for h in all_harmonics]):.3f}")
        print(f"Entropy evolution: {np.mean([h['entropy'] for h in all_harmonics]):.3f}")
        print(f"Symmetry preservation: {np.mean([h['symmetry'] for h in all_harmonics]):.3f}")

        # Verify natural properties are preserved
        for var in variations:
            harmonics = analyze_harmonic_relationships(var, "Verification")
            assert (
                abs(harmonics["phi_ratio"] - metrics.phi_ratio) < 0.1
            ), "Variations should maintain mathematical harmony"


if __name__ == "__main__":
    unittest.main()
