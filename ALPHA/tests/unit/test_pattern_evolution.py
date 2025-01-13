"""Tests for pattern evolution system."""

import array
import time
from typing import Dict, List, Optional, Union

import numpy as np
import psutil
import pytest
from numpy.typing import NDArray

from ALPHA.core.hardware.pulse import Pulse
from ALPHA.core.memory.space import MemoryBlock, MemoryMetrics
from ALPHA.core.patterns.pattern_evolution import (
    BloomEnvironment,
    KymaState,
    NaturalPattern,
    PatternEvolution,
    TimeWarp,
)


def generate_observation_sequence(
    base_interval: float = 0.1, steps: int = 8
) -> NDArray[np.float64]:
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


def analyze_harmonic_relationships(
    pattern: Union[bytes, NDArray[np.uint8]], stage: str
) -> Dict[str, float]:
    """Analyze mathematical harmonies in pattern structure."""
    if isinstance(pattern, bytes):
        pattern_array = np.frombuffer(pattern, dtype=np.uint8)
    else:
        pattern_array = pattern

    # Calculate various mathematical relationships
    transitions: int = int(np.sum(pattern_array[1:] != pattern_array[:-1]))
    density: float = float(np.mean(pattern_array) / 255)

    # Calculate entropy with proper type handling
    counts = np.bincount(pattern_array)
    probs = counts / len(pattern_array)
    entropy: float = float(-np.sum(probs * np.log2(probs + 1e-10)))

    # Fourier analysis for frequency patterns
    fft = np.abs(np.fft.fft(pattern_array))
    dominant_freq = np.argmax(fft[1:]) + 1  # Skip DC component
    freq_ratio: float = float(dominant_freq / len(pattern_array))

    # Calculate harmonic relationships
    phi_ratio: float = float(transitions / len(pattern_array))
    pi_relation: float = float((transitions / len(pattern_array)) / 0.318)  # π/10
    e_relation: float = float(entropy / np.e)

    # Detect symmetry patterns
    symmetry: float = float(1 - np.mean(np.abs(pattern_array - pattern_array[::-1]) / 255))

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
    pattern: Union[bytes, NDArray[np.uint8]],
    metrics: MemoryMetrics,
    stage: str,
    harmonics: Optional[Dict[str, float]] = None,
) -> None:
    """Visualize pattern state with detailed metrics and natural relationships."""
    if isinstance(pattern, bytes):
        pattern_array = np.frombuffer(pattern, dtype=np.uint8)
    else:
        pattern_array = pattern

    # Pattern visualization with enhanced detail
    pattern_viz = "".join("█" if b > 127 else "░" for b in pattern_array[:32])

    print(f"\n{'=' * 60}")
    print(f"Pattern State: {stage}")
    print(f"{'=' * 60}")

    # Core pattern properties
    print("\nPattern Structure:")
    print(f"Binary Form:  {pattern_viz}")
    if harmonics:
        print(f"Density:      {'█' * int(harmonics['density'] * 32)}")
        print(f"Symmetry:     {'█' * int(harmonics['symmetry'] * 32)}")
        print(f"Entropy:      {'█' * int(harmonics['entropy'] * 32)}")

    # Experience and wonder metrics
    print("\nExperiential Properties:")
    print(
        "Experience:   "
        f"{'█' * int(metrics.experience_depth * 32):<32} "
        f"{metrics.experience_depth:.3f}"
    )
    print(
        "Wonder:       "
        f"{'█' * int(metrics.wonder_potential * 32):<32} "
        f"{metrics.wonder_potential:.3f}"
    )
    print(
        "Resonance:    "
        f"{'█' * int(metrics.resonance_stability * 32):<32} "
        f"{metrics.resonance_stability:.3f}"
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
            print(f"  {var_id[-8:]}: " f"{'█' * int(res * 32):<32} {res:.3f}")

    print(f"\n{'=' * 60}")


@pytest.fixture
def evolution() -> PatternEvolution:
    """Create pattern evolution instance for testing."""
    return PatternEvolution()


@pytest.fixture
def test_data() -> array.array:
    """Create test pattern data."""
    return array.array(
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
            0,  # Gradual decrease
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


@pytest.fixture
def test_context() -> Dict[str, Union[str, Dict[str, float]]]:
    """Create test context with expected behaviors."""
    return {
        "expected_role": "processor",
        "expected_behavior": {
            "regularity": 0.8,
            "symmetry": 0.7,
            "complexity": 0.5,
        },
        "performance_targets": {
            "efficiency": 0.7,
            "reliability": 0.8,
            "accuracy": 0.9,
        },
    }


def test_pattern_metrics(evolution, test_data, test_context):
    """Test pattern metrics calculation."""
    metrics = evolution._calculate_pattern_metrics(test_data, test_context)

    # Verify all expected metrics are present
    expected_metrics = {
        "success_rate",
        "adaptation_rate",
        "improvement_rate",
        "stability",
        "effectiveness",
        "complexity",
    }
    assert set(metrics.keys()) == expected_metrics

    # Verify metric values are in valid range [0, 1]
    for metric, value in metrics.items():
        assert 0.0 <= value <= 1.0


def test_pattern_evolution(evolution, test_data, test_context):
    """Test pattern evolution over multiple iterations."""
    pattern_id = "test_pattern"

    # Simulate pattern evolution over multiple iterations
    for i in range(5):
        # Evolve pattern with slight modifications
        evolved_data = array.array(
            "B",
            np.clip(
                np.frombuffer(test_data, dtype=np.uint8) + np.random.normal(0, 10, len(test_data)),
                0,
                255,
            ).astype(np.uint8),
        )

        # Calculate metrics for evolved pattern
        metrics = evolution._calculate_pattern_metrics(evolved_data, test_context)

        # Record evolution in state
        if pattern_id not in evolution.states:
            evolution.states[pattern_id] = evolution.EvolutionState(pattern_id)

        state = evolution.states[pattern_id]
        state.adaptation_history.append(metrics["adaptation_rate"])

    # Calculate adaptation rate
    adaptation_rate = evolution._calculate_adaptation_rate(pattern_id, test_data)

    # Verify adaptation rate is in valid range
    assert 0.0 <= adaptation_rate <= 1.0


def test_pattern_analysis(evolution, test_data):
    """Test pattern analysis capabilities."""
    # Test entropy calculation
    entropy = evolution._calculate_entropy(np.frombuffer(test_data, dtype=np.uint8))
    assert 0.0 <= entropy <= 1.0

    # Test symmetry calculation
    symmetry = evolution._calculate_symmetry(np.frombuffer(test_data, dtype=np.uint8))
    assert 0.0 <= symmetry <= 1.0

    # Test consistency calculation
    consistency = evolution._calculate_consistency(np.frombuffer(test_data, dtype=np.uint8))
    assert 0.0 <= consistency <= 1.0


def test_error_resistance(evolution, test_data):
    """Test error resistance capabilities."""
    # Test error resistance calculation
    resistance = evolution._calculate_error_resistance(np.frombuffer(test_data, dtype=np.uint8))
    assert 0.0 <= resistance <= 1.0


@pytest.mark.integration
def test_timewarp_evolution():
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
        assert len(time_warp.quantum_states) >= 3  # At least base states
        assert all(0 <= prob <= 1 for _, prob in time_warp.quantum_states)
        assert abs(sum(prob for _, prob in time_warp.quantum_states) - 1.0) < 1e-5

        # Verify coherence
        assert 0 <= time_warp.quantum_coherence <= 1

        # Check temporal state
        assert "quantum_coherence" in time_warp.temporal_state
        assert "crystallization_count" in time_warp.temporal_state
        assert "resonance_stability" in time_warp.temporal_state

    # Verify crystallization points emerged naturally
    assert len(time_warp.crystallization_points) > 0

    # Verify nodal points at crystallization
    assert len(time_warp.nodal_points) > 0

    # Verify resonance memory
    assert any(len(hist) > 5 for hist in time_warp.resonance_memory.values())


@pytest.fixture
def memory_block():
    """Create memory block for testing."""
    return MemoryBlock()


@pytest.fixture
def pulse():
    """Create pulse observer for natural patterns."""
    return Pulse()


@pytest.fixture
def kyma_state():
    """Create wave communication state."""
    return KymaState()


@pytest.fixture
def time_warp():
    """Create temporal experience handler."""
    return TimeWarp()


@pytest.mark.integration
def test_pattern_experience_and_evolution(memory_block, pulse, kyma_state, time_warp):
    """Test how patterns naturally gain experience and evolve."""
    # Create initial pattern from hardware state
    print("\n[Step 1] Creating initial pattern from hardware state...")
    pattern_data = []
    for _ in range(32):  # Collect 32 samples
        state = pulse.sense()
        if state is not None:
            pattern_data.append(state)
        time.sleep(0.01)  # Natural timing between observations

    initial_data = np.array(pattern_data, dtype=np.uint8)
    ref = "test_pattern"

    # Write pattern to memory with proper type handling
    if isinstance(initial_data, bytes):
        data_array = np.frombuffer(initial_data, dtype=np.uint8)
    else:
        data_array = np.array(initial_data, dtype=np.uint8)

    assert memory_block.write(ref, data_array)
    metrics = memory_block.get_metrics(ref)
    harmonics = analyze_harmonic_relationships(data_array.tobytes(), "Initial")
    visualize_pattern_state(data_array.tobytes(), metrics, "Initial", harmonics)

    # Generate observation sequence aligned with natural timing
    observation_times = generate_observation_sequence(base_interval=0.1, steps=8)
    print(f"\nObservation sequence (seconds): {observation_times}")

    # Let pattern gain experience through multiple interactions
    print("\n[Step 2] Pattern gaining experience through interactions...")
    for i, wait_time in enumerate(observation_times):
        time.sleep(wait_time)  # Wait for harmonic interval

        # Use actual hardware state for experience
        hardware_state = psutil.cpu_percent() / 100.0
        metrics.update_experience(metrics.resonance_stability, hardware_state)

        # Record current system state
        current_state = pulse.sense()
        if current_state is not None:
            pattern_data.append(current_state)

        harmonics = analyze_harmonic_relationships(data_array.tobytes(), f"Experience {i+1}")
        visualize_pattern_state(data_array.tobytes(), metrics, f"Experience {i+1}", harmonics)

    # Check if pattern can dream
    print("\n[Step 3] Checking pattern's ability to dream...")
    assert metrics.can_dream(), "Pattern should have gained enough experience to dream"

    # Generate and observe variations with temporal integration
    print("\n[Step 4] Observing pattern variations...")
    variations = memory_block.dream_variations(ref)

    for i, var in enumerate(variations):
        var_metrics = MemoryMetrics()

        # Use hardware state for experience
        hardware_state = psutil.cpu_percent() / 100.0
        var_metrics.update_experience(metrics.resonance_stability, hardware_state)

        # Create spatial pattern from variation
        var_data = np.frombuffer(var, dtype=np.uint8)
        spatial_pattern = {
            "resonance": np.mean(np.diff(var_data)),
            "stability": var_metrics.resonance_stability,
            "complexity": len(np.unique(var_data)) / len(var_data),
            "harmony": 1.0 - np.std(var_data) / 128.0,  # Normalized to [0,1]
        }

        # Integrate with KymaState
        kyma_state.integrate_memory_space(spatial_pattern, var_metrics)

        # Update temporal experience
        time_warp.update_time_dilation(
            NaturalPattern(
                name=f"variation_{i}",
                confidence=var_metrics.resonance_stability,
                ratio=spatial_pattern["harmony"],
                resonance_frequency=spatial_pattern["resonance"],
            ),
            BloomEnvironment(environmental_rhythm=hardware_state),
        )

        harmonics = analyze_harmonic_relationships(var, f"Variation {i+1}")
        visualize_pattern_state(var, var_metrics, f"Variation {i+1}", harmonics)

    # Analyze overall evolution
    print("\n[Step 5] Analyzing pattern evolution...")
    all_harmonics = [
        analyze_harmonic_relationships(var, f"Variation {i}") for i, var in enumerate(variations)
    ]

    print("\nEvolution Summary:")
    print(f"Total Variations: {len(variations)}")
    print(f"Mean φ ratio: {np.mean([h['phi_ratio'] for h in all_harmonics]):.3f}")
    print(f"φ stability: {np.std([h['phi_ratio'] for h in all_harmonics]):.3f}")
    print(f"Entropy evolution: {np.mean([h['entropy'] for h in all_harmonics]):.3f}")
    print(f"Symmetry preservation: {np.mean([h['symmetry'] for h in all_harmonics]):.3f}")

    # Temporal-Spatial Integration Metrics
    print("\nTemporal-Spatial Integration:")
    print(f"Quantum Coherence: {time_warp.quantum_coherence:.3f}")
    print(f"Crystallization Points: {len(time_warp.crystallization_points)}")
    print(f"Standing Waves: {len(time_warp.standing_waves)}")
    print(f"Resonance Channels: {len(kyma_state.resonance_channels)}")

    # Verify temporal-spatial integration
    assert time_warp.quantum_coherence >= 0.5, "Should maintain quantum coherence"
    assert len(time_warp.crystallization_points) > 0, "Should form crystallization points"
    assert len(time_warp.standing_waves) > 0, "Should establish standing waves"
    assert len(kyma_state.resonance_channels) > 0, "Should create resonance channels"

    # Verify natural properties are preserved
    for var in variations:
        var_data = np.frombuffer(var, dtype=np.uint8)
        harmonics = analyze_harmonic_relationships(var, "Verification")
        assert (
            abs(harmonics["phi_ratio"] - metrics.phi_ratio) < 0.1
        ), "Variations should maintain mathematical harmony"
