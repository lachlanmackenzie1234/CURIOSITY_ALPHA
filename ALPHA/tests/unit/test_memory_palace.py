"""Tests for Memory Palace pattern organization and natural emergence."""

import random
import time
from typing import List, Optional

import numpy as np
import pytest

from ...core.memory.memory import Space, SpaceType
from ...core.memory.space import MemoryBlock
from ...core.patterns.binary_pulse import Pulse
from ...core.patterns.pattern import Pattern
from ...core.patterns.pattern_evolution import (
    BloomEvent,
    KymaState,
    NaturalPattern,
    PatternEvolution,
)


def create_pattern_from_pulse(id: str, duration: float = 0.5) -> Pattern:
    """Create pattern from actual system binary pulse."""
    pattern = Pattern(id)
    pulse = Pulse()

    # Natural timing variation
    sample_count = 0
    max_samples = 50  # Collect up to 50 samples

    # Collect binary data with natural timing
    start_time = time.time()
    while time.time() - start_time < duration and sample_count < max_samples:
        # Let system find its natural rhythm
        pulse.sense()
        if pulse.last_change:
            pattern.data.append(1)
        else:
            pattern.data.append(0)
        sample_count += 1

        # Allow natural pause between observations
        natural_pause = random.uniform(0.001, 0.02)  # 1-20ms natural variation
        time.sleep(natural_pause)

    return pattern


@pytest.mark.integration
def test_natural_pattern_association() -> None:
    """Observe how patterns naturally form associations and relationships."""
    # Create a space for observation
    space = Space(type=SpaceType.MEDITATION)

    # Create patterns from actual system states
    print("\nGenerating patterns from system states...")
    print("-" * 40)

    pattern1 = create_pattern_from_pulse("pattern1")
    print("Pattern 1 binary:", "".join(str(b) for b in pattern1.data))

    # Brief pause to ensure different system states
    time.sleep(0.2)

    pattern2 = create_pattern_from_pulse("pattern2")
    print("Pattern 2 binary:", "".join(str(b) for b in pattern2.data))

    time.sleep(0.2)

    pattern3 = create_pattern_from_pulse("pattern3")
    print("Pattern 3 binary:", "".join(str(b) for b in pattern3.data))

    # Add patterns to space
    space.add_pattern(pattern1)
    space.add_pattern(pattern2)
    space.add_pattern(pattern3)

    # Create natural associations based on binary similarity
    harmony1 = pattern1.calculate_natural_harmony()
    harmony2 = pattern2.calculate_natural_harmony()
    harmony3 = pattern3.calculate_natural_harmony()

    print("\nNatural Harmonies:")
    print("-" * 40)
    print(f"Pattern 1: {harmony1:.3f}")
    print(f"Pattern 2: {harmony2:.3f}")
    print(f"Pattern 3: {harmony3:.3f}")

    # Associate based on natural harmonies
    space.create_association("pattern1", "stillness", harmony1)
    space.create_association("pattern2", "flow", harmony2)
    space.create_association("pattern3", "stillness", harmony3)

    # Observe spatial relationships
    print("\nObserving Natural Pattern Organization:")
    print("-" * 40)
    for p1_id, relationships in space.spatial_relationships.items():
        for p2_id, strength in relationships.items():
            print(f"Relationship {p1_id} → {p2_id}: {strength:.3f}")

    # Find patterns by concept
    stillness_patterns = space.find_patterns_by_concept("stillness")
    print("\nPatterns Associated with Stillness:")
    print("-" * 40)
    for pattern in stillness_patterns:
        print(f"Pattern {pattern.id}: {pattern.associations['stillness']:.3f}")

    # Create and observe a natural path
    path = space.find_natural_path("pattern1", "pattern3")
    if path:
        print("\nNatural Path Discovered:")
        print("-" * 40)
        print(" → ".join(path))

    # Allow natural evolution
    print("\nAllowing Natural Evolution:")
    print("-" * 40)
    space.evolve()

    # Observe metrics after evolution
    print("\nSpace Metrics After Evolution:")
    print("-" * 40)
    metrics = space.metrics
    print(f"Harmony: {metrics.harmony:.3f}")
    print(f"Spatial Coherence: {metrics.spatial_coherence:.3f}")
    print(f"Pathway Strength: {metrics.pathway_strength:.3f}")


@pytest.mark.integration
def test_concept_path_formation() -> None:
    """Observe how concept paths naturally form and strengthen."""
    space = Space(type=SpaceType.GATEWAY)

    # Create patterns from system states for each concept
    patterns = []
    concepts = ["earth", "water", "air"]

    print("\nGenerating Natural Element Patterns:")
    print("-" * 40)

    for i, concept in enumerate(concepts):
        pattern = create_pattern_from_pulse(f"nature_{i}")
        print(f"{concept} binary:", "".join(str(b) for b in pattern.data))
        space.add_pattern(pattern)
        harmony = pattern.calculate_natural_harmony()
        space.create_association(pattern.id, concept, harmony)
        patterns.append(pattern)
        time.sleep(0.2)  # Ensure different system states

    # Allow natural organization
    space.evolve()

    # Try to find path between concepts
    path = space.create_concept_path("earth", "air")

    print("\nConcept Translation Path:")
    print("-" * 40)
    if path:
        print(" → ".join(path))
        # Show resonance along path
        for i in range(len(path) - 1):
            current = path[i]
            next_id = path[i + 1]
            if (
                current in space.spatial_relationships
                and next_id in space.spatial_relationships[current]
            ):
                resonance = space.spatial_relationships[current][next_id]
                print(f"Resonance {current} → {next_id}: {resonance:.3f}")

    # Observe pattern stability
    print("\nPattern Stability:")
    print("-" * 40)
    for pattern in patterns:
        print(f"Pattern {pattern.id}: {pattern.pattern_stability:.3f}")


@pytest.mark.integration
def test_association_strengthening() -> None:
    """Observe how associations naturally strengthen through interaction."""
    space = Space(type=SpaceType.RESONANCE)

    # Create pattern from system state
    print("\nGenerating Learning Pattern:")
    print("-" * 40)
    pattern = create_pattern_from_pulse("learning_pattern")
    print("Binary:", "".join(str(b) for b in pattern.data))
    space.add_pattern(pattern)

    # Create initial association based on natural harmony
    harmony = pattern.calculate_natural_harmony()
    space.create_association(pattern.id, "learning", harmony)

    print("\nAssociation Strength Evolution:")
    print("-" * 40)
    print(f"Initial: {pattern.associations['learning']:.3f}")

    # Strengthen through natural interaction
    for i in range(3):
        # Get new system state each time
        new_pattern = create_pattern_from_pulse(f"learning_evolution_{i}")
        resonance = pattern.calculate_resonance_with(new_pattern)
        space.strengthen_association(pattern.id, "learning", amount=resonance)
        print(f"After interaction {i+1}: {pattern.associations['learning']:.3f}")
        space.evolve()

    # Observe final metrics
    print("\nFinal Space Metrics:")
    print("-" * 40)
    metrics = space.metrics
    print(f"Harmony: {metrics.harmony:.3f}")
    print(f"Resonance: {metrics.resonance:.3f}")
    print(f"Natural Alignment: {metrics.natural_alignment:.3f}")


@pytest.mark.integration
def test_temporal_spatial_integration() -> None:
    """Test the integration between Memory Palace, KymaState and blooms."""

    # Initialize components
    evolution = PatternEvolution()
    memory_block = MemoryBlock()
    evolution.connect_memory(memory_block)

    # Create pulse observer
    pulse = Pulse()
    pattern_data = []

    # Collect natural binary patterns
    try:
        print("\nCollecting binary patterns from system states...")
        print("-" * 40)
        for _ in range(20):  # Collect 20 samples
            state = pulse.sense()
            if state is not None:
                pattern_data.append(state)
                print(state, end="")  # Print each binary state
            time.sleep(0.1)  # Natural pause between observations
        print("\n")  # New line after pattern
    except KeyboardInterrupt:
        pulse.stop()

    pattern_data = np.array(pattern_data)
    print(f"Collected pattern data: {pattern_data}")
    print(f"Pattern length: {len(pattern_data)}")

    # Detect natural patterns
    patterns = evolution._detect_natural_patterns(pattern_data)
    assert len(patterns) > 0, "Should detect at least one natural pattern"

    # Get primary pattern for testing
    pattern = patterns[0]
    pattern.kyma_state = KymaState()  # Initialize KymaState

    # Create initial memory metrics
    memory_block.write(pattern.name, pattern_data)
    metrics = memory_block.get_metrics(pattern.name)
    assert metrics is not None

    # Test temporal-spatial bridge
    spatial_pattern = {
        "resonance": pattern.resonance_frequency,
        "stability": pattern.properties.get("stability", 0.5),
        "harmony": pattern.properties.get("harmony", 0.5),
    }

    # Integrate memory space into temporal structure
    pattern.kyma_state.integrate_memory_space(spatial_pattern, metrics)

    # Verify temporal structure
    assert len(pattern.kyma_state.standing_waves) > 0, "Should create standing waves"
    assert (
        len(pattern.kyma_state.crystallization_points) > 0
    ), "Should identify crystallization points"

    # Test bloom resonance detection
    bloom_potential = pattern.kyma_state.detect_bloom_resonance(pattern, metrics)
    assert 0.0 <= bloom_potential <= 1.0, "Bloom potential should be normalized"

    # Create and process a bloom event
    bloom = BloomEvent(
        timestamp=time.time(),
        parent_pattern=pattern.name,
        variation_magnitude=0.5,
        resonance_shift=0.3,
        polar_influence=0.4,
        environmental_factors={
            "resonance_support": 0.6,
            "stability_support": 0.7,
            "environmental_rhythm": 0.5,
        },
        stability_impact=0.8,
        emergence_path=[pattern.name],
    )

    # Update temporal structure with bloom
    pattern.kyma_state.update_from_bloom(bloom, metrics)

    # Verify temporal evolution
    assert pattern.kyma_state.quantum_coherence > 0, "Should have non-zero quantum coherence"
    assert len(pattern.kyma_state.quantum_states) > 0, "Should have quantum states"

    # Verify memory palace integration
    updated_metrics = memory_block.get_metrics(pattern.name)
    assert updated_metrics.experience_depth > metrics.experience_depth, "Experience should deepen"
    assert updated_metrics.wonder_potential > 0, "Should have wonder potential"

    # Clean up
    pulse.stop()
