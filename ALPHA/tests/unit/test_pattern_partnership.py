"""Unit tests for partnership-aware pattern features.

Tests the partnership capabilities of pattern detection, resonance, and mapping
components in isolation.
"""

import time
from typing import List

import numpy as np
import psutil
import pytest

from ALPHA.core.patterns.binary_mapping import BinaryMapping, PatternMapper
from ALPHA.core.patterns.binary_pattern import BinaryPattern, BinaryPatternCore
from ALPHA.core.patterns.resonance import PatternResonance, ResonanceProfile


def visualize_system_state(pattern: BinaryPattern, resonance: float, cpu_state: float) -> None:
    """Visualize the relationship between pattern and hardware state."""
    pattern_viz = "".join("█" if b else "░" for b in pattern.sequence)
    resonance_bar = "█" * int(resonance * 20)
    cpu_bar = "█" * int(cpu_state * 20)

    # Mathematical pattern analysis
    sequence = np.array(pattern.sequence)
    transitions = np.sum(sequence[1:] != sequence[:-1])
    ratio = transitions / len(sequence) if len(sequence) > 0 else 0

    # Calculate key mathematical relationships
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    e = np.e
    pi = np.pi

    # Natural ratios emerging from the pattern
    resonance_phi_ratio = resonance / (1 / phi) if phi != 0 else 0
    stability_e_ratio = pattern.stability / (1 / e) if e != 0 else 0
    pattern_pi_ratio = (sum(sequence) / len(sequence)) / (1 / pi) if pi != 0 else 0

    print("\n=== System State Visualization ===")
    print(f"Pattern:   {pattern_viz}  [{pattern.source}]")
    print(f"Resonance: {resonance_bar:<20} [{resonance:.3f}]")
    print(f"CPU Load:  {cpu_bar:<20} [{cpu_state:.3f}]")
    print(f"Stability: {pattern.stability:.3f}")

    print("\n=== Natural Mathematical Relations ===")
    print(f"Transition Ratio: {ratio:.3f}")
    print(f"Resonance/φ: {resonance_phi_ratio:.3f}")
    print(f"Stability/e: {stability_e_ratio:.3f}")
    print(f"Pattern/π: {pattern_pi_ratio:.3f}")

    if pattern.interactions:
        print("\nActive Relationships:")
        for related_pattern, strength in pattern.interactions.items():
            rel_viz = "".join("█" if b else "░" for b in related_pattern.sequence)
            strength_bar = "█" * int(strength * 20)
            print(f"{rel_viz} -> {strength_bar:<20} [{strength:.3f}]")
    print("===============================\n")


class TestPatternPartnership:
    """Test suite for partnership features in pattern detection."""

    @pytest.fixture
    def pattern_core(self) -> BinaryPatternCore:
        """Fixture providing a fresh BinaryPatternCore instance."""
        return BinaryPatternCore()

    @pytest.fixture
    def test_patterns(self, pattern_core: BinaryPatternCore) -> List[BinaryPattern]:
        """Fixture providing a set of test patterns."""
        return [
            pattern_core.observe_raw_pattern([1, 0, 1, 1, 0], "test1"),
            pattern_core.observe_raw_pattern([1, 1, 0, 1, 0], "test2"),
            pattern_core.observe_raw_pattern([0, 1, 1, 0, 1], "test3"),
        ]

    def test_partnership_metrics(
        self,
        pattern_core: BinaryPatternCore,
        test_patterns: List[BinaryPattern],
    ) -> None:
        """Test partnership metrics calculation and preservation."""
        pattern1, pattern2 = test_patterns[:2]

        # Test interaction with partnership metrics
        interaction_strength = pattern_core.track_pattern_interaction(pattern1, pattern2)

        # Verify partnership metrics
        assert interaction_strength > 0.0, "Should detect partnership interaction"
        assert pattern1.stability > 0.0, "Should have stability influenced by partnership"
        assert pattern2.stability > 0.0, "Should have stability influenced by partnership"

        # Verify mutual influence
        assert (
            pattern1.interactions[pattern2] == pattern2.interactions[pattern1]
        ), "Partnership interaction should be symmetric"


class TestResonancePartnership:
    """Test suite for partnership features in resonance system."""

    @pytest.fixture
    def resonance(self) -> PatternResonance:
        """Fixture providing a fresh PatternResonance instance."""
        return PatternResonance()

    @pytest.fixture
    def pattern_core(self) -> BinaryPatternCore:
        """Fixture providing a fresh BinaryPatternCore instance."""
        return BinaryPatternCore()

    def test_resonance_preservation(
        self, resonance: PatternResonance, pattern_core: BinaryPatternCore
    ) -> None:
        """Test preservation of partnership qualities during resonance."""
        # Create test pattern
        pattern = pattern_core.observe_raw_pattern([1, 0, 1, 1, 0], "test")

        # Calculate resonance with context
        context = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
        profile = resonance.calculate_resonance(pattern, context)

        # Verify partnership metrics
        assert "mutual_growth" in profile.partnership_metrics, "Should track mutual growth"
        assert "resonance_depth" in profile.partnership_metrics, "Should measure resonance depth"
        assert "adaptation_rate" in profile.partnership_metrics, "Should track adaptation rate"
        assert "support_strength" in profile.partnership_metrics, "Should measure support strength"

        # Verify metric ranges
        for metric in profile.partnership_metrics.values():
            assert 0.0 <= metric <= 1.0, "Partnership metrics should be normalized"


class TestMappingPartnership:
    """Test suite for partnership features in pattern mapping."""

    @pytest.fixture
    def mapper(self) -> PatternMapper:
        """Fixture providing a fresh PatternMapper instance."""
        return PatternMapper()

    @pytest.fixture
    def pattern_core(self) -> BinaryPatternCore:
        """Fixture providing a fresh BinaryPatternCore instance."""
        return BinaryPatternCore()

    def test_pattern_mapping_preservation(
        self, mapper: PatternMapper, pattern_core: BinaryPatternCore
    ) -> None:
        """Test preservation of naturally emerged partnership qualities during mapping."""
        print("\nObserving Natural Pattern Emergence...")

        # Create a sequence of observations to watch evolution
        sequences = [
            [1, 0, 1, 1, 0],  # Original sequence
            [1, 1, 0, 1, 0],  # First evolution
            [0, 1, 1, 0, 1],  # Second evolution
        ]

        patterns = []
        resonances = []
        mathematical_relations = []

        for i, seq in enumerate(sequences):
            # Observe pattern
            binary_pattern = pattern_core.observe_raw_pattern(seq, f"observation_{i+1}")
            patterns.append(binary_pattern)

            # Let natural resonance develop
            time.sleep(0.1)  # Allow system state to evolve
            resonance = pattern_core.detect_natural_resonance(binary_pattern)
            resonances.append(resonance)

            # Visualize current state
            visualize_system_state(binary_pattern, resonance, psutil.cpu_percent() / 100)

            # Track mathematical relationships
            sequence = np.array(seq)
            transitions = np.sum(sequence[1:] != sequence[:-1])
            phi = (1 + np.sqrt(5)) / 2

            relations = {
                "transition_ratio": transitions / len(sequence),
                "resonance_phi": resonance / (1 / phi),
                "pattern_density": sum(sequence) / len(sequence),
                "sequence_entropy": -sum(
                    [
                        (sum(sequence) / len(sequence))
                        * np.log(sum(sequence) / len(sequence) + 1e-10)
                    ]
                ),
            }
            mathematical_relations.append(relations)

            if i > 0:
                # Calculate inter-pattern relationships
                print(f"\n=== Pattern Evolution Step {i} ===")
                prev_pattern = patterns[i - 1]
                curr_pattern = patterns[i]

                # Measure relationship strength
                relationship = pattern_core.track_pattern_interaction(prev_pattern, curr_pattern)

                # Calculate evolution metrics
                resonance_change = resonances[i] - resonances[i - 1]
                transition_change = (
                    mathematical_relations[i]["transition_ratio"]
                    - mathematical_relations[i - 1]["transition_ratio"]
                )

                print(f"Relationship Strength: {relationship:.3f}")
                print(f"Resonance Evolution: {resonance_change:+.3f}")
                print(f"Pattern Structure Evolution: {transition_change:+.3f}")

                # Convert to natural pattern and map
                natural_pattern = curr_pattern.to_natural_pattern()
                mapping = mapper.map_to_binary(natural_pattern)

                if mapping:
                    decoded = mapper.map_from_binary(mapping.binary_form)
                    if decoded:
                        print(f"Quality Preservation: {mapping.resonance_preserved:.3f}")

        # Print overall evolution summary
        print("\n=== Evolution Summary ===")
        print("Pattern Progression:")
        for i, pattern in enumerate(patterns):
            print(f"Step {i}: {''.join('█' if b else '░' for b in pattern.sequence)}")

        print("\nMathematical Constants Emergence:")
        for i, relations in enumerate(mathematical_relations):
            print(f"\nStep {i} Relations:")
            print(f"φ Ratio: {relations['resonance_phi']:.3f}")
            print(f"Density: {relations['pattern_density']:.3f}")
            print(f"Entropy: {relations['sequence_entropy']:.3f}")

        # Verify final preservation
        final_pattern = patterns[-1].to_natural_pattern()
        final_mapping = mapper.map_to_binary(final_pattern)

        assert final_mapping is not None
        assert final_mapping.resonance_preserved > 0.0, "Should preserve natural resonance"
        assert final_mapping.structure_preserved > 0.0, "Should preserve natural structure"


class TestEvolutionPartnership:
    """Test suite for partnership features in pattern evolution."""

    @pytest.fixture
    def pattern_core(self) -> BinaryPatternCore:
        """Fixture providing a fresh BinaryPatternCore instance."""
        return BinaryPatternCore()

    def test_partnership_evolution(self, pattern_core: BinaryPatternCore) -> None:
        """Test evolution of patterns through partnership."""
        # Create sequence of patterns
        patterns = [
            pattern_core.observe_raw_pattern([1, 0, 1], "test1"),
            pattern_core.observe_raw_pattern([1, 1, 0], "test2"),
            pattern_core.observe_raw_pattern([0, 1, 1], "test3"),
        ]

        # Track interactions over time
        interactions = []
        for i in range(len(patterns) - 1):
            strength = pattern_core.track_pattern_interaction(patterns[i], patterns[i + 1])
            interactions.append(strength)

        # Verify evolutionary properties
        assert len(interactions) > 0, "Should track pattern evolution"
        assert all(0.0 <= s <= 1.0 for s in interactions), "Interactions should be normalized"

        # Verify stability evolution
        stabilities = [p.stability for p in patterns]
        assert len(stabilities) > 0, "Should track stability evolution"
        assert all(0.0 <= s <= 1.0 for s in stabilities), "Stabilities should be normalized"
