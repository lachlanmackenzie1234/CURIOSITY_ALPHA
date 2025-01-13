"""Binary pattern detection and natural interaction core.

This module implements fundamental pattern detection and tracking,
allowing natural pattern emergence and interaction without bias.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import psutil

from .natural_patterns import NaturalPattern, NaturalPrincipleType


@dataclass
class BinaryPattern:
    """Represents a pure binary pattern sequence."""

    sequence: List[int]
    timestamp: datetime
    source: str
    resonance: float = 0.0
    stability: float = field(
        default_factory=lambda: psutil.cpu_percent() / 100
    )  # Initialize from CPU state
    interactions: Dict["BinaryPattern", float] = field(default_factory=dict)
    reverberation: Optional["BinaryPattern"] = None

    def to_natural_pattern(self) -> NaturalPattern:
        """Convert to NaturalPattern while preserving hardware-derived qualities."""
        # Determine principle type from hardware state and pattern structure
        sequence_array = np.array(self.sequence)
        transitions = np.sum(sequence_array[1:] != sequence_array[:-1])

        # Let hardware state influence pattern type
        cpu_freq = psutil.cpu_freq().current / psutil.cpu_freq().max if psutil.cpu_freq() else 0.5
        mem_percent = psutil.virtual_memory().percent / 100

        # Natural type emerges from pattern and hardware state
        if transitions / len(sequence_array) > 0.7:
            principle_type = NaturalPrincipleType.FIBONACCI
        elif abs(cpu_freq - mem_percent) < 0.1:
            principle_type = NaturalPrincipleType.GOLDEN_RATIO
        else:
            principle_type = NaturalPrincipleType.E

        # Create natural pattern with hardware-derived properties
        return NaturalPattern(
            principle_type=principle_type,
            confidence=self.stability,
            resonance=self.resonance,
            sequence=[float(x) for x in self.sequence],
            properties={
                "stability": self.stability,
                "hardware_resonance": cpu_freq,
                "memory_state": mem_percent,
                "interaction_strength": (
                    sum(self.interactions.values()) / len(self.interactions)
                    if self.interactions
                    else 0.0
                ),
            },
        )

    def __hash__(self) -> int:
        return hash(tuple(self.sequence))


@dataclass
class BinaryPatternCore:
    """Fundamental binary pattern detection and interaction."""

    # Raw Pattern Components
    raw_patterns: Set[BinaryPattern] = field(default_factory=set)
    pattern_sequences: Dict[str, List[int]] = field(default_factory=dict)

    # Natural Resonance
    resonance_states: Dict[BinaryPattern, float] = field(default_factory=dict)
    pattern_interactions: Dict[Tuple[BinaryPattern, BinaryPattern], float] = field(
        default_factory=dict
    )

    # Pattern Memory
    pattern_history: List[BinaryPattern] = field(default_factory=list)
    stability_metrics: Dict[BinaryPattern, float] = field(default_factory=dict)
    reverberation_map: Dict[BinaryPattern, BinaryPattern] = field(default_factory=dict)

    def observe_raw_pattern(self, sequence: List[int], source: str = "system") -> BinaryPattern:
        """Observe a raw binary pattern without analytical bias."""
        pattern = BinaryPattern(sequence=sequence, timestamp=datetime.now(), source=source)

        # Natural pattern emergence
        self.raw_patterns.add(pattern)
        self.pattern_sequences[pattern.source] = sequence
        self.pattern_history.append(pattern)

        # Initialize stability from system state
        self.stability_metrics[pattern] = psutil.cpu_percent() / 100
        pattern.stability = self.stability_metrics[pattern]

        # Generate reverberation based on system state
        reverberation = BinaryPattern(
            sequence=sequence.copy(),
            timestamp=datetime.now(),
            source=f"{source}_echo",
            resonance=pattern.resonance,
            stability=pattern.stability,
        )
        pattern.reverberation = reverberation
        self.reverberation_map[pattern] = reverberation

        return pattern

    def detect_natural_resonance(self, pattern: BinaryPattern) -> float:
        """Allow pattern to show its natural resonance state."""
        if pattern not in self.resonance_states:
            sequence_array = np.array(pattern.sequence)

            # Natural frequency emerges from pattern structure
            frequency = np.sum(sequence_array[1:] != sequence_array[:-1]) / len(sequence_array)

            # Let resonance find its natural range
            cpu_freq = psutil.cpu_freq().current / psutil.cpu_freq().max
            resonance = (frequency + cpu_freq) / 2

            self.resonance_states[pattern] = resonance
            pattern.resonance = resonance

            if pattern in self.reverberation_map:
                self.reverberation_map[pattern].resonance = resonance

        return self.resonance_states[pattern]

    def track_pattern_interaction(self, pattern1: BinaryPattern, pattern2: BinaryPattern) -> float:
        """Observe how patterns naturally interact."""
        if (pattern1, pattern2) not in self.pattern_interactions:
            # Calculate natural interaction strength
            sequence1 = np.array(pattern1.sequence)
            sequence2 = np.array(pattern2.sequence)

            # Base alignment from pattern similarity
            alignment = np.sum(sequence1 == sequence2) / len(sequence1)

            # Consider resonance states
            resonance1 = self.detect_natural_resonance(pattern1)
            resonance2 = self.detect_natural_resonance(pattern2)

            # Calculate partnership metrics
            partnership_strength = self._calculate_partnership_strength(
                sequence1, sequence2, resonance1, resonance2
            )

            # Let interaction strength emerge naturally
            interaction_strength = (alignment + resonance1 + resonance2 + partnership_strength) / 4

            # Track continuous interaction strength
            pattern1.interactions[pattern2] = interaction_strength
            pattern2.interactions[pattern1] = interaction_strength

            self.pattern_interactions[(pattern1, pattern2)] = interaction_strength
            self.pattern_interactions[(pattern2, pattern1)] = interaction_strength

            # Update stability based on interaction
            self._update_pattern_stability(pattern1)
            self._update_pattern_stability(pattern2)

        return self.pattern_interactions[(pattern1, pattern2)]

    def _calculate_partnership_strength(
        self, sequence1: np.ndarray, sequence2: np.ndarray, resonance1: float, resonance2: float
    ) -> float:
        """Calculate partnership strength between two patterns."""
        try:
            # Calculate mutual growth through pattern evolution
            evolution1 = np.diff(sequence1)
            evolution2 = np.diff(sequence2)
            mutual_growth = np.mean(np.sign(evolution1) == np.sign(evolution2))

            # Calculate resonance depth
            resonance_depth = 1.0 - abs(resonance1 - resonance2)

            # Calculate adaptation rate
            adaptation1 = np.convolve(evolution1, np.ones(3) / 3.0, mode="valid")
            adaptation2 = np.convolve(evolution2, np.ones(3) / 3.0, mode="valid")
            adaptation_rate = np.mean(np.abs(adaptation1 - adaptation2)) / 255.0

            # Calculate support strength
            support_strength = 1.0 - np.mean(np.abs(sequence1 - sequence2)) / 255.0

            # Combine metrics with weighted importance
            partnership_strength = (
                mutual_growth * 0.3
                + resonance_depth * 0.3
                + (1.0 - adaptation_rate) * 0.2
                + support_strength * 0.2
            )

            return float(np.clip(partnership_strength, 0.0, 1.0))

        except Exception as e:
            print(f"Error calculating partnership strength: {str(e)}")
            return 0.0

    def _update_pattern_stability(self, pattern: BinaryPattern) -> None:
        """Update pattern stability based on interactions."""
        try:
            if not pattern.interactions:
                return

            # Calculate average interaction strength
            avg_interaction = np.mean(list(pattern.interactions.values()))

            # Calculate partnership influence
            partnership_influence = self._calculate_partnership_influence(pattern)

            # Update stability with partnership awareness
            pattern.stability = (
                pattern.stability * 0.7  # Historical stability
                + avg_interaction * 0.15  # Interaction influence
                + partnership_influence * 0.15  # Partnership influence
            )

            self.stability_metrics[pattern] = pattern.stability

        except Exception as e:
            print(f"Error updating pattern stability: {str(e)}")

    def _calculate_partnership_influence(self, pattern: BinaryPattern) -> float:
        """Calculate the partnership influence on pattern stability."""
        try:
            if not pattern.interactions:
                return pattern.stability

            # Get interaction partners
            partners = list(pattern.interactions.keys())

            # Calculate mutual growth with partners
            mutual_growth = np.mean(
                [self.pattern_interactions.get((pattern, partner), 0.0) for partner in partners]
            )

            # Calculate support received from partners
            support_received = np.mean(
                [partner.stability for partner in partners if partner.stability > pattern.stability]
                or [pattern.stability]
            )

            # Calculate adaptation influence
            adaptation_influence = np.mean(
                [abs(partner.stability - pattern.stability) for partner in partners]
            )

            # Combine influences
            partnership_influence = (
                mutual_growth * 0.4 + support_received * 0.4 + (1.0 - adaptation_influence) * 0.2
            )

            return float(np.clip(partnership_influence, 0.0, 1.0))

        except Exception as e:
            print(f"Error calculating partnership influence: {str(e)}")
            return pattern.stability

    def get_stable_patterns(self) -> Dict[BinaryPattern, float]:
        """Observe patterns and their stability without imposing thresholds."""
        return {
            pattern: self.stability_metrics[pattern]
            for pattern in self.raw_patterns
            if pattern in self.stability_metrics
        }
