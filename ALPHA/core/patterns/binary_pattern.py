"""Binary pattern detection and natural interaction core.

This module implements fundamental pattern detection and tracking,
allowing natural pattern emergence and interaction without bias.
"""

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import psutil

from .natural_patterns import NaturalPattern, NaturalPrincipleType


@dataclass
class BinaryPattern:
    """Fundamental binary pattern with natural rhythm detection."""

    sequence: List[int]
    timestamp: datetime
    source: str
    stability: float = 0.0
    resonance: float = 0.0
    interactions: Dict["BinaryPattern", float] = field(default_factory=dict)
    reverberation: Optional["BinaryPattern"] = None
    _rhythm_cache: Optional[List[int]] = field(default=None, repr=False)
    _stability_score: float = field(default=0.0, repr=False)

    def find_natural_rhythm(self) -> "BinaryPattern":
        """Find the natural rhythm within this pattern."""
        if not self._rhythm_cache:
            self._rhythm_cache = self._detect_rhythm()
        return BinaryPattern(
            sequence=self._rhythm_cache,
            timestamp=self.timestamp,
            source=f"{self.source}_rhythm",
        )

    def next_pulse(self) -> "BinaryPattern":
        """Generate next pulse in the pattern's rhythm."""
        if not self._rhythm_cache:
            self.find_natural_rhythm()
        # Rotate rhythm by one position
        if self._rhythm_cache:
            rotated = self._rhythm_cache[1:] + self._rhythm_cache[:1]
            return BinaryPattern(
                sequence=rotated,
                timestamp=datetime.now(),
                source=f"{self.source}_pulse",
            )
        return BinaryPattern(
            sequence=self.sequence,
            timestamp=datetime.now(),
            source=f"{self.source}_pulse",
        )

    def is_stable(self) -> bool:
        """Check if pattern has achieved stability."""
        if self._stability_score == 0.0:
            self._stability_score = self._calculate_stability()
        return self._stability_score > 0.7  # Threshold for stability

    def _detect_rhythm(self) -> List[int]:
        """Detect natural rhythm in binary sequence."""
        if len(self.sequence) < 2:
            return self.sequence

        # Find most common subsequence
        window = len(self.sequence) // 2
        while window > 1:
            for i in range(len(self.sequence) - window + 1):
                subsequence = self.sequence[i : i + window]
                if self._is_rhythmic(subsequence):
                    return subsequence
            window = window // 2
        return self.sequence

    def _is_rhythmic(self, subsequence: List[int]) -> bool:
        """Check if subsequence forms a rhythm."""
        # Look for pattern repetition
        if len(subsequence) < 2:
            return False

        repeats = 0
        for i in range(len(self.sequence) - len(subsequence) + 1):
            if self.sequence[i : i + len(subsequence)] == subsequence:
                repeats += 1

        return repeats >= 2  # At least two repetitions

    def _calculate_stability(self) -> float:
        """Calculate pattern stability score."""
        if not self.sequence:
            return 0.0

        # Factors in rhythm presence and consistency
        rhythm = self._detect_rhythm()
        if not rhythm:
            return 0.0

        rhythm_length = len(rhythm)
        sequence_length = len(self.sequence)

        # Calculate stability ratio
        repeats = self._count_rhythm_repeats(rhythm)
        ratio = sequence_length / rhythm_length if rhythm_length > 0 else 1
        stability = (rhythm_length / sequence_length) * (repeats / ratio)

        return min(1.0, stability)

    def _count_rhythm_repeats(self, rhythm: List[int]) -> int:
        """Count how many times rhythm repeats in sequence."""
        if not rhythm:
            return 0

        repeats = 0
        rhythm_len = len(rhythm)
        for i in range(0, len(self.sequence) - rhythm_len + 1, rhythm_len):
            if self.sequence[i : i + rhythm_len] == rhythm:
                repeats += 1
        return repeats

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

    def to_dict(self) -> Dict:
        """Convert pattern to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BinaryPattern":
        """Create BinaryPattern from dictionary."""
        return cls(**data)

    @classmethod
    def merge(cls, patterns: List["BinaryPattern"]) -> "BinaryPattern":
        """Merge multiple patterns into one."""
        if not patterns:
            return cls(sequence=[], timestamp=0, source="empty_merge")

        # Combine sequences using XOR
        max_len = max(len(p.sequence) for p in patterns)
        merged = []
        for i in range(max_len):
            bit = 0
            for pattern in patterns:
                if i < len(pattern.sequence):
                    bit ^= pattern.sequence[i]
            merged.append(bit)

        return cls(
            sequence=merged, timestamp=max(p.timestamp for p in patterns), source="merged_patterns"
        )


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

    def __init__(self):
        self.interaction_history = []
        self.pattern_history = []
        self.resonance_scores = []
        self.raw_patterns = set()  # Store raw pattern data
        self.pattern_sequences = {}  # Store pattern sequences by source
        self.stability_metrics = {}  # Store stability metrics by pattern
        self.reverberation_map = {}  # Store pattern reverberations
        self.resonance_states = {}  # Store pattern resonance states
        self.max_patterns = 1000

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

            # Calculate pattern entropy for complexity
            hist, _ = np.histogram(sequence_array, bins="auto", density=True)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            max_entropy = np.log2(len(hist)) if len(hist) > 0 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Calculate symmetry
            half_len = len(sequence_array) // 2
            symmetry = np.mean(sequence_array[:half_len] == sequence_array[-half_len:])

            # Let resonance find its natural range
            cpu_freq = psutil.cpu_freq().current / psutil.cpu_freq().max

            # Combine metrics with natural weighting
            resonance = (
                frequency * 0.3  # Base frequency
                + (1 - normalized_entropy) * 0.3  # Complexity (lower entropy is better)
                + symmetry * 0.2  # Structural harmony
                + cpu_freq * 0.2  # Hardware alignment
            )

            self.resonance_states[pattern] = resonance
            pattern.resonance = resonance

            if pattern in self.reverberation_map:
                self.reverberation_map[pattern].resonance = resonance

        return self.resonance_states[pattern]

    def track_pattern_interaction(self, pattern, previous_pattern=None):
        """Track interaction between patterns and update history"""
        if previous_pattern:
            # Convert sequences to numpy arrays for correlation
            current = np.array(pattern.sequence)
            prev = np.array(previous_pattern.sequence)

            # Pad shorter sequence if needed
            if len(current) != len(prev):
                max_len = max(len(current), len(prev))
                current = np.pad(current, (0, max_len - len(current)))
                prev = np.pad(prev, (0, max_len - len(prev)))

            # Calculate correlation
            correlation = np.corrcoef(current, prev)[0, 1]

            # Store interaction data
            interaction = {
                "timestamp": time.time(),
                "correlation": correlation,
                "pattern_length": len(current),
            }
            self.interaction_history.append(interaction)

            # Prune history if needed
            if len(self.interaction_history) > self.max_patterns:
                self.interaction_history = self.interaction_history[-self.max_patterns :]

        # Store raw pattern data
        self.raw_patterns.add(pattern)
        if len(self.raw_patterns) > self.max_patterns:
            # Remove random pattern to maintain size limit
            self.raw_patterns.pop()

        return True

    def _calculate_partnership_strength(
        self,
        sequence1: np.ndarray,
        sequence2: np.ndarray,
        resonance1: float,
        resonance2: float,
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
