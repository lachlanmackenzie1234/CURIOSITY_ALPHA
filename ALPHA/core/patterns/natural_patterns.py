"""Natural pattern foundations for ALPHA.

This module provides the fundamental natural mathematical patterns and
principles that serve as the foundation for ALPHA's pattern recognition
and translation system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np


class NaturalPrincipleType(Enum):
    """Types of natural mathematical principles."""

    GOLDEN_RATIO = "golden_ratio"  # φ ≈ 1.618033988749895
    FIBONACCI = "fibonacci"  # 1, 1, 2, 3, 5, 8, 13...
    E = "e"  # e ≈ 2.718281828459045
    PI = "pi"  # π ≈ 3.141592653589793
    PHI = "phi"  # φ (same as golden ratio)
    SQRT2 = "sqrt2"  # √2 ≈ 1.4142135623730951
    EULER = "euler"  # γ ≈ 0.5772156649015329
    LN2 = "ln2"  # ln(2) ≈ 0.6931471805599453


@dataclass
class NaturalPattern:
    """Represents a pattern that follows natural mathematical principles."""

    principle_type: NaturalPrincipleType
    confidence: float = 0.0
    resonance: float = 0.0
    sequence: Optional[List[float]] = None
    properties: Dict[str, float] = field(default_factory=dict)
    sub_patterns: Set["NaturalPattern"] = field(default_factory=set)


class NaturalPatternHierarchy:
    """Manages the hierarchy of natural mathematical patterns."""

    def __init__(self):
        """Initialize the natural pattern hierarchy."""
        # Fundamental constants
        self.constants = {
            NaturalPrincipleType.GOLDEN_RATIO: 1.618033988749895,
            NaturalPrincipleType.E: 2.718281828459045,
            NaturalPrincipleType.PI: 3.141592653589793,
            NaturalPrincipleType.SQRT2: 1.4142135623730951,
            NaturalPrincipleType.EULER: 0.5772156649015329,
            NaturalPrincipleType.LN2: 0.6931471805599453,
        }

        # Fundamental sequences
        self.sequences = {NaturalPrincipleType.FIBONACCI: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]}

        # Pattern relationships
        self.relationships: Dict[NaturalPrincipleType, Set[NaturalPrincipleType]] = {
            NaturalPrincipleType.GOLDEN_RATIO: {NaturalPrincipleType.FIBONACCI},
            NaturalPrincipleType.PHI: {NaturalPrincipleType.GOLDEN_RATIO},
            NaturalPrincipleType.E: {NaturalPrincipleType.LN2},
        }

    def detect_natural_pattern(
        self, data: np.ndarray, threshold: float = 0.6
    ) -> Optional[NaturalPattern]:
        """Detect natural patterns in numerical data."""
        best_pattern = None
        max_confidence = threshold

        # Check for each principle type
        for principle_type in NaturalPrincipleType:
            confidence = self._calculate_principle_confidence(data, principle_type)
            if confidence > max_confidence:
                pattern = NaturalPattern(
                    principle_type=principle_type,
                    confidence=confidence,
                    resonance=self._calculate_resonance(data, principle_type),
                )

                # Add related patterns
                if principle_type in self.relationships:
                    for related_type in self.relationships[principle_type]:
                        sub_confidence = self._calculate_principle_confidence(data, related_type)
                        if sub_confidence > threshold:
                            pattern.sub_patterns.add(
                                NaturalPattern(
                                    principle_type=related_type,
                                    confidence=sub_confidence,
                                )
                            )

                best_pattern = pattern
                max_confidence = confidence

        return best_pattern

    def _calculate_principle_confidence(
        self, data: np.ndarray, principle_type: NaturalPrincipleType
    ) -> float:
        """Calculate how closely data matches a natural principle."""
        try:
            if principle_type in self.constants:
                return self._check_constant_ratio(data, self.constants[principle_type])
            elif principle_type in self.sequences:
                return self._check_sequence_match(data, self.sequences[principle_type])
            return 0.0
        except Exception:
            return 0.0

    def _check_constant_ratio(self, data: np.ndarray, constant: float) -> float:
        """Check how closely ratios in data match a constant."""
        if len(data) < 2:
            return 0.0

        ratios = []
        for i in range(len(data) - 1):
            if data[i] != 0:
                ratios.append(abs(data[i + 1] / data[i] - constant))

        if not ratios:
            return 0.0

        # Convert deviations to confidence score
        avg_deviation = np.mean(ratios)
        return 1.0 / (1.0 + avg_deviation)

    def _check_sequence_match(self, data: np.ndarray, sequence: List[float]) -> float:
        """Check how closely data matches a sequence pattern."""
        if len(data) < len(sequence):
            return 0.0

        # Calculate normalized cross-correlation
        correlation = np.correlate(data / np.max(data), sequence / np.max(sequence), mode="valid")

        return float(np.max(correlation))

    def _calculate_resonance(self, data: np.ndarray, principle_type: NaturalPrincipleType) -> float:
        """Calculate pattern resonance with natural principle."""
        try:
            # Base resonance on pattern stability and harmony
            confidence = self._calculate_principle_confidence(data, principle_type)

            # Calculate pattern entropy
            entropy = self._calculate_entropy(data)

            # Calculate pattern symmetry
            symmetry = self._calculate_symmetry(data)

            # Combine metrics with weights
            resonance = (
                confidence * 0.4 + (1 - entropy) * 0.3 + symmetry * 0.3  # Lower entropy is better
            )

            return float(np.clip(resonance, 0.0, 1.0))

        except Exception:
            return 0.0

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate normalized Shannon entropy of the data."""
        try:
            # Calculate histogram
            hist, _ = np.histogram(data, bins="auto", density=True)

            # Calculate entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-10))

            # Normalize to [0, 1]
            max_entropy = np.log2(len(hist))
            if max_entropy == 0:
                return 0.0

            return entropy / max_entropy

        except Exception:
            return 0.0

    def _calculate_symmetry(self, data: np.ndarray) -> float:
        """Calculate symmetry score of the data."""
        try:
            if len(data) < 2:
                return 1.0

            # Compare first half with reversed second half
            mid = len(data) // 2
            first_half = data[:mid]
            second_half = data[-mid:][::-1]  # Reversed

            # Calculate symmetry score
            differences = np.abs(first_half - second_half)
            max_diff = np.max(data) - np.min(data)

            if max_diff == 0:
                return 1.0

            symmetry = 1.0 - np.mean(differences) / max_diff
            return float(symmetry)

        except Exception:
            return 0.0
