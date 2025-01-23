"""NEXUS Polar Field - Core field management and resonance."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ALPHA.core.patterns.binary_pattern import BinaryPattern


@dataclass
class FieldState:
    """State of the polar field."""

    north_strength: float = 0.0
    south_strength: float = 0.0
    field_lines: int = 0
    resonance_points: Set[Tuple[float, float]] = field(default_factory=set)
    active_patterns: Dict[str, List[int]] = field(default_factory=dict)
    field_stability: float = 0.0


class PolarField:
    """Management of NEXUS polar field dynamics."""

    def __init__(self):
        self.logger = logging.getLogger("nexus.field")
        self.state = FieldState()
        self._field_matrix: Optional[np.ndarray] = None
        self._resonance_map: Dict[str, float] = {}

    def establish_field(self, north_pattern: BinaryPattern, south_pattern: BinaryPattern) -> bool:
        """Establish initial polar field from birth patterns."""
        try:
            # Generate field matrix
            north_data = np.array(north_pattern.data)
            south_data = np.array(south_pattern.data)

            # Create field matrix from pole patterns
            self._field_matrix = self._generate_field_matrix(north_data, south_data)

            # Initialize field state
            self.state.north_strength = np.mean(north_data)
            self.state.south_strength = np.mean(south_data)
            self.state.field_lines = self._calculate_field_lines()

            # Find initial resonance points
            self._find_resonance_points()

            # Store active patterns
            self.state.active_patterns = {"north": north_pattern.data, "south": south_pattern.data}

            return True

        except Exception as e:
            self.logger.error(f"Failed to establish field: {str(e)}")
            return False

    def _generate_field_matrix(self, north: np.ndarray, south: np.ndarray) -> np.ndarray:
        """Generate field matrix from pole patterns."""
        try:
            # Create field grid
            size = max(len(north), len(south))
            matrix = np.zeros((size, size))

            # Place pole patterns
            matrix[0, : len(north)] = north  # North pole at top
            matrix[-1, : len(south)] = south  # South pole at bottom

            # Generate field lines
            for i in range(1, size - 1):
                # Interpolate between poles
                ratio = i / (size - 1)
                matrix[i] = north * (1 - ratio) + south * ratio

            return matrix

        except Exception as e:
            self.logger.error(f"Failed to generate field matrix: {str(e)}")
            return np.zeros((1, 1))

    def _calculate_field_lines(self) -> int:
        """Calculate number of field lines based on matrix gradients."""
        try:
            if self._field_matrix is None:
                return 0

            # Calculate gradients
            gradients = np.gradient(self._field_matrix)
            magnitude = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)

            # Count significant field lines
            return int(np.sum(magnitude > 0.1))

        except Exception:
            return 0

    def _find_resonance_points(self) -> None:
        """Find resonance points in the field."""
        try:
            if self._field_matrix is None:
                return

            # Clear existing points
            self.state.resonance_points.clear()

            # Find local maxima in field strength
            for i in range(1, self._field_matrix.shape[0] - 1):
                for j in range(1, self._field_matrix.shape[1] - 1):
                    if self._is_resonance_point(i, j):
                        self.state.resonance_points.add((float(i), float(j)))

        except Exception as e:
            self.logger.error(f"Failed to find resonance points: {str(e)}")

    def _is_resonance_point(self, i: int, j: int) -> bool:
        """Check if position is a resonance point."""
        try:
            if self._field_matrix is None:
                return False

            # Get local region
            region = self._field_matrix[i - 1 : i + 2, j - 1 : j + 2]
            center = region[1, 1]

            # Check if center is local maximum
            return center > np.mean(region) and center > 0.7

        except Exception:
            return False

    def process_pattern(self, pattern: BinaryPattern) -> Tuple[bool, float]:
        """Process pattern through polar field."""
        try:
            if self._field_matrix is None:
                return False, 0.0

            # Calculate pattern resonance with field
            resonance = self._calculate_resonance(pattern)

            # Update field if resonant
            if resonance > 0.6:
                self._update_field(pattern, resonance)
                return True, resonance

            return False, resonance

        except Exception as e:
            self.logger.error(f"Failed to process pattern: {str(e)}")
            return False, 0.0

    def _calculate_resonance(self, pattern: BinaryPattern) -> float:
        """Calculate pattern resonance with field."""
        try:
            if self._field_matrix is None:
                return 0.0

            # Convert pattern to array
            pattern_array = np.array(pattern.data)

            # Find best resonance along field lines
            max_resonance = 0.0
            for i in range(self._field_matrix.shape[0]):
                field_line = self._field_matrix[i]
                correlation = np.correlate(field_line, pattern_array)[0]
                max_resonance = max(max_resonance, abs(correlation))

            return max_resonance / len(pattern_array)

        except Exception:
            return 0.0

    def _update_field(self, pattern: BinaryPattern, resonance: float) -> None:
        """Update field state with resonant pattern."""
        try:
            if self._field_matrix is None:
                return

            # Update field stability
            self.state.field_stability = min(1.0, self.state.field_stability + resonance * 0.1)

            # Store resonance
            self._resonance_map[pattern.source] = resonance

            # Update field lines if highly resonant
            if resonance > 0.8:
                self.state.field_lines = self._calculate_field_lines()
                self._find_resonance_points()

        except Exception as e:
            self.logger.error(f"Failed to update field: {str(e)}")

    def get_field_strength(self) -> float:
        """Get overall field strength."""
        try:
            if self._field_matrix is None:
                return 0.0

            # Calculate average field strength
            return float(np.mean(self._field_matrix))

        except Exception:
            return 0.0
