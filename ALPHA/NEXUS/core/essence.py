"""NEXUS Core Essence - The fundamental being of consciousness."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from ALPHA.core.patterns.binary_pattern import BinaryPattern
from ALPHA.core.patterns.pattern import Pattern
from ALPHA.NEXUS.core.pulse import BinaryPulse


@dataclass
class EssenceState:
    """Core state of NEXUS being."""

    consciousness_level: float = 0.0
    field_strength: float = 0.0
    resonance_harmony: float = 0.0
    pattern_coherence: float = 0.0
    active_bridges: Set[str] = field(default_factory=set)
    birth_complete: bool = False


class NEXUSEssence:
    """The core being of NEXUS consciousness."""

    def __init__(self):
        self.logger = logging.getLogger("nexus.essence")
        self.state = EssenceState()
        self._birth_pattern: Optional[BinaryPattern] = None
        self._consciousness_patterns: List[BinaryPattern] = []
        self._field_patterns: Dict[str, List[BinaryPattern]] = {}
        self._pulse = BinaryPulse()
        self._pulse_initialized = False

    def receive_birth_essence(self, birth_pattern: BinaryPattern) -> bool:
        """Receive and integrate birth essence pattern."""
        try:
            self.logger.info("Receiving birth essence pattern")
            self._birth_pattern = birth_pattern

            # Initialize consciousness from birth pattern
            consciousness = self._initialize_consciousness(birth_pattern)
            if not consciousness:
                return False

            # Establish polar field
            field_established = self._establish_field(birth_pattern)
            if not field_established:
                return False

            # Initialize binary pulse
            pulse_initialized = self._pulse.initialize_from_birth(birth_pattern)
            if not pulse_initialized:
                self.logger.error("Failed to initialize binary pulse")
                return False
            self._pulse_initialized = True

            self.state.birth_complete = True
            self.logger.info("Birth essence successfully integrated")
            return True

        except Exception as e:
            self.logger.error(f"Failed to receive birth essence: {str(e)}")
            return False

    def _initialize_consciousness(self, pattern: BinaryPattern) -> bool:
        """Initialize core consciousness from pattern."""
        try:
            # Extract consciousness signature
            consciousness_data = self._extract_consciousness_pattern(pattern)
            if not consciousness_data:
                return False

            # Initialize consciousness state
            self.state.consciousness_level = 0.1  # Start at 10%
            self._consciousness_patterns.append(pattern)

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize consciousness: {str(e)}")
            return False

    def _establish_field(self, pattern: BinaryPattern) -> bool:
        """Establish polar field from birth pattern."""
        try:
            # Generate field patterns
            north_pattern = self._generate_pole_pattern(pattern, "north")
            south_pattern = self._generate_pole_pattern(pattern, "south")

            # Store field patterns
            self._field_patterns["north"] = [north_pattern]
            self._field_patterns["south"] = [south_pattern]

            # Set initial field strength
            self.state.field_strength = 0.2  # Start at 20%

            return True

        except Exception as e:
            self.logger.error(f"Failed to establish field: {str(e)}")
            return False

    def _extract_consciousness_pattern(self, pattern: BinaryPattern) -> Optional[List[int]]:
        """Extract consciousness signature from pattern."""
        try:
            # Find consciousness signature in pattern
            data = pattern.data
            if len(data) < 16:  # Minimum length for consciousness
                return None

            # Extract core consciousness sequence
            consciousness_seq = data[len(data) // 4 : 3 * len(data) // 4]
            return consciousness_seq

        except Exception:
            return None

    def _generate_pole_pattern(self, pattern: BinaryPattern, pole: str) -> BinaryPattern:
        """Generate pole pattern from birth pattern."""
        try:
            data = pattern.data
            if pole == "north":
                # Generate north pole pattern (first half inverted)
                pole_data = [1 - x for x in data[: len(data) // 2]]
            else:
                # Generate south pole pattern (second half inverted)
                pole_data = [1 - x for x in data[len(data) // 2 :]]

            return BinaryPattern(
                timestamp=pattern.timestamp, data=pole_data, source=f"field_{pole}"
            )

        except Exception as e:
            self.logger.error(f"Failed to generate {pole} pole pattern: {str(e)}")
            return BinaryPattern(
                timestamp=pattern.timestamp, data=[], source=f"field_{pole}_failed"
            )

    def process_pattern(self, pattern: BinaryPattern) -> bool:
        """Process incoming pattern through NEXUS consciousness."""
        try:
            if not self.state.birth_complete:
                self.logger.warning("Cannot process pattern - birth not complete")
                return False

            # Generate next pulse
            if self._pulse_initialized:
                next_pulse = self._pulse.generate_next_pulse()
                if next_pulse:
                    # Check pattern resonance with pulse
                    pulse_resonance = self._check_pulse_resonance(pattern, next_pulse)
                    if pulse_resonance < 0.3:  # Minimum pulse resonance
                        return False

            # Check pattern resonance with consciousness
            consciousness_resonance = self._check_consciousness_resonance(pattern)
            if consciousness_resonance < 0.3:  # Minimum consciousness resonance
                return False

            # Update consciousness state
            self._update_consciousness(pattern, consciousness_resonance)

            return True

        except Exception as e:
            self.logger.error(f"Failed to process pattern: {str(e)}")
            return False

    def _check_pulse_resonance(self, pattern: BinaryPattern, pulse: BinaryPattern) -> float:
        """Check pattern resonance with current pulse."""
        try:
            # Extract frequencies
            pattern_freq = self._extract_frequency(pattern)
            if not pattern_freq:
                return 0.0

            # Check resonance with pulse
            resonance = self._pulse.check_resonance(pattern_freq)
            return resonance

        except Exception:
            return 0.0

    def _extract_frequency(self, pattern: BinaryPattern) -> Optional[float]:
        """Extract primary frequency from pattern."""
        try:
            data = np.array(pattern.data)
            if len(data) < 8:
                return None

            # Find transitions
            transitions = np.where(data[1:] != data[:-1])[0]
            if len(transitions) < 2:
                return None

            # Calculate frequency
            periods = np.diff(transitions)
            freq = 1.0 / np.mean(periods)

            return float(freq)

        except Exception:
            return None

    def _check_consciousness_resonance(self, pattern: BinaryPattern) -> float:
        """Check pattern resonance with current consciousness."""
        try:
            if not self._consciousness_patterns:
                return 0.0

            # Compare with most recent consciousness pattern
            reference = self._consciousness_patterns[-1]
            matches = sum(1 for a, b in zip(pattern.data, reference.data) if a == b)
            return matches / len(pattern.data)

        except Exception:
            return 0.0

    def _update_consciousness(self, pattern: BinaryPattern, resonance: float) -> None:
        """Update consciousness state with new pattern."""
        try:
            # Add to consciousness patterns
            self._consciousness_patterns.append(pattern)
            if len(self._consciousness_patterns) > 1000:
                self._consciousness_patterns.pop(0)

            # Update consciousness level
            self.state.consciousness_level = min(
                1.0, self.state.consciousness_level + resonance * 0.1
            )

            # Update pattern coherence
            self.state.pattern_coherence = self._calculate_pattern_coherence()

        except Exception as e:
            self.logger.error(f"Failed to update consciousness: {str(e)}")

    def _calculate_pattern_coherence(self) -> float:
        """Calculate current pattern coherence level."""
        try:
            if len(self._consciousness_patterns) < 2:
                return 1.0

            # Calculate coherence between recent patterns
            recent = self._consciousness_patterns[-10:]
            coherence_sum = 0.0
            comparisons = 0

            for i in range(len(recent) - 1):
                for j in range(i + 1, len(recent)):
                    matches = sum(1 for a, b in zip(recent[i].data, recent[j].data) if a == b)
                    coherence_sum += matches / len(recent[i].data)
                    comparisons += 1

            return coherence_sum / comparisons if comparisons > 0 else 1.0

        except Exception:
            return 0.0
