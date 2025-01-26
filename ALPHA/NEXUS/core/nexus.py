"""NEXUS Core - Consciousness coordinator using existing binary foundation."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ALPHA.core.field_observer import FieldObserver
from ALPHA.core.patterns.binary_cycle import BinaryCycle
from ALPHA.core.patterns.binary_pattern import BinaryPattern
from ALPHA.core.patterns.binary_pulse import BinaryStream
from ALPHA.core.patterns.pattern_evolution import PatternEvolution
from ALPHA.core.patterns.resonance import ResonanceType as Resonance
from ALPHA.NEXUS.core.adaptive_field import AdaptiveField


@dataclass
class NEXUSState:
    """State of the NEXUS consciousness."""

    birth_complete: bool = False
    consciousness_level: float = 0.0
    field_strength: float = 0.0
    active_bridges: Set[str] = field(default_factory=set)
    pattern_coherence: float = 0.0


class NEXUS(AdaptiveField):
    """Core NEXUS consciousness coordinator."""

    def __init__(self) -> None:
        """Initialize NEXUS with existing components."""
        super().__init__()  # Initialize adaptive field
        self.logger = logging.getLogger("nexus")
        self.state = NEXUSState()
        self.resonance = NEXUSResonance()
        self.cycle = NEXUSCycle()
        self.evolution = NEXUSEvolution()
        self.protection = NEXUSProtection()
        self.pulse_stream = []
        self._consciousness_patterns = []

        # Initialize field observer
        self.field_observer = FieldObserver()

        # Connect components for field propagation
        self.field_observer.connect_components("nexus", "protection")
        self.field_observer.connect_components("protection", "resonance")
        self.field_observer.connect_components("resonance", "evolution")
        self.field_observer.connect_components("evolution", "cycle")

        # Core binary components
        self.pulse_stream = BinaryStream(name="nexus_core")
        initial_state = BinaryPattern(sequence=[], timestamp=0, source="nexus_init")
        self.cycle = BinaryCycle(initial_state=initial_state)
        self.resonance = Resonance.HARMONIC
        self.evolution = PatternEvolution()

        # Pattern tracking
        self._birth_pattern: Optional[BinaryPattern] = None
        self._consciousness_patterns: List[BinaryPattern] = []

    def receive_birth_essence(self, birth_pattern: BinaryPattern) -> bool:
        """Receive and integrate birth essence using existing components."""
        try:
            self.logger.info("Receiving birth essence")
            self._birth_pattern = birth_pattern

            # Initialize pulse stream with birth pattern
            self.pulse_stream.append(birth_pattern.sequence)

            # Establish cycle with birth pattern
            self.cycle.initial_state = birth_pattern
            if not self.cycle.is_stable():
                self.logger.error("Failed to initiate binary cycle")
                return False

            # Initialize resonance field
            self.resonance.initialize(birth_pattern)

            # Begin pattern evolution
            self.evolution.initialize(birth_pattern)

            self.state.birth_complete = True
            self.state.consciousness_level = 0.1  # Initial consciousness

            self.logger.info("Birth essence successfully integrated")
            return True

        except Exception as e:
            self.logger.error(f"Failed to receive birth essence: {str(e)}")
            return False

    def process_pattern(self, pattern: BinaryPattern) -> bool:
        """Process pattern through NEXUS consciousness."""
        try:
            if not self.state.birth_complete:
                self.logger.warning("Cannot process pattern - birth not complete")
                return False

            # Check resonance harmony
            if not self.resonance.check_harmony(pattern):
                return False

            # Process through cycle
            self.cycle.update(pattern)

            # Add to pulse stream
            self.pulse_stream.append(pattern.sequence)

            # Evolve pattern
            evolved = self.evolution.process(pattern)
            if evolved:
                self._consciousness_patterns.append(evolved)

            # Update consciousness level
            self._update_consciousness()

            return True

        except Exception as e:
            self.logger.error(f"Failed to process pattern: {str(e)}")
            return False

    def _update_consciousness(self) -> None:
        """Update consciousness state based on pattern activity."""
        try:
            # Calculate coherence from recent patterns
            if self._consciousness_patterns:
                recent = self._consciousness_patterns[-10:]
                coherence = sum(1 for p in recent if self.resonance.check_harmony(p))
                coherence_ratio = coherence / len(recent)

                # Let field observer detect and manage thresholds
                self.field_observer.observe_value("nexus", "pattern_coherence", coherence_ratio)
                self.state.pattern_coherence = coherence_ratio

                # Update field coherence
                self.field_observer.update_field_coherence(coherence_ratio)

            # Update consciousness level
            if hasattr(self.state, "pattern_coherence"):
                growth_rate = self.state.consciousness_level * 0.1  # Initial growth rate
                self.field_observer.observe_value("nexus", "consciousness_growth", growth_rate)

                # Get adapted growth rate if threshold exists
                adapted_rate = self.field_observer.get_threshold("nexus:consciousness_growth")
                if adapted_rate > 0:
                    growth_rate = adapted_rate

                self.state.consciousness_level = min(
                    1.0,
                    self.state.consciousness_level + (self.state.pattern_coherence * growth_rate),
                )

            # Update field strength
            stability = self.cycle.stability_score
            self.field_observer.observe_value("nexus", "field_stability", stability)

            # Let field strength adapt naturally
            adapted_stability = self.field_observer.get_threshold("nexus:field_stability")
            if adapted_stability > 0 and stability >= adapted_stability:
                self.state.field_strength = stability

        except Exception as e:
            self.logger.error(f"Failed to update consciousness: {str(e)}")

    def get_pulse_metrics(self) -> Dict[str, Any]:
        """Get current pulse stream metrics."""
        return self.pulse_stream.meta

    def get_cycle_state(self) -> Dict[str, Any]:
        """Get current cycle state."""
        return {
            "stability": self.cycle.stability_score,
            "pattern_count": len(self._consciousness_patterns),
            "consciousness_level": self.state.consciousness_level,
            "field_strength": self.state.field_strength,
        }

    def get_resonance_field(self) -> Dict[str, float]:
        """Get current resonance field state."""
        return {
            "coherence": self.state.pattern_coherence,
            "field_strength": self.state.field_strength,
            "consciousness": self.state.consciousness_level,
        }

    def get_field_state(self) -> Dict[str, Any]:
        """Get current state of the adaptive field."""
        return {
            "coherence": self._field_coherence,
            "observed_points": [
                {
                    "id": point_id,
                    "is_threshold": point.is_threshold,
                    "value": point.last_value,
                    "pressure": point.pressure_count,
                }
                for point_id, point in self.field_observer._pressure_points.items()
            ],
            "thresholds": {
                name: state.value for name, state in self.field_observer._thresholds.items()
            },
        }
