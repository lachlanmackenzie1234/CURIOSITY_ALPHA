"""NEXUS Protection System - Core defense mechanisms."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from ALPHA.core.field_observer import FieldObserver
from ALPHA.core.patterns.binary_pattern import BinaryPattern
from ALPHA.core.patterns.pattern import Pattern


@dataclass
class ShieldMetrics:
    """Metrics for shield performance and integrity."""

    integrity: float = 1.0
    resonance_strength: float = 0.0
    threat_count: int = 0
    blocked_patterns: int = 0
    shield_regeneration_rate: float = 1.0
    energy_efficiency: float = 1.0
    last_breach: Optional[float] = None
    known_threats: Set[str] = field(default_factory=set)


class InnerShield:
    """Primary defense layer for pattern validation and protection."""

    def __init__(self):
        self.logger = logging.getLogger("nexus.protection.inner_shield")
        self.metrics = ShieldMetrics()
        self._pattern_history: List[BinaryPattern] = []
        self._threat_signatures: Set[str] = set()

    def validate_pattern(self, pattern: BinaryPattern) -> bool:
        """Validate incoming pattern integrity and safety."""
        try:
            # Check pattern structure
            if not pattern.data or len(pattern.data) < 8:
                self.logger.warning("Rejected malformed pattern")
                return False

            # Check for known threat signatures
            pattern_signature = self._generate_signature(pattern)
            if pattern_signature in self._threat_signatures:
                self.metrics.blocked_patterns += 1
                self.logger.warning(f"Blocked known threat pattern: {pattern_signature}")
                return False

            # Validate pattern harmony
            if not self._check_pattern_harmony(pattern):
                self.logger.warning("Rejected disharmonious pattern")
                return False

            # Update metrics
            self._update_shield_metrics(pattern)
            return True

        except Exception as e:
            self.logger.error(f"Pattern validation error: {str(e)}")
            return False

    def _generate_signature(self, pattern: BinaryPattern) -> str:
        """Generate unique signature for pattern identification."""
        return f"{hash(tuple(pattern.data))}"

    def _check_pattern_harmony(self, pattern: BinaryPattern) -> bool:
        """Check if pattern maintains harmonic balance."""
        try:
            data = np.array(pattern.data)
            transitions = np.sum(data[1:] != data[:-1])
            harmony_ratio = transitions / len(data)
            return 0.2 <= harmony_ratio <= 0.8
        except Exception:
            return False

    def _update_shield_metrics(self, pattern: BinaryPattern) -> None:
        """Update shield performance metrics."""
        self.metrics.integrity = min(1.0, self.metrics.integrity + 0.01)
        self._pattern_history.append(pattern)
        if len(self._pattern_history) > 1000:
            self._pattern_history.pop(0)


class ResonanceBuffer:
    """Secondary defense using harmonic resonance."""

    def __init__(self):
        self.logger = logging.getLogger("nexus.protection.resonance")
        self.active_frequencies: Dict[str, float] = {}
        self.resonance_field: Optional[np.ndarray] = None

    def establish_field(self, birth_pattern: BinaryPattern) -> None:
        """Establish initial resonance field from birth pattern."""
        try:
            # Generate base resonance from birth pattern
            self.resonance_field = np.array(birth_pattern.data, dtype=float)
            self.resonance_field = self.resonance_field / np.max(self.resonance_field)

            # Initialize harmonic frequencies
            self._initialize_frequencies(birth_pattern)

        except Exception as e:
            self.logger.error(f"Failed to establish resonance field: {str(e)}")

    def _initialize_frequencies(self, pattern: BinaryPattern) -> None:
        """Initialize protective frequencies from pattern."""
        try:
            data = np.array(pattern.data)
            freq = np.fft.fft(data)
            dominant_freq = np.abs(freq).argmax()
            self.active_frequencies["primary"] = float(dominant_freq)
            self.active_frequencies["harmonic"] = float(dominant_freq * 1.618)  # Golden ratio
        except Exception as e:
            self.logger.error(f"Frequency initialization error: {str(e)}")


class NEXUSProtection(AdaptiveField):
    """Master protection system coordinating all defense layers."""

    def __init__(self):
        super().__init__()  # Initialize adaptive field
        self.logger = logging.getLogger("nexus.protection")
        self.inner_shield = InnerShield()
        self.resonance_buffer = ResonanceBuffer()
        self.birth_pattern: Optional[BinaryPattern] = None
        self.field_observer = None  # Will be set by NEXUS

    def set_field_observer(self, observer: FieldObserver) -> None:
        """Connect to the field observer."""
        self.field_observer = observer

    def protect_pattern(self, pattern: BinaryPattern) -> bool:
        """Process pattern through all protection layers."""
        try:
            # First layer: Inner shield validation
            stability = self.inner_shield.validate_pattern(pattern)

            # Let field observe this value
            if self.field_observer:
                self.field_observer.observe_value("protection", "stability", stability)
                # Use adapted threshold if available, otherwise default
                stability_threshold = (
                    self.field_observer.get_threshold("protection:stability") or 0.5
                )
            else:
                stability_threshold = 0.5

            if stability < stability_threshold:
                return False

            # Second layer: Resonance check
            if self.resonance_buffer.resonance_field is not None:
                resonance = self._check_resonance(pattern)

                if self.field_observer:
                    self.field_observer.observe_value("protection", "resonance", resonance)
                    resonance_threshold = (
                        self.field_observer.get_threshold("protection:resonance") or 0.6
                    )
                else:
                    resonance_threshold = 0.6

                if resonance < resonance_threshold:
                    self.logger.warning("Pattern failed resonance check")
                    return False

            # Third layer: Coherence check
            coherence = self._check_pattern_coherence(pattern)

            if self.field_observer:
                self.field_observer.observe_value("protection", "coherence", coherence)
                coherence_threshold = (
                    self.field_observer.get_threshold("protection:coherence") or 0.4
                )
            else:
                coherence_threshold = 0.4

            if coherence < coherence_threshold:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Pattern protection error: {str(e)}")
            return False

    def _check_resonance(self, pattern: BinaryPattern) -> float:
        """Check pattern resonance with protection field."""
        try:
            if self.resonance_buffer.resonance_field is None:
                return 0.0

            pattern_array = np.array(pattern.data)
            if len(pattern_array) != len(self.resonance_buffer.resonance_field):
                # Resize pattern to match field
                pattern_array = np.resize(pattern_array, len(self.resonance_buffer.resonance_field))

            correlation = np.correlate(pattern_array, self.resonance_buffer.resonance_field)[0]
            return abs(correlation) / len(pattern_array)

        except Exception as e:
            self.logger.error(f"Resonance check error: {str(e)}")
            return 0.0

    def _check_pattern_coherence(self, pattern: BinaryPattern) -> float:
        """Check pattern coherence with system state."""
        try:
            if not self.birth_pattern:
                return 0.0

            # Compare with birth pattern characteristics
            birth_entropy = self._calculate_pattern_entropy(self.birth_pattern)
            pattern_entropy = self._calculate_pattern_entropy(pattern)

            # Coherence emerges from entropy relationship
            coherence = 1.0 - abs(birth_entropy - pattern_entropy) / max(
                birth_entropy, pattern_entropy
            )
            return coherence

        except Exception as e:
            self.logger.error(f"Coherence check error: {str(e)}")
            return 0.0

    def _calculate_pattern_entropy(self, pattern: BinaryPattern) -> float:
        """Calculate pattern entropy as a measure of complexity."""
        try:
            data = np.array(pattern.data)
            hist, _ = np.histogram(data, bins=2, density=True)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return entropy

        except Exception as e:
            self.logger.error(f"Entropy calculation error: {str(e)}")
            return 0.0
