"""Adaptive Field - Natural threshold adaptation system."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ThresholdState:
    """State of an adaptive threshold."""

    value: float
    pressure_history: List[float] = field(default_factory=list)
    adaptation_rate: float = 0.1
    pressure_window: int = 100
    last_adaptation: float = 0.0


class AdaptiveField:
    """Base class for components that need adaptive thresholds."""

    def __init__(self):
        self.logger = logging.getLogger("adaptive_field")
        self._thresholds: Dict[str, ThresholdState] = {}
        self._field_coherence: float = 0.5
        self._adaptation_enabled: bool = True

    def register_threshold(self, name: str, initial_value: float) -> None:
        """Register a threshold to be managed by the adaptive field."""
        self._thresholds[name] = ThresholdState(value=initial_value)

    def get_threshold(self, name: str) -> float:
        """Get current threshold value."""
        return self._thresholds[name].value if name in self._thresholds else 0.0

    def sense_pressure(self, name: str, value: float) -> None:
        """Let the field sense pressure on a threshold."""
        if name not in self._thresholds or not self._adaptation_enabled:
            return

        state = self._thresholds[name]
        threshold = state.value

        # Track values near threshold (within field coherence range)
        range_factor = 0.2 + (0.3 * self._field_coherence)  # Range expands with coherence
        if (1 - range_factor) * threshold <= value <= (1 + range_factor) * threshold:
            state.pressure_history.append(value)
            state.pressure_history = state.pressure_history[-state.pressure_window :]

            if len(state.pressure_history) >= state.pressure_window:
                self._adapt_threshold(name)

    def _adapt_threshold(self, name: str) -> None:
        """Let threshold adapt based on accumulated pressure."""
        state = self._thresholds[name]

        # Calculate pressure trend
        avg_pressure = sum(state.pressure_history) / len(state.pressure_history)
        pressure_variance = np.var(state.pressure_history)

        # Adaptation rate influenced by:
        # - Field coherence (more coherent = more stable adaptation)
        # - Pressure variance (high variance = more cautious adaptation)
        # - Time since last adaptation
        adaptation_factor = (
            self._field_coherence * (1.0 - min(1.0, pressure_variance)) * state.adaptation_rate
        )

        # Move threshold toward pressure average
        if abs(avg_pressure - state.value) > 0.1 * state.value:
            old_value = state.value
            state.value += adaptation_factor * (avg_pressure - state.value)
            state.pressure_history.clear()
            state.last_adaptation = avg_pressure

            self.logger.info(f"Threshold {name} adapted: {old_value:.3f} -> {state.value:.3f}")

    def update_field_coherence(self, coherence: float) -> None:
        """Update field coherence which influences adaptation behavior."""
        self._field_coherence = coherence

    def enable_adaptation(self) -> None:
        """Enable threshold adaptation."""
        self._adaptation_enabled = True

    def disable_adaptation(self) -> None:
        """Disable threshold adaptation."""
        self._adaptation_enabled = False
