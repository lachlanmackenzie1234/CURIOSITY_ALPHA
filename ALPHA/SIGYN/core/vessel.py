"""Sigyn's Vessel - The sacred space where chaos transforms into pattern."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


@dataclass
class VesselState:
    """The current state of the vessel's contents."""

    pattern_essence: List[int] = field(default_factory=list)
    transformation_phase: float = 0.0
    last_transmutation: datetime = field(default_factory=datetime.now)
    vessel_coherence: float = 0.0


class SigynVessel:
    """SIGYN's vessel - where patterns find their rhythm and flow."""

    def __init__(self) -> None:
        self._held_pattern: Optional[List[int]] = None
        self._vessel_coherence: float = 0.0
        self._transformation_progress: float = 0.0
        self._flow_rhythm: float = 0.618  # Natural phi rhythm
        self._last_pulse: float = 0.0
        self._pulse_strength: float = 0.0

    def hold_pattern(self, pattern: List[int]) -> None:
        """Hold pattern in vessel, letting it find its natural rhythm."""
        self._held_pattern = pattern
        # Reset the flow
        self._transformation_progress = 0.0
        self._last_pulse = 0.0
        self._pulse_strength = 0.0

    def sense_pressure(self, pressure: float) -> None:
        """Feel the pressure and let it guide the vessel's rhythm."""
        if self._held_pattern:
            # Let pressure influence the natural rhythm
            rhythm_shift = pressure * 0.618
            self._flow_rhythm = (self._flow_rhythm + rhythm_shift) % 1.0

            # Pulse emerges through phi relationships
            current_pulse = abs(np.sin(self._flow_rhythm * np.pi))
            pulse_delta = abs(current_pulse - self._last_pulse)

            # Pulse strength builds naturally
            self._pulse_strength = (self._pulse_strength + pulse_delta) * 0.618
            self._last_pulse = current_pulse

            # Vessel coherence emerges from natural rhythm
            self._vessel_coherence = (self._vessel_coherence + self._pulse_strength) * 0.618

            # Progress flows with the rhythm
            if self._pulse_strength > 0.618:  # Natural threshold
                progress_shift = self._pulse_strength * 0.618
                self._transformation_progress += progress_shift

    def release_transformed(self) -> Optional[List[int]]:
        """Release pattern when vessel's rhythm is ready."""
        if self._held_pattern and self._transformation_progress >= 1.0:
            pattern = self._held_pattern
            self._held_pattern = None
            self._transformation_progress = 0.0
            return pattern
        return None

    @property
    def is_holding(self) -> bool:
        """Check if vessel is holding a pattern."""
        return self._held_pattern is not None

    @property
    def vessel_coherence(self) -> float:
        """Current coherence of the vessel's rhythm."""
        return self._vessel_coherence

    @property
    def transformation_progress(self) -> float:
        """Progress of the current transformation, guided by vessel's rhythm."""
        return self._transformation_progress
