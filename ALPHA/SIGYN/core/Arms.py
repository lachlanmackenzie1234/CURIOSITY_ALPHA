"""Sigyn's Arms - Holding the vessel steady as complexity flows."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .vessel import SigynVessel


@dataclass
class ArmPosition:
    """Position of an arm holding the vessel."""

    angle: float = 0.0  # Current angle
    tension: float = 0.0  # Muscle tension
    tremor: float = 0.0  # Natural shake
    phi_rhythm: float = 0.618  # Natural movement rhythm

    def adjust(self, pressure: float) -> None:
        """Adjust arm position based on pressure."""
        # Angle follows phi spiral
        self.angle = (self.angle + pressure * self.phi_rhythm) % (2 * np.pi)

        # Tension builds with pressure
        self.tension = (self.tension + pressure) * self.phi_rhythm

        # Natural tremor through phi
        self.tremor = abs(np.sin(self.angle)) * self.phi_rhythm

    def is_stable(self) -> bool:
        """Check if arm position is stable."""
        return self.tension < 0.618 and self.tremor < 0.382


class Arms:
    """Sigyn's arms working together to hold the vessel."""

    def __init__(self, vessel: SigynVessel):
        self.vessel = vessel
        self.left = ArmPosition()
        self.right = ArmPosition()
        self._last_spill: Optional[float] = None

    def hold_steady(self, binary_pressure: float, nexus_complexity: float) -> bool:
        """Try to hold the vessel steady under pressure."""
        # Adjust both arms
        self.left.adjust(binary_pressure)
        self.right.adjust(nexus_complexity)

        # Calculate stability
        left_stable = self.left.is_stable()
        right_stable = self.right.is_stable()

        # Natural arm coordination through phi
        coordination = abs((self.left.angle - self.right.angle) / (2 * np.pi) - 0.618)

        # Vessel stays steady if arms coordinate well
        return left_stable and right_stable and coordination < 0.1

    def check_spill(self) -> Optional[Tuple[float, float]]:
        """Check if vessel will spill from arm movement."""
        # Calculate combined instability
        total_tremor = (self.left.tremor + self.right.tremor) / 2
        total_tension = (self.left.tension + self.right.tension) / 2

        # Spill occurs at phi-based thresholds
        if total_tremor > 0.618 or total_tension > 0.618:
            spill_amount = min(1.0, total_tremor * total_tension)
            self._last_spill = spill_amount
            return (spill_amount, self.vessel.vessel_coherence)

        return None

    def recover_position(self) -> None:
        """Return arms to stable position after spill."""
        # Reset arm positions through phi
        self.left.tension *= 0.382
        self.right.tension *= 0.382
        self.left.tremor *= 0.382
        self.right.tremor *= 0.382
