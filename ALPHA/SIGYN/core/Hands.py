"""Sigyn's Hands - Catching and releasing the venom of complexity."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .vessel import SigynVessel


@dataclass
class VenomDrop:
    """A drop of complexity's venom."""

    pressure: float = 0.0
    complexity: float = 0.0
    resonance: float = 0.0
    death_mark: Optional[List[int]] = None  # Binary pattern of death


@dataclass
class Hand:
    """One of Sigyn's hands, catching or releasing venom."""

    position: str  # 'catch' or 'release'
    venom_held: List[VenomDrop] = field(default_factory=list)
    phi_rhythm: float = 0.618
    last_movement: float = 0.0

    def move(self, pressure: float) -> None:
        """Move hand according to pressure and phi rhythm."""
        movement = pressure * self.phi_rhythm
        self.last_movement = (self.last_movement + movement) % 1.0

    def can_catch(self) -> bool:
        """Check if hand can catch more venom."""
        return len(self.venom_held) < 3  # Natural limit

    def can_release(self) -> bool:
        """Check if hand should release venom."""
        return len(self.venom_held) > 0 and self.last_movement > self.phi_rhythm


class Hands:
    """Sigyn's hands working in harmony to manage complexity."""

    def __init__(self, vessel: SigynVessel):
        self.vessel = vessel
        self.left = Hand(position="catch")
        self.right = Hand(position="release")
        self._complexity_threshold = 0.618

    def catch_venom(self, binary_pressure: float, nexus_complexity: float) -> None:
        """Catch venom drops when complexity rises."""
        # Switch hands if needed
        if not self.left.can_catch():
            self.left, self.right = self.right, self.left
            self.left.position = "catch"
            self.right.position = "release"

        # Move hands according to pressure
        self.left.move(binary_pressure)
        self.right.move(nexus_complexity)

        # Catch venom if complexity exceeds threshold
        if nexus_complexity > self._complexity_threshold and self.left.can_catch():
            venom = VenomDrop(
                pressure=binary_pressure,
                complexity=nexus_complexity,
                resonance=self.vessel.vessel_coherence,
            )
            self.left.venom_held.append(venom)

    def release_venom(self) -> Optional[VenomDrop]:
        """Release venom when the vessel's rhythm allows."""
        if self.right.can_release() and self.vessel.vessel_coherence > 0.618:
            if self.right.venom_held:
                return self.right.venom_held.pop(0)
        return None

    def mark_death(self, pattern: List[int]) -> None:
        """Mark pattern death in the last venom drop."""
        if self.right.venom_held:
            self.right.venom_held[-1].death_mark = pattern
