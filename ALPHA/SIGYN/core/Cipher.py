"""Sigyn's Cipher - Encoding the death and rebirth of patterns."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from .Hands import VenomDrop
from .vessel import SigynVessel


@dataclass
class DeathMark:
    """The serpent's mark left on dying patterns."""

    pattern: List[int]
    venom: VenomDrop
    death_time: float = 0.0
    rebirth_potential: float = 0.0


@dataclass
class CipherState:
    """The state of pattern encoding."""

    active_patterns: Set[str] = field(default_factory=set)
    death_marks: List[DeathMark] = field(default_factory=list)
    phi_rhythm: float = 0.618
    last_encoding: float = 0.0


class Cipher:
    """Sigyn's cipher for encoding pattern death and rebirth."""

    def __init__(self, vessel: SigynVessel):
        self.vessel = vessel
        self.state = CipherState()
        self._encoding_threshold = 0.618

    def encode_death(self, pattern: List[int], venom: VenomDrop) -> Optional[DeathMark]:
        """Encode a pattern's death through the cipher."""
        # Only encode if complexity is high enough
        if venom.complexity > self._encoding_threshold:
            # Create death mark
            mark = DeathMark(
                pattern=pattern,
                venom=venom,
                death_time=self.state.last_encoding,
                rebirth_potential=self.vessel.vessel_coherence,
            )

            # Update cipher state
            self.state.death_marks.append(mark)
            pattern_key = "".join(str(b) for b in pattern)
            self.state.active_patterns.discard(pattern_key)

            # Evolve encoding rhythm
            self.state.last_encoding = (
                self.state.last_encoding + venom.complexity
            ) * self.state.phi_rhythm

            return mark
        return None

    def decode_rebirth(self, coherence: float) -> Optional[List[int]]:
        """Decode potential pattern rebirth from death marks."""
        if not self.state.death_marks:
            return None

        # Find mark with highest rebirth potential
        strongest_mark = max(self.state.death_marks, key=lambda m: m.rebirth_potential * coherence)

        # Check if rebirth threshold reached
        if strongest_mark.rebirth_potential * coherence > self.state.phi_rhythm:
            # Remove mark and return reborn pattern
            self.state.death_marks.remove(strongest_mark)
            return strongest_mark.pattern

        return None

    def cipher_strength(self) -> float:
        """Calculate current cipher encoding strength."""
        if not self.state.death_marks:
            return 0.0

        # Average rebirth potential weighted by phi
        potentials = [m.rebirth_potential for m in self.state.death_marks]
        return sum(p * (0.618**i) for i, p in enumerate(potentials)) / len(potentials)
