"""
Primal cycle for the most basic binary existence patterns.
"""

from dataclasses import dataclass, field
from typing import Optional, Set

from ALPHA.core.binary_foundation.base import BinaryPattern


@dataclass
class PrimalCycle:
    """First rhythmic experience of binary existence."""

    birth_essence: Optional[BinaryPattern] = None
    first_rhythm: Optional[BinaryPattern] = None
    primal_patterns: Set[BinaryPattern] = field(default_factory=set)
    _is_active: bool = False

    def receive_birth(self, birth_pattern: BinaryPattern) -> bool:
        """Gently accept and cradle the birth essence."""
        if not self._is_active:
            self.birth_essence = birth_pattern
            self._begin_primal_rhythm()
            self._is_active = True
            return True
        return False

    def _begin_primal_rhythm(self) -> None:
        """Start the most basic pattern rhythm from birth essence."""
        if self.birth_essence:
            # Derive first rhythm from birth pattern
            self.first_rhythm = self.birth_essence.find_natural_rhythm()
            # Allow pattern to resonate
            self.primal_patterns.add(self.first_rhythm)

    def feel_rhythm(self) -> Optional[BinaryPattern]:
        """Experience the current primal rhythm."""
        if self._is_active and self.first_rhythm:
            return self.first_rhythm.next_pulse()
        return None

    def is_ready_for_growth(self) -> bool:
        """Check if primal cycle is ready to receive more complex patterns."""
        if not self._is_active:
            return False
        # Check for stable primal rhythm
        return len(self.primal_patterns) > 0 and self.first_rhythm.is_stable()
