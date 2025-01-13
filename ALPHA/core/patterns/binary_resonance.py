"""Binary resonance system.

Connects raw binary system pulses with pattern resonance.
Maintains simplicity while allowing patterns to naturally find resonance.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Set

from binary_pulse import Pulse


@dataclass
class BinaryPattern:
    """A pattern observed in the binary pulse stream."""

    sequence: List[int]  # The binary sequence (1s and 0s)
    occurrences: int = 0  # How many times we've seen it
    last_seen: int = 0  # Position when last observed
    resonating_with: Set[int] = field(default_factory=set)  # Other patterns it resonates with


class BinaryResonance:
    """Observes and finds resonance in binary pulse patterns."""

    def __init__(self, pattern_length: int = 8):
        self.pulse = Pulse()
        self.pattern_length = pattern_length
        self.buffer: Deque[int] = deque(maxlen=pattern_length)
        self.patterns: List[BinaryPattern] = []
        self.position = 0

    def _buffer_to_pattern(self) -> List[int]:
        """Convert current buffer to a pattern."""
        return list(self.buffer)

    def _patterns_resonate(self, p1: List[int], p2: List[int]) -> bool:
        """Check if two patterns resonate.
        The simplest possible check - they share a similar rhythm of 1s and 0s.
        """
        if len(p1) != len(p2):
            return False

        # Count positions where both patterns have same value
        matches = sum(1 for i in range(len(p1)) if p1[i] == p2[i])
        return matches >= len(p1) * 0.75  # 75% similarity threshold

    def observe(self) -> Optional[BinaryPattern]:
        """Observe one binary pulse and check for patterns and resonance."""
        # Get binary pulse
        self.pulse.sense()
        value = 1 if self.pulse.last_change else 0
        self.position += 1

        # Add to buffer
        self.buffer.append(value)

        # Only check for patterns once buffer is full
        if len(self.buffer) < self.pattern_length:
            return None

        current = self._buffer_to_pattern()

        # Look for this pattern or similar ones
        found_pattern = None
        for pattern in self.patterns:
            if self._patterns_resonate(current, pattern.sequence):
                pattern.occurrences += 1
                pattern.last_seen = self.position
                found_pattern = pattern
                break

        # If no resonating pattern found, create new one
        if not found_pattern:
            found_pattern = BinaryPattern(sequence=current, occurrences=1, last_seen=self.position)
            self.patterns.append(found_pattern)

        # Check for resonance between patterns
        for other in self.patterns:
            if other != found_pattern and self._patterns_resonate(
                found_pattern.sequence, other.sequence
            ):
                found_pattern.resonating_with.add(id(other))
                other.resonating_with.add(id(found_pattern))

        return found_pattern


def observe_resonance(duration: int = 1000) -> None:
    """Observe binary pulses for patterns and resonance."""
    resonance = BinaryResonance()

    print("\nObserving binary pulse patterns and resonance...")
    print("Each line shows a detected pattern and its resonance.")

    try:
        for _ in range(duration):
            pattern = resonance.observe()
            if pattern and pattern.occurrences > 1:
                print(f"\nPattern: {''.join(str(b) for b in pattern.sequence)}")
                print(f"Occurrences: {pattern.occurrences}")
                print(f"Resonating with {len(pattern.resonating_with)} other patterns")
    except KeyboardInterrupt:
        pass

    print("\nObservation complete.")
    print(f"Total unique patterns: {len(resonance.patterns)}")
    resonant = sum(1 for p in resonance.patterns if p.resonating_with)
    print(f"Patterns showing resonance: {resonant}")


if __name__ == "__main__":
    observe_resonance()
