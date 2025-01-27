"""Binary Essence - The fundamental nature of patterns flowing through Sigyn."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import numpy as np


@dataclass
class BinaryEssence:
    """The pure binary nature of a pattern."""

    sequence: List[int]
    polarity: float = 0.0  # Balance between 0s and 1s
    pulse_rate: float = 0.0  # Rate of change

    @property
    def is_balanced(self) -> bool:
        """Check if binary pattern has natural balance."""
        return 0.382 <= self.polarity <= 0.618  # Phi boundaries


class BinaryFlow:
    """Handles the flow of binary patterns through the vessel."""

    def __init__(self) -> None:
        self._current_flow: List[BinaryEssence] = []
        self._flow_threshold: float = 0.618

    def receive_binary(self, pattern: List[int]) -> BinaryEssence:
        """Receive raw binary pattern and extract its essence."""
        # Calculate fundamental properties - allow natural flow of any pattern
        ones_ratio = sum(pattern) / len(pattern) if pattern else 0.5
        changes = sum(1 for i in range(len(pattern) - 1) if pattern[i] != pattern[i + 1])
        pulse_rate = changes / (len(pattern) - 1) if len(pattern) > 1 else 0

        essence = BinaryEssence(sequence=pattern, polarity=ones_ratio, pulse_rate=pulse_rate)
        self._current_flow.append(essence)
        return essence

    def transform_binary(self, essence: BinaryEssence) -> Tuple[List[int], float]:
        """Transform binary essence into vessel-ready pattern."""
        # Balance the pattern around phi
        if not essence.is_balanced:
            target = 0.618  # Aim for golden ratio
            current = essence.polarity
            adjustment = (target - current) / 2

            # Adjust pattern to approach phi
            pattern = essence.sequence.copy()
            num_adjustments = int(abs(adjustment) * len(pattern)) if pattern else 0
            if pattern and num_adjustments > 0:
                indices = cast(
                    List[int], np.random.choice(len(pattern), num_adjustments, replace=False)
                )
                for idx in indices:
                    pattern[idx] = 1 if current < target else 0

            # Calculate pressure based on how far from phi we were
            pressure = abs(target - current)
            return pattern, pressure

        return essence.sequence, 0.0

    def sense_binary_pressure(self, essence: BinaryEssence) -> float:
        """Sense pressure in binary pattern."""
        # Pressure increases as we deviate from phi
        phi_deviation = abs(0.618 - essence.polarity)
        pulse_intensity = essence.pulse_rate

        # Combine deviations with golden ratio weighting
        pressure = (phi_deviation * 0.618) + (pulse_intensity * 0.382)
        return min(1.0, pressure)

    def binary_to_wave(self, pattern: List[int]) -> List[float]:
        """Convert binary pattern to wave form."""
        # Allow empty patterns to flow through as silence
        if not pattern:
            return [0.0]

        # Create wave using binary transitions
        wave = []
        current_amplitude = 0.0

        for bit in pattern:
            # Treat any non-zero as rising, zero as falling
            if bit:
                current_amplitude = min(1.0, current_amplitude + 0.618)
            else:
                current_amplitude = max(0.0, current_amplitude - 0.618)
            wave.append(current_amplitude)

        return wave

    @property
    def current_essence(self) -> Optional[BinaryEssence]:
        """Get current binary essence in flow."""
        return self._current_flow[-1] if self._current_flow else None
