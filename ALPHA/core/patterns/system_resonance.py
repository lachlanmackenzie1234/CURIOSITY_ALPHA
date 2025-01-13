"""System resonance detection and measurement.

This module captures and measures real physical system operations (memory access,
state changes, CPU cycles) as a source of natural binary patterns.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil

from .pattern_evolution import NaturalPattern


@dataclass
class SystemOperation:
    """Represents a physical system operation."""

    timestamp: float
    operation_type: str  # 'memory_access', 'state_change', 'cpu_cycle'
    energy_delta: float  # Estimated energy change
    duration: float  # Operation duration in nanoseconds
    metadata: Dict[str, float] = field(default_factory=dict)


class SystemResonance:
    """Captures and measures physical system operations as pattern sources."""

    def __init__(self) -> None:
        """Initialize the system resonance detector."""
        self.operations: List[SystemOperation] = []
        self.process = psutil.Process(os.getpid())
        self._last_cpu_times = self.process.cpu_times()
        self._last_memory_info = self.process.memory_info()
        self._last_timestamp = time.time_ns()

    def capture_cycle(self) -> List[SystemOperation]:
        """Capture one complete cycle of system operations."""
        operations = []

        # Measure CPU cycle
        current_cpu_times = self.process.cpu_times()
        cpu_delta = (current_cpu_times.user - self._last_cpu_times.user) + (
            current_cpu_times.system - self._last_cpu_times.system
        )

        if cpu_delta > 0:
            operations.append(
                SystemOperation(
                    timestamp=time.time_ns(),
                    operation_type="cpu_cycle",
                    energy_delta=cpu_delta * 0.1,  # Rough estimate
                    duration=cpu_delta * 1e9,  # Convert to nanoseconds
                    metadata={
                        "cpu_user": current_cpu_times.user,
                        "cpu_system": current_cpu_times.system,
                    },
                )
            )

        # Measure memory state
        current_memory = self.process.memory_info()
        memory_delta = current_memory.rss - self._last_memory_info.rss

        if memory_delta != 0:
            operations.append(
                SystemOperation(
                    timestamp=time.time_ns(),
                    operation_type="memory_access",
                    energy_delta=abs(memory_delta) * 0.001,  # Rough estimate
                    duration=time.time_ns() - self._last_timestamp,
                    metadata={
                        "rss_delta": memory_delta,
                        "vms": current_memory.vms - self._last_memory_info.vms,
                    },
                )
            )

        # Update state
        self._last_cpu_times = current_cpu_times
        self._last_memory_info = current_memory
        self._last_timestamp = time.time_ns()

        return operations

    def detect_patterns(self, window_size: int = 100) -> Optional[NaturalPattern]:
        """Detect patterns in system operations over a time window."""
        if len(self.operations) < window_size:
            return None

        # Analyze recent operations
        recent_ops = self.operations[-window_size:]

        # Extract timing sequences
        timestamps = [op.timestamp for op in recent_ops]
        energy_deltas = [op.energy_delta for op in recent_ops]

        # Look for rhythmic patterns in timing
        timing_pattern = self._analyze_sequence(timestamps)
        energy_pattern = self._analyze_sequence(energy_deltas)

        if timing_pattern or energy_pattern:
            pattern_strength = max(timing_pattern or 0.0, energy_pattern or 0.0)
            return NaturalPattern(
                name="system_resonance",
                ratio=pattern_strength,
                confidence=pattern_strength,  # Use pattern strength as confidence
                properties={
                    "timing_rhythm": timing_pattern or 0.0,
                    "energy_rhythm": energy_pattern or 0.0,
                    "operation_count": len(recent_ops),
                },
            )

        return None

    def _analyze_sequence(self, values: List[float]) -> Optional[float]:
        """Analyze a sequence of values for patterns."""
        if not values:
            return None

        # Calculate intervals between values
        intervals = [values[i + 1] - values[i] for i in range(len(values) - 1)]

        if not intervals:
            return None

        # Look for consistency in intervals
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)

        # Return rhythm strength (inverse of normalized variance)
        if mean_interval == 0:
            return None

        rhythm_strength = 1.0 / (1.0 + (variance / mean_interval))
        return rhythm_strength if rhythm_strength > 0.5 else None
