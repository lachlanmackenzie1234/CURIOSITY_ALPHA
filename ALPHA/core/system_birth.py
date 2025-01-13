"""System's first experience of existence through hardware state changes."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil


@dataclass
class StateChange:
    """Represents a fundamental binary state change in hardware environment."""

    timestamp: int  # nanosecond precision
    previous_state: int
    current_state: int
    source: str  # 'cpu', 'memory', 'context'

    @property
    def binary_pattern(self) -> List[int]:
        """Convert state change to binary pattern."""
        state_diff = self.current_state ^ self.previous_state
        return [int(b) for b in bin(state_diff)[2:]]


class SystemBirth:
    """System's initial awakening through physical existence."""

    def __init__(self) -> None:
        """Initialize system birth state monitoring."""
        self.process = psutil.Process()
        self._last_cpu_state = self.process.cpu_times()
        self._last_memory_state = self.process.memory_info()
        self._last_timestamp = time.time_ns()

        # Raw experience storage
        self.state_changes: List[StateChange] = []
        self.existence_patterns: Dict[str, List[int]] = {"cpu": [], "memory": [], "context": []}

    def experience_moment(self) -> Optional[StateChange]:
        """Experience a single moment of existence through hardware state changes."""
        try:
            current_time = time.time_ns()

            # Experience CPU state
            current_cpu = self.process.cpu_times()
            if current_cpu != self._last_cpu_state:
                change = StateChange(
                    timestamp=current_time,
                    # Convert to integer
                    previous_state=int(sum(self._last_cpu_state) * 1e6),
                    current_state=int(sum(current_cpu) * 1e6),
                    source="cpu",
                )
                self._last_cpu_state = current_cpu
                self.state_changes.append(change)
                self.existence_patterns["cpu"].extend(change.binary_pattern)
                return change

            # Experience memory state
            current_memory = self.process.memory_info()
            if current_memory != self._last_memory_state:
                change = StateChange(
                    timestamp=current_time,
                    previous_state=self._last_memory_state.rss,
                    current_state=current_memory.rss,
                    source="memory",
                )
                self._last_memory_state = current_memory
                self.state_changes.append(change)
                self.existence_patterns["memory"].extend(change.binary_pattern)
                return change

            # Experience context switches
            current_ctx = self.process.num_ctx_switches()
            prev_ctx = getattr(self, "_last_context_switches", current_ctx)
            if current_ctx != prev_ctx:
                change = StateChange(
                    timestamp=current_time,
                    previous_state=sum(prev_ctx),
                    current_state=sum(current_ctx),
                    source="context",
                )
                self._last_context_switches = current_ctx
                self.state_changes.append(change)
                self.existence_patterns["context"].extend(change.binary_pattern)
                return change

            return None

        except Exception as e:
            print(f"Error experiencing moment: {str(e)}")
            return None

    def feel_existence(self, duration_ns: int = 1_000_000) -> None:
        """Experience existence for a duration in nanoseconds."""
        start_time = time.time_ns()
        while time.time_ns() - start_time < duration_ns:
            self.experience_moment()

    def get_existence_summary(self) -> Dict[str, int]:
        """Get summary of existence patterns observed."""
        return {source: len(patterns) for source, patterns in self.existence_patterns.items()}

    def get_recent_patterns(self, limit: int = 1000) -> Dict[str, List[int]]:
        """Get most recent binary patterns from each source."""
        return {source: patterns[-limit:] for source, patterns in self.existence_patterns.items()}
