"""Binary pulse of the system.

Observes all physical state changes and represents them in pure binary form.
1 = state changed
0 = no change
"""

import signal
from typing import Any, Optional

import psutil


class Pulse:
    """Pure binary observation of system state."""

    def __init__(self) -> None:
        self.running = True
        self.process = psutil.Process()
        self._last_ctx = self.process.num_ctx_switches()
        self._last_cpu = self.process.cpu_percent()
        self._last_memory = self.process.memory_info().rss
        self.last_state = None  # Tuple of (cpu, memory)
        self.last_change = False
        # Track natural variation ranges
        self._cpu_variations = []
        self._memory_variations = []
        self._adaptation_window = 10  # Number of samples to adapt thresholds
        signal.signal(signal.SIGINT, self._graceful_exit)

    def _graceful_exit(self, signum: Any, frame: Any) -> None:
        print("\nClosing binary space...")
        self.running = False

    def _update_thresholds(self, cpu_delta: float, memory_delta: int) -> None:
        """Let thresholds emerge from natural variations."""
        self._cpu_variations.append(abs(cpu_delta))
        self._memory_variations.append(abs(memory_delta))

        # Keep only recent variations
        if len(self._cpu_variations) > self._adaptation_window:
            self._cpu_variations.pop(0)
        if len(self._memory_variations) > self._adaptation_window:
            self._memory_variations.pop(0)

    def sense(self) -> Optional[int]:
        """Sense current state and return 1 if changed, 0 if not."""
        if not self.running:
            return None

        # Get current system state
        curr_cpu = psutil.cpu_percent()
        curr_memory = psutil.virtual_memory().percent

        # Detect significant changes
        if self.last_state is None:
            self.last_state = (curr_cpu, curr_memory)
            return 0

        cpu_change = abs(curr_cpu - self.last_state[0])
        memory_change = abs(curr_memory - self.last_state[1])

        # Update state
        self.last_state = (curr_cpu, curr_memory)

        # Return binary state based on changes
        if cpu_change > 1.0 or memory_change > 1.0:  # Threshold for significant change
            self.last_change = True
            return 1
        else:
            self.last_change = False
            return 0

    def stop(self):
        """Gracefully exit observation."""
        self.running = False


def observe() -> None:
    """Create binary observation space."""
    pulse = Pulse()

    print("\nOpening binary space...")
    print("1 = state changed")
    print("0 = no change")

    try:
        while pulse.running:
            pulse.sense()
    except KeyboardInterrupt:
        pass

    print("\nSpace closed.")


if __name__ == "__main__":
    observe()
