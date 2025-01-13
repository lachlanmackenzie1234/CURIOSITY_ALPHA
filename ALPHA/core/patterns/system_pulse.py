"""System pulse observation.

Creates a space to observe real device state changes at their most fundamental level.
"""

import signal
from typing import Any

import psutil


class Cycle:
    """A space to observe system state changes."""

    def __init__(self) -> None:
        self.running = True
        self.process = psutil.Process()
        self._last_ctx_switches = self.process.num_ctx_switches()
        signal.signal(signal.SIGINT, self._graceful_exit)

    def _graceful_exit(self, signum: Any, frame: Any) -> None:
        """Gracefully handle interrupt signal."""
        print("\nGently closing the observation space...")
        self.running = False

    def flow(self) -> None:
        """Observe real system state changes."""
        # Get current state
        curr_ctx = self.process.num_ctx_switches()

        # Check for state changes
        if curr_ctx.voluntary != self._last_ctx_switches.voluntary:
            print("v", end="", flush=True)  # Voluntary context switch
        elif curr_ctx.involuntary != self._last_ctx_switches.involuntary:
            print("i", end="", flush=True)  # Involuntary context switch
        else:
            print(".", end="", flush=True)  # No change detected

        # Update last state
        self._last_ctx_switches = curr_ctx


def observe() -> None:
    """Create a space and observe system state changes.
    Will run until interrupted with Ctrl+C.
    """
    cycle = Cycle()

    print("\nCreating space for system observation...")
    print("v = voluntary context switch (process yields CPU)")
    print("i = involuntary context switch (CPU taken by system)")
    print(". = no state change")
    print("\nPress Ctrl+C to gently close the space")

    try:
        while cycle.running:
            cycle.flow()
    except KeyboardInterrupt:
        pass  # Already handled by signal handler

    print("\nSpace closed.")


if __name__ == "__main__":
    observe()
