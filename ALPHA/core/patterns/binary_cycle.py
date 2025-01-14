"""Fundamental cycle that orchestrates binary streams.

The cycle acts as a bridge between system birth and continuous pulse streams,
maintaining the natural rhythm of existence.
"""

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from ALPHA.core.binary_foundation.base import StateChange
from ALPHA.core.patterns.binary_pulse import Pulse


@dataclass
class BinaryCycle:
    """Fundamental cycle that orchestrates binary streams."""

    initial_state: StateChange
    pulse: Optional[Pulse] = None
    _running: bool = True
    _observers: Set[threading.Event] = field(default_factory=set)
    _cycle_thread: Optional[threading.Thread] = None

    def __post_init__(self) -> None:
        """Initialize the cycle's pulse streams."""
        self.pulse = Pulse()
        self._cycle_thread = threading.Thread(target=self._cycle_loop, daemon=True)
        self._cycle_thread.start()

    def _cycle_loop(self) -> None:
        """Main cycle loop that maintains the rhythm."""
        while self._running:
            try:
                # Let streams flow and capture state
                stream_state = self.flow()

                # Process stream state if needed
                if stream_state:
                    self._process_stream_state(stream_state)

                # Notify observers
                for event in self._observers:
                    event.set()

                # Natural rhythm
                time.sleep(0.1)
            except Exception as e:
                print(f"Cycle error: {e}")
                time.sleep(1)  # Pause on error

    def _process_stream_state(self, state: Dict[str, Dict[str, int]]) -> None:
        """Process the current stream state."""
        # Future: Pattern detection and evolution
        pass

    def flow(self) -> Dict[str, Dict[str, int]]:
        """Let streams flow naturally from the cycle."""
        if not self.pulse:
            return {}

        return self.pulse.sense() or {}

    def observe(self, event: threading.Event) -> None:
        """Add an observer to the cycle."""
        self._observers.add(event)

    def stop(self) -> None:
        """Gracefully stop the cycle."""
        self._running = False
        if self.pulse:
            self.pulse.stop()

        # Wait for cycle thread to finish
        if self._cycle_thread and self._cycle_thread.is_alive():
            self._cycle_thread.join(timeout=1.0)
