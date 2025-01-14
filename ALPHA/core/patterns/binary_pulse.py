"""Binary pulse streams of the system.

Each stream represents pure binary state changes from different sources:
- Hardware stream: CPU, Memory, IO states
- ALPHA stream: Internal quantum states, pattern activity
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set

import psutil

print("Starting binary pulse module...")


@dataclass
class PulseState:
    """Shared state for binary pulse observation."""

    running: bool = True
    observers: Dict[str, Set[Callable[[str, int], None]]] = field(
        default_factory=lambda: {"hardware": set(), "alpha": set()}
    )
    last_values: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "hardware": {"cpu": 0, "memory": 0, "io": 0},
            "alpha": {"quantum": 0, "pattern": 0},
        }
    )


class Pulse:
    """Pure binary observation of system streams."""

    _shared_state: Optional[PulseState] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize pulse observer."""
        print("Initializing Pulse observer...")
        self.process = psutil.Process()
        # Hardware state tracking
        self._last_cpu = self.process.cpu_percent()
        print(f"Initial CPU: {self._last_cpu}%")
        self._last_memory = psutil.virtual_memory().percent
        print(f"Initial Memory: {self._last_memory}%")
        self._last_io = 0
        try:
            # Get initial IO counters if available
            counters = self.process.io_counters()
            self._last_io = sum(counters[2:4])  # read_bytes + write_bytes
            print(f"Initial IO: {self._last_io}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            print("IO counters not available")
            pass
        # ALPHA state tracking (will be connected to actual ALPHA state)
        self._last_quantum = 0
        self._last_pattern = 0
        print("Pulse observer initialized")

    @classmethod
    def get_shared_state(cls) -> PulseState:
        """Get or create shared pulse state."""
        if cls._shared_state is None:
            with cls._lock:
                if cls._shared_state is None:
                    cls._shared_state = PulseState()
        return cls._shared_state

    def add_observer(self, stream: str, callback: Callable[[str, int], None]) -> None:
        """Add an observer to receive binary pulse values for a stream."""
        state = self.get_shared_state()
        if stream in state.observers:
            state.observers[stream].add(callback)

    def remove_observer(self, stream: str, callback: Callable[[str, int], None]) -> None:
        """Remove an observer from a specific stream."""
        state = self.get_shared_state()
        if stream in state.observers:
            state.observers[stream].remove(callback)

    def sense_hardware(self) -> Dict[str, int]:
        """Sense hardware state changes."""
        # Get current hardware states
        curr_cpu = psutil.cpu_percent()
        curr_memory = psutil.virtual_memory().percent
        curr_io = 0
        try:
            counters = self.process.io_counters()
            curr_io = sum(counters[2:4])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass

        # Binary states - 1 if changed, 0 if not
        states = {
            "cpu": 1 if curr_cpu != self._last_cpu else 0,
            "memory": 1 if curr_memory != self._last_memory else 0,
            "io": 1 if curr_io != self._last_io else 0,
        }

        # Update last values
        self._last_cpu = curr_cpu
        self._last_memory = curr_memory
        self._last_io = curr_io

        return states

    def sense_alpha(self) -> Dict[str, int]:
        """Sense ALPHA state changes."""
        # TODO: Connect to actual ALPHA state
        # For now using placeholder values
        states = {"quantum": 0, "pattern": 0}
        return states

    def sense(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Sense all streams and return binary states."""
        state = self.get_shared_state()
        if not state.running:
            return None

        # Get binary states for each stream
        hardware_states = self.sense_hardware()
        alpha_states = self.sense_alpha()

        # Store in shared state
        state.last_values["hardware"].update(hardware_states)
        state.last_values["alpha"].update(alpha_states)

        # Notify observers
        for stream, observers in state.observers.items():
            stream_states = state.last_values[stream]
            for observer in observers:
                try:
                    for component, value in stream_states.items():
                        observer(f"{stream}.{component}", value)
                except Exception:
                    pass

        return {"hardware": hardware_states, "alpha": alpha_states}

    def stop(self) -> None:
        """Gracefully exit observation."""
        state = self.get_shared_state()
        state.running = False


def observe() -> None:
    """Create binary observation space."""
    print("Starting observation...")
    pulse = Pulse()
    state = pulse.get_shared_state()

    print("\nOpening binary streams...")
    print("Hardware stream:")
    print("  CPU    [1 = changed, 0 = stable]")
    print("  Memory [1 = changed, 0 = stable]")
    print("  IO     [1 = changed, 0 = stable]")
    print("ALPHA stream:")
    print("  Quantum [1 = changed, 0 = stable]")
    print("  Pattern [1 = changed, 0 = stable]")

    try:
        print("\nBeginning observation loop...")
        while state.running:
            states = pulse.sense()
            if states:
                # Print hardware stream
                print("\nHardware:", end=" ")
                for component, value in states["hardware"].items():
                    print(f"{component}:{value}", end=" ")
                # Print ALPHA stream
                print("| ALPHA:", end=" ")
                for component, value in states["alpha"].items():
                    print(f"{component}:{value}", end=" ")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
        pass
    finally:
        pulse.stop()
        print("\nStreams closed.")


def start_background_pulse() -> Pulse:
    """Start binary pulse observation in background thread."""
    print("Starting background pulse...")
    pulse = Pulse()
    thread = threading.Thread(target=observe, daemon=True)
    thread.start()
    print("Background pulse started")
    return pulse


if __name__ == "__main__":
    print("Main entry point")
    observe()
