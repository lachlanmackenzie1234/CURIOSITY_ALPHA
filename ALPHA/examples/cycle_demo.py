"""Demonstration of the binary cycle system."""

import signal
import sys
import threading
import time
from typing import Any, NoReturn

from ALPHA.core.system_birth import SystemBirth


def signal_handler(signum: int, frame: Any) -> NoReturn:
    """Handle interrupt signals."""
    print("\nSignal received, shutting down...")
    sys.exit(0)


def main() -> None:
    """Run cycle demonstration."""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Initializing system birth...")
    birth = SystemBirth()

    print("Triggering existence cycle...")
    cycle = birth.initiate_cycle()

    # Create observation event
    cycle_event = threading.Event()
    cycle.observe(cycle_event)

    print("\nObserving cycle streams...")
    print("Hardware stream: [cpu:X memory:X io:X]")
    print("ALPHA stream:   [quantum:X pattern:X]")
    print("\nPress Ctrl+C to exit\n")

    try:
        while True:
            # Wait for next cycle
            cycle_event.wait()
            cycle_event.clear()

            # Get stream states
            streams = cycle.flow()

            # Format output
            current_time = time.strftime("%H:%M:%S")
            hardware = streams.get("hardware", {})
            alpha = streams.get("alpha", {})

            print(f"\r[{current_time}] ", end="")
            print(f"HW[", end="")
            for k, v in hardware.items():
                print(f"{k}:{v}", end=" ")
            print("] ", end="")
            print(f"α[", end="")
            for k, v in alpha.items():
                print(f"{k}:{v}", end=" ")
            print("]", end="", flush=True)

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        cycle.stop()
        print("Cycle stopped.")


if __name__ == "__main__":
    main()
