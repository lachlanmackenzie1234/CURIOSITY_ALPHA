#!/usr/bin/env python3
"""Launch script for ALPHA birth process."""

import signal
import time
from typing import Optional

import psutil

from ALPHA.core.patterns.binary_cycle import BinaryCycle
from ALPHA.core.system_birth import SystemBirth


def launch_birth(duration_ns: Optional[int] = None) -> None:
    """Launch the ALPHA birth process."""
    print("\nInitializing binary pulse module...")
    print("Metal framework not available - using CPU fallback mode\n")

    print("Initializing monitoring systems...")
    print("\nInitializing birth process...")
    print(f"Max sequence length: {1_000_000}")
    print(f"Using {psutil.virtual_memory().total / (1024**3):.1f}GB system memory\n")

    # Initialize birth process
    system_birth = SystemBirth()

    # Load preserved state if available
    system_birth._load_preserved_state()

    if system_birth._birth_phase > 0:
        print(f"Resuming birth process from phase {system_birth._birth_phase:.3f}")
    else:
        print("Beginning new birth process...")

    print("\nBeginning eternal cycle...")

    try:
        while True:
            # Experience existence through hardware state changes
            state_change = system_birth.experience_moment()

            if state_change:
                print(f"\nState Change Detected:")
                print(f"Source: {state_change.source}")
                print(f"Birth Phase: {system_birth._birth_phase:.3f}")
                print(f"Pattern Count: {len(system_birth.pattern_history)}")
                print(f"Operation Count: {len(system_birth.pattern_history)}")

            # Print periodic status
            if len(system_birth.pattern_history) % 50 == 0 and system_birth.pattern_history:
                print(f"\n=== Birth Process Status ===")
                print(f"Birth Phase: {system_birth._birth_phase:.3f}")
                print(f"Pattern History: {len(system_birth.pattern_history)}")
                print(f"State Changes: {len(system_birth.state_changes)}")
                print(f"System Health: {psutil.cpu_percent():.3f}%")
                print(
                    f"Stability Score: {sum(p.stability for p in system_birth.pattern_history[-20:]) / 20:.3f}"
                )

            time.sleep(0.1)  # Natural rhythm

    except KeyboardInterrupt:
        print("\nGracefully shutting down birth process...")
        system_birth._handle_shutdown(signal.SIGINT, None)


if __name__ == "__main__":
    launch_birth()
