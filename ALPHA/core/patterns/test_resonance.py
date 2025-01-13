"""Test script for system resonance detection."""

import time
from typing import List

import numpy as np

from .system_resonance import SystemOperation, SystemResonance


def print_operation(op: SystemOperation) -> None:
    """Print details of a system operation."""
    print(f"\nOperation Type: {op.operation_type}")
    print(f"Timestamp: {op.timestamp}")
    print(f"Energy Delta: {op.energy_delta:.6f}")
    print(f"Duration: {op.duration:.6f} ns")
    print("Metadata:", op.metadata)


def run_test_cycles(cycles: int = 3) -> None:
    """Run system resonance detection for specified number of cycles."""
    resonance = SystemResonance()

    print(f"\nRunning {cycles} system resonance cycles...")

    # Create some system load
    for cycle in range(cycles):
        print(f"\n=== Cycle {cycle + 1} ===")

        # Generate some CPU and memory activity
        array_size = np.random.randint(1000, 10000)
        _ = np.random.rand(array_size, array_size) @ np.random.rand(array_size, array_size)

        # Capture and analyze the cycle
        operations = resonance.capture_cycle()
        print(f"Detected {len(operations)} operations:")
        for op in operations:
            print_operation(op)

        # Add operations to history
        resonance.operations.extend(operations)

        # Check for patterns
        pattern = resonance.detect_patterns(window_size=max(2, len(resonance.operations)))
        if pattern:
            print("\nPattern detected!")
            print(f"Name: {pattern.name}")
            print(f"Confidence: {pattern.confidence:.3f}")
            print(f"Ratio: {pattern.ratio:.3f}")
            print("Properties:", pattern.properties)

        # Small delay between cycles
        time.sleep(0.1)


if __name__ == "__main__":
    run_test_cycles()
