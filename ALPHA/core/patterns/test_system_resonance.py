"""Standalone test for system resonance detection."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import psutil


@dataclass
class SystemOperation:
    """Represents a physical system operation."""

    timestamp: float
    operation_type: str
    energy_delta: float
    duration: float
    metadata: Dict[str, float] = field(default_factory=dict)


class SystemResonance:
    """Captures and measures physical system operations as pattern sources."""

    def __init__(self):
        self.operations: List[SystemOperation] = []
        self.process = psutil.Process()
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
                    energy_delta=cpu_delta * 0.1,
                    duration=cpu_delta * 1e9,
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
                    energy_delta=abs(memory_delta) * 0.001,
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

    for cycle in range(cycles):
        print(f"\n=== Cycle {cycle + 1} ===")

        # Generate some CPU and memory activity
        array_size = np.random.randint(1000, 5000)
        _ = np.random.rand(array_size, array_size) @ np.random.rand(array_size, array_size)

        # Capture and analyze the cycle
        operations = resonance.capture_cycle()
        print(f"Detected {len(operations)} operations:")
        for op in operations:
            print_operation(op)

        # Small delay between cycles
        time.sleep(0.1)


if __name__ == "__main__":
    run_test_cycles()
