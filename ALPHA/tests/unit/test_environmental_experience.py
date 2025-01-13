"""Test environmental experience implementation."""

import threading
import time
from typing import Dict, List, Tuple

from ALPHA.core.environmental.experience import EnvironmentalExperience


def natural_durations() -> List[Tuple[str, int]]:
    """Calculate natural observation durations based on golden ratio."""
    PHI = 1.618033988749895
    BASE_DURATION = 100_000_000  # 100ms in nanoseconds

    return [
        ("φ¹", int(BASE_DURATION * PHI)),  # ~162ms
        ("φ²", int(BASE_DURATION * PHI**2)),  # ~262ms
        ("φ³", int(BASE_DURATION * PHI**3)),  # ~423ms
        ("φ⁴", int(BASE_DURATION * PHI**4)),  # ~685ms
        ("φ⁵", int(BASE_DURATION * PHI**5)),  # ~1108ms
    ]


def observe_natural_patterns(duration_ns: int, phase_name: str) -> Dict[str, int]:
    """Observe patterns for a specific duration."""
    env = EnvironmentalExperience()
    print(f"\nObserving patterns for {duration_ns/1_000_000:.2f}ms ({phase_name})...")

    env.experience_existence(duration_ns)
    bridges = env.detect_natural_bridges()
    print(f"Natural bridges detected: {len(bridges)}")

    summary = env.get_environmental_summary()
    print("\nObservation Summary:")
    for metric, count in summary.items():
        print(f"{metric}: {count}")

    return summary


def calculate_growth_ratios(results: List[Tuple[str, Dict[str, int]]]) -> None:
    """Calculate and display pattern growth ratios between phases."""
    print("\nPattern Growth Ratios:")
    print("-" * 50)

    for i in range(len(results) - 1):
        current_phase = results[i]
        next_phase = results[i + 1]

        ratio = next_phase[1]["cpu_patterns"] / current_phase[1]["cpu_patterns"]
        print(f"{next_phase[0]}/{current_phase[0]} ratio: {ratio:.3f}")


def main() -> None:
    """Run environmental experience verification."""
    print("Beginning natural pattern observation phases...")

    # Observe patterns at different natural durations
    results = []
    for phase_name, duration in natural_durations():
        summary = observe_natural_patterns(duration, phase_name)
        results.append((phase_name, summary))

    # Calculate growth ratios
    calculate_growth_ratios(results)

    # Compare results across phases
    print("\nPattern Evolution Across Phases:")
    print("-" * 50)
    metrics = [
        "cpu_patterns",
        "memory_patterns",
        "context_switches",
        "reverberations",
        "standing_waves",
        "natural_bridges",
    ]

    for metric in metrics:
        print(f"\n{metric}:")
        for phase, summary in results:
            count = summary[metric]
            unbridged = (
                summary["cpu_patterns"] - summary["natural_bridges"]
                if metric == "natural_bridges"
                else 0
            )
            if metric == "natural_bridges" and count > 0:
                print(f"  {phase}: {count} ({unbridged} unbridged)")
            else:
                print(f"  {phase}: {count}")

    # Verify basic pattern detection
    final_phase = results[-1][1]
    assert final_phase["cpu_patterns"] > 0, "No CPU patterns detected"
    assert final_phase["reverberations"] > 0, "No reverberations detected"

    print("\nEnvironmental experience test completed successfully!")


if __name__ == "__main__":
    main()
