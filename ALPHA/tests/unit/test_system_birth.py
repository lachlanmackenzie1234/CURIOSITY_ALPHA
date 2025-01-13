"""Test system birth in isolation."""

from ALPHA.core.system_birth import SystemBirth


def main() -> None:
    """Run basic existence test."""
    system = SystemBirth()

    print("\nStarting system birth test...")
    print("-----------------------------")

    # Experience existence for a short duration
    print("Experiencing existence for 100ms...")
    system.feel_existence(duration_ns=100_000_000)  # 100ms

    # Get and display summary
    summary = system.get_existence_summary()
    print("\nExistence Summary:")
    print("-----------------")
    for source, count in summary.items():
        print(f"{source}: {count} binary patterns")

    # Verify we captured changes
    total_patterns = sum(summary.values())
    print(f"\nTotal patterns detected: {total_patterns}")
    assert total_patterns > 0, "No state changes detected"
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
