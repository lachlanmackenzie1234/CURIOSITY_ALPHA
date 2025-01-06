"""Demonstration of ALPHA autonomous capabilities."""

import random
import time

from .autonomous import AutonomousCore
from .base import Binary


def generate_input() -> bytes:
    """Generate random binary input."""
    binary = Binary()
    for i in range(16):
        binary.write(i, random.random() > 0.5)
    return binary.to_bytes()


def main():
    """Demonstrate autonomous evolution."""
    system = AutonomousCore()

    # Initial learning phase
    print("Initial Learning Phase:")
    for _ in range(5):
        data = generate_input()
        system.observe(data)
        state = system.get_state()
        print(f"Patterns: {state['patterns']}, Energy: {state['energy']:.2f}")

    # Autonomous evolution
    print("\nAutonomous Evolution:")
    for i in range(10):
        # Let system think and evolve
        if evolved_data := system.think():
            evolved = Binary.from_bytes(evolved_data)
            print(f"\nGeneration {i+1}:")
            print(f"Pattern: {evolved}")

            # Simulate environment feedback
            success = random.random()  # Random success rate
            system.adapt(success)

            state = system.get_state()
            print(f"Energy: {state['energy']:.2f}")
            print(f"Patterns: {state['patterns']}")
            print(f"Avg Strength: {state['avg_strength']:.2f}")

            # Optional: Add new observations
            if random.random() > 0.7:  # 30% chance
                system.observe(generate_input())
                print("New pattern observed")

        time.sleep(1)  # Pause to observe evolution


if __name__ == "__main__":
    main()
