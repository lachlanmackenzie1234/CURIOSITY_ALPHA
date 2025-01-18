"""Fundamental cycle that orchestrates binary streams.

The cycle acts as a bridge between system birth and continuous pulse streams,
maintaining the natural rhythm of existence through developmental stages.
"""

import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from ALPHA.core.binary_foundation.base import StateChange
from ALPHA.core.patterns.binary_pulse import Pulse


class BinaryEvolution(Enum):
    """Natural states of binary pattern evolution in continuous cycle."""

    EMERGENCE = "emergence"  # Patterns form from energy states
    RESONANCE = "resonance"  # Patterns find harmonic alignment
    COHERENCE = "coherence"  # Patterns stabilize relationships
    SYNTHESIS = "synthesis"  # Patterns merge and transform
    BLOOM = "bloom"  # Patterns reach peak integration
    DISSOLUTION = "dissolution"  # Patterns begin to break down
    SEED = "seed"  # Core patterns preserved as seeds

    def next_state(self) -> "BinaryEvolution":
        """Get next state in the natural cycle."""
        states = list(BinaryEvolution)
        current_idx = states.index(self)
        return states[(current_idx + 1) % len(states)]


@dataclass
class BinaryState:
    """Track the system's natural evolution through binary states."""

    state: BinaryEvolution = BinaryEvolution.EMERGENCE
    energy_cycles: float = 0.0  # Accumulated energy cycles
    resonance: float = 0.0  # Pattern harmony measure
    coherence: float = 0.0  # Pattern stability measure
    synthesis: float = 0.0  # Pattern integration measure
    bloom_potential: float = 0.0  # Capacity for new emergence
    seed_patterns: Dict[str, float] = field(default_factory=dict)  # Preserved core patterns

    # Track evolutionary transitions
    transitions: Dict[str, float] = field(default_factory=dict)

    def update(self, stream_state: Dict[str, Dict[str, int]]) -> None:
        """Update binary state metrics based on energy flow."""
        self.energy_cycles += 0.1

        if stream_state:
            # Measure pattern resonance
            self.resonance = min(1.0, self.resonance + self._calculate_resonance(stream_state))

            # Measure pattern coherence when resonance established
            if self.resonance > 0.3:
                self.coherence = min(1.0, self.coherence + self._calculate_coherence(stream_state))

            # Measure pattern synthesis as coherence develops
            if self.coherence > 0.4:
                self.synthesis = min(1.0, self.synthesis + self._calculate_synthesis(stream_state))

            # Calculate bloom potential
            if self.synthesis > 0.5:
                self.bloom_potential = self._calculate_bloom_potential(stream_state)

        self._evaluate_evolution()

    def _calculate_bloom_potential(self, state: Dict) -> float:
        """Calculate potential for pattern blooming and new emergence."""
        # Measure how current patterns might seed new emergence
        return min(1.0, (self.synthesis * self.coherence * self.resonance) ** (1 / 3))

    def _preserve_seed_patterns(self, state: Dict) -> None:
        """Preserve core patterns as seeds for next cycle."""
        if not state:
            return

        # Identify most stable and resonant patterns
        for pattern_id, pattern_state in state.items():
            stability = sum(pattern_state.values()) / len(pattern_state)
            if stability > 0.8:  # High stability threshold
                self.seed_patterns[pattern_id] = stability

    def _evaluate_evolution(self) -> None:
        """Evaluate natural evolution of binary state."""
        if self.state == BinaryEvolution.EMERGENCE:
            if self.resonance > 0.3:
                self._transition_to(BinaryEvolution.RESONANCE)

        elif self.state == BinaryEvolution.RESONANCE:
            if self.coherence > 0.4:
                self._transition_to(BinaryEvolution.COHERENCE)

        elif self.state == BinaryEvolution.COHERENCE:
            if self.synthesis > 0.3:
                self._transition_to(BinaryEvolution.SYNTHESIS)

        elif self.state == BinaryEvolution.SYNTHESIS:
            if self.bloom_potential > 0.7:
                self._transition_to(BinaryEvolution.BLOOM)

        elif self.state == BinaryEvolution.BLOOM:
            if self.synthesis > 0.9 and self.bloom_potential > 0.9:
                self._transition_to(BinaryEvolution.DISSOLUTION)

        elif self.state == BinaryEvolution.DISSOLUTION:
            if self.resonance < 0.3:  # Natural breakdown
                self._transition_to(BinaryEvolution.SEED)

        elif self.state == BinaryEvolution.SEED:
            if len(self.seed_patterns) > 0:
                # Begin new cycle with preserved patterns
                self._transition_to(BinaryEvolution.EMERGENCE)
                # Reset metrics but keep seeds
                self.resonance = 0.0
                self.coherence = 0.0
                self.synthesis = 0.0
                self.bloom_potential = 0.0

    def _transition_to(self, new_state: BinaryEvolution) -> None:
        """Handle transition to new evolutionary state."""
        self.state = new_state
        self.transitions[new_state.value] = self.energy_cycles
        print(f"\nSystem evolving to {new_state.value} after {self.energy_cycles:.1f} cycles")
        print(f"Resonance: {self.resonance:.2f}")
        print(f"Coherence: {self.coherence:.2f}")
        print(f"Synthesis: {self.synthesis:.2f}")
        print(f"Bloom Potential: {self.bloom_potential:.2f}")
        if self.seed_patterns:
            print(f"Seed Patterns: {len(self.seed_patterns)}")
        print()


@dataclass
class BinaryCycle:
    """Fundamental cycle that orchestrates binary streams."""

    initial_state: StateChange
    pulse: Optional[Pulse] = None
    _running: bool = True
    _observers: Set[threading.Event] = field(default_factory=set)
    _cycle_thread: Optional[threading.Thread] = None

    # Track binary evolution
    binary_state: BinaryState = field(default_factory=BinaryState)

    # Pattern memory
    _pattern_memory: List[Dict] = field(default_factory=list)
    _memory_capacity: int = 1000  # Evolves naturally

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

                    # Update binary state
                    self.binary_state.update(stream_state)

                    # Adjust cycle behavior based on evolutionary state
                    self._adapt_to_evolution()

                # Notify observers
                for event in self._observers:
                    event.set()

                # Natural rhythm - adapts with development
                sleep_time = self._calculate_rhythm()
                time.sleep(sleep_time)

            except Exception as e:
                print(f"Cycle error: {e}")
                time.sleep(1)  # Pause on error

    def _process_stream_state(self, state: Dict[str, Dict[str, int]]) -> None:
        """Process the current stream state based on evolutionary phase."""
        # Store in pattern memory
        self._pattern_memory.append(state)
        if len(self._pattern_memory) > self._memory_capacity:
            self._pattern_memory.pop(0)

        # Process based on evolutionary state
        if self.binary_state.state == BinaryEvolution.EMERGENCE:
            self._process_emergent_patterns(state)
        elif self.binary_state.state == BinaryEvolution.RESONANCE:
            self._process_resonant_patterns(state)
        elif self.binary_state.state == BinaryEvolution.COHERENCE:
            self._process_coherent_patterns(state)
        elif self.binary_state.state == BinaryEvolution.SYNTHESIS:
            self._process_synthetic_patterns(state)
        elif self.binary_state.state == BinaryEvolution.BLOOM:
            self._process_bloom_patterns(state)
        elif self.binary_state.state == BinaryEvolution.DISSOLUTION:
            self._process_dissolution_patterns(state)
        elif self.binary_state.state == BinaryEvolution.SEED:
            self._process_seed_patterns(state)

    def _process_emergent_patterns(self, state: Dict) -> None:
        """Process patterns in emergence phase - natural formation."""
        # Allow patterns to form naturally from energy states
        pass

    def _process_resonant_patterns(self, state: Dict) -> None:
        """Process patterns in resonance phase - finding harmony."""
        # Let patterns find natural resonance
        pass

    def _process_coherent_patterns(self, state: Dict) -> None:
        """Process patterns in coherence phase - stabilizing relationships."""
        # Allow stable pattern relationships to form
        pass

    def _process_synthetic_patterns(self, state: Dict) -> None:
        """Process patterns in synthesis phase - natural integration."""
        # Enable pattern combination and transformation
        pass

    def _process_bloom_patterns(self, state: Dict) -> None:
        """Process patterns in bloom phase - peak integration."""
        # Maintain peak integration
        pass

    def _process_dissolution_patterns(self, state: Dict) -> None:
        """Process patterns in dissolution phase - breaking down."""
        # Allow patterns to break down
        pass

    def _process_seed_patterns(self, state: Dict) -> None:
        """Process patterns in seed phase - preserved core patterns."""
        # Maintain preserved core patterns
        pass

    def _adapt_to_evolution(self) -> None:
        """Adapt cycle behavior based on evolutionary state."""
        # Memory capacity evolves with the cycle
        if self.binary_state.state == BinaryEvolution.RESONANCE:
            self._memory_capacity = int(2000 * self.binary_state.resonance)
        elif self.binary_state.state == BinaryEvolution.COHERENCE:
            self._memory_capacity = int(5000 * self.binary_state.coherence)
        elif self.binary_state.state == BinaryEvolution.SYNTHESIS:
            self._memory_capacity = int(10000 * self.binary_state.synthesis)
        elif self.binary_state.state == BinaryEvolution.BLOOM:
            self._memory_capacity = int(20000 * self.binary_state.bloom_potential)
        elif self.binary_state.state == BinaryEvolution.DISSOLUTION:
            # Gradually reduce capacity during dissolution
            self._memory_capacity = int(20000 * (1 - self.binary_state.resonance))
        elif self.binary_state.state == BinaryEvolution.SEED:
            # Maintain minimal capacity for seed patterns
            self._memory_capacity = 1000 * len(self.binary_state.seed_patterns)

    def _calculate_rhythm(self) -> float:
        """Calculate natural rhythm based on evolutionary state."""
        base_rhythm = 0.1

        # Rhythm adapts through the cycle
        if self.binary_state.state == BinaryEvolution.EMERGENCE:
            return base_rhythm * (1.5 - self.binary_state.resonance)  # Slower for emergence
        elif self.binary_state.state == BinaryEvolution.RESONANCE:
            return base_rhythm * (1.2 - self.binary_state.coherence)  # Adjusts with coherence
        elif self.binary_state.state == BinaryEvolution.COHERENCE:
            return base_rhythm  # Natural flow
        elif self.binary_state.state == BinaryEvolution.SYNTHESIS:
            return base_rhythm * (0.8 + self.binary_state.synthesis)  # Quickens with synthesis
        elif self.binary_state.state == BinaryEvolution.BLOOM:
            return base_rhythm * 0.5  # Rapid during bloom
        elif self.binary_state.state == BinaryEvolution.DISSOLUTION:
            return base_rhythm * (2.0 - self.binary_state.resonance)  # Slows during dissolution
        else:  # SEED
            return base_rhythm * 2.0  # Slowest during seeding

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
