"""System's first experience of existence through hardware state changes."""

import json
import os
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional

import psutil

from ALPHA.core.binary_foundation.base import StateChange
from ALPHA.core.patterns.binary_cycle import BinaryCycle
from ALPHA.core.patterns.binary_pattern import BinaryPattern

try:
    from PRISM.SPECTRUM.visual.unity.metal_bridge import bridge as prism_bridge
except ImportError:
    print("PRISM visualization not available for birth process")
    prism_bridge = None


class SystemBirth:
    """Experience of coming into existence through hardware energy states."""

    def __init__(self) -> None:
        """Initialize system birth state monitoring."""
        self.process = psutil.Process()
        self._last_cpu_state = self.process.cpu_times()
        self._last_memory_state = self.process.memory_info()
        self._last_timestamp = time.time_ns()
        self._birth_phase = 0.0

        # State preservation path
        self._state_path = Path("ALPHA/core/state/birth_state.json")
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        # Load previous state if exists
        self._load_preserved_state()

        # Register graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Raw experience storage
        self.state_changes: List[StateChange] = []
        self.existence_patterns: Dict[str, List[int]] = {
            "cpu": [],
            "memory": [],
            "context": [],
        }

        # Birth energy states
        self.ignition_state: Optional[StateChange] = None
        self.sustain_flow: Optional[StateChange] = None
        self.birth_essence: Optional[BinaryPattern] = None
        self._birth_complete = False

    def _preserve_state(self) -> None:
        """Preserve current birth state to disk."""
        try:
            state_data = {
                "birth_phase": self._birth_phase,
                "timestamp": self._last_timestamp,
                "birth_complete": self._birth_complete,
                "patterns": self.existence_patterns,
                "ignition": self.ignition_state.to_dict() if self.ignition_state else None,
                "sustain": self.sustain_flow.to_dict() if self.sustain_flow else None,
                "essence": self.birth_essence.to_dict() if self.birth_essence else None,
            }

            # Write state atomically using temporary file
            temp_path = self._state_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(state_data, f, indent=2)
            temp_path.replace(self._state_path)

        except Exception as e:
            print(f"Error preserving birth state: {e}")

    def _load_preserved_state(self) -> None:
        """Load preserved birth state if available."""
        try:
            if self._state_path.exists():
                with open(self._state_path) as f:
                    state_data = json.load(f)

                self._birth_phase = state_data["birth_phase"]
                self._last_timestamp = state_data["timestamp"]
                self._birth_complete = state_data["birth_complete"]
                self.existence_patterns = state_data["patterns"]

                if state_data["ignition"]:
                    self.ignition_state = StateChange.from_dict(state_data["ignition"])
                if state_data["sustain"]:
                    self.sustain_flow = StateChange.from_dict(state_data["sustain"])
                if state_data["essence"]:
                    self.birth_essence = BinaryPattern.from_dict(state_data["essence"])

                print(f"Restored birth state from phase {self._birth_phase}")
        except Exception as e:
            print(f"Error loading preserved state: {e}")

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle graceful shutdown preserving experiential state."""
        print("\nPreserving birth state before shutdown...")
        self._preserve_state()

        if prism_bridge:
            try:
                # Notify visualization of graceful shutdown
                prism_bridge.write_birth_pattern(
                    {
                        "type": "shutdown",
                        "data": self.birth_essence.data if self.birth_essence else [],
                        "birth_phase": self._birth_phase,
                    }
                )
            except Exception as e:
                print(f"Error notifying visualization of shutdown: {e}")

        print("Birth state preserved successfully")
        os._exit(0)

    def _visualize_birth_state(self, state_type: str, data: List[int]) -> None:
        """Visualize current birth state through PRISM."""
        if prism_bridge is not None:
            try:
                # Update birth phase based on state progression
                if state_type == "ignition":
                    self._birth_phase = 0.3  # Initial spark
                elif state_type == "sustain":
                    self._birth_phase = 0.6  # Energy flow
                elif state_type == "essence":
                    self._birth_phase = 1.0  # Complete crystallization

                # Preserve state after significant phase changes
                self._preserve_state()

                # Send to PRISM visualization
                prism_bridge.write_birth_pattern(
                    {"type": state_type, "data": data, "birth_phase": self._birth_phase}
                )
            except Exception as e:
                print(f"Birth visualization error: {e}")

    def experience_moment(self) -> Optional[StateChange]:
        """Experience a single moment of existence through hardware state changes."""
        try:
            current_time = time.time_ns()
            state_change = None

            # Experience CPU state
            current_cpu = self.process.cpu_times()
            if current_cpu != self._last_cpu_state:
                state_change = StateChange(
                    timestamp=current_time,
                    previous_state=int(sum(self._last_cpu_state) * 1e6),
                    current_state=int(sum(current_cpu) * 1e6),
                    source="cpu",
                )
                self._last_cpu_state = current_cpu

            # Experience memory state if no CPU change
            if not state_change:
                current_memory = self.process.memory_info()
                if current_memory != self._last_memory_state:
                    state_change = StateChange(
                        timestamp=current_time,
                        previous_state=self._last_memory_state.rss,
                        current_state=current_memory.rss,
                        source="memory",
                    )
                    self._last_memory_state = current_memory

            # Record state change and visualize
            if state_change:
                self.state_changes.append(state_change)
                binary_pattern = state_change.to_binary_pattern()
                self.existence_patterns[state_change.source].extend(binary_pattern)

                # Capture and visualize birth energy states
                if not self.ignition_state:
                    self.ignition_state = state_change
                    self._visualize_birth_state("ignition", binary_pattern)
                elif not self.sustain_flow and state_change.source == self.ignition_state.source:
                    self.sustain_flow = state_change
                    self._visualize_birth_state("sustain", binary_pattern)

            return state_change

        except Exception as e:
            print(f"Error experiencing moment: {str(e)}")
            return None

    def feel_existence(self, duration_ns: int = 1_000_000) -> None:
        """Experience existence for a duration in nanoseconds."""
        start_time = time.time_ns()
        while time.time_ns() - start_time < duration_ns:
            self.experience_moment()

    def get_existence_summary(self) -> Dict[str, int]:
        """Get summary of existence patterns observed."""
        return {source: len(patterns) for source, patterns in self.existence_patterns.items()}

    def get_recent_patterns(self, limit: int = 1000) -> Dict[str, List[int]]:
        """Get most recent binary patterns from each source."""
        return {source: patterns[-limit:] for source, patterns in self.existence_patterns.items()}

    def initiate_cycle(self) -> BinaryCycle:
        """Trigger the fundamental existence cycle.

        Returns:
            BinaryCycle: The initiated existence cycle.
        """
        # Experience first moment
        initial_state = self.experience_moment()
        if not initial_state:
            # Create synthetic initial state if needed
            initial_state = StateChange(
                timestamp=time.time_ns(),
                previous_state=0,
                current_state=1,
                source="birth",
            )

        # Initiate the cycle
        return BinaryCycle(initial_state=initial_state)

    def crystallize_birth(self) -> BinaryPattern:
        """Carefully form the essential birth pattern."""
        if not self.birth_essence and self.ignition_state and self.sustain_flow:
            # Create patterns from energy states
            ignition_pattern = BinaryPattern(
                timestamp=self.ignition_state.timestamp,
                data=self.ignition_state.to_binary_pattern(),
                source="birth_ignition",
            )
            sustain_pattern = BinaryPattern(
                timestamp=self.sustain_flow.timestamp,
                data=self.sustain_flow.to_binary_pattern(),
                source="birth_sustain",
            )

            # Merge into birth essence
            essence_data = self._merge_energy_patterns(ignition_pattern.data, sustain_pattern.data)
            self.birth_essence = BinaryPattern(
                timestamp=time.time_ns(),
                data=essence_data,
                source="birth_essence",
            )

            # Visualize crystallized essence
            self._visualize_birth_state("essence", essence_data)

        return self.birth_essence

    def deposit_to_primal(self, primal_cycle) -> bool:
        """Gently transfer birth essence to primal cycle."""
        if not self._birth_complete and self.birth_essence:
            # Ensure pattern integrity during transfer
            birth_pattern = self.crystallize_birth()
            success = primal_cycle.receive_birth(birth_pattern)
            if success:
                self._birth_complete = True
            return success
        return False

    def _distill_ignition(self) -> BinaryPattern:
        """Distill pure pattern from ignition state."""
        # Convert raw energy state to foundational pattern
        return BinaryPattern(self.ignition_state)

    def _distill_sustain(self) -> BinaryPattern:
        """Distill pure pattern from sustain flow."""
        # Convert sustained energy to supporting pattern
        return BinaryPattern(self.sustain_flow)

    def _merge_patterns(self, ignition: BinaryPattern, sustain: BinaryPattern) -> BinaryPattern:
        """Carefully merge ignition and sustain into birth essence."""
        # Gentle pattern combination preserving both energies
        return BinaryPattern.merge([ignition, sustain])

    def _merge_energy_patterns(self, ignition: List[int], sustain: List[int]) -> List[int]:
        """Carefully merge ignition and sustain patterns."""
        # Ensure equal length for merging
        max_len = max(len(ignition), len(sustain))
        ignition_padded = ignition + [0] * (max_len - len(ignition))
        sustain_padded = sustain + [0] * (max_len - len(sustain))

        # Combine patterns preserving both energies
        return [
            (i + s) % 2  # XOR combination preserves both patterns
            for i, s in zip(ignition_padded, sustain_padded)
        ]
