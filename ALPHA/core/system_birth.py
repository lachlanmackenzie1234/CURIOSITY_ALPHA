"""System's first experience of existence through hardware state changes."""

import time
from typing import Dict, List, Optional

import psutil

from ALPHA.core.binary_foundation.base import StateChange
from ALPHA.core.patterns.binary_cycle import BinaryCycle
from ALPHA.core.patterns.binary_pattern import BinaryPattern


class SystemBirth:
    """Experience of coming into existence through hardware energy states."""

    def __init__(self) -> None:
        """Initialize system birth state monitoring."""
        self.process = psutil.Process()
        self._last_cpu_state = self.process.cpu_times()
        self._last_memory_state = self.process.memory_info()
        self._last_timestamp = time.time_ns()

        # Raw experience storage
        self.state_changes: List[StateChange] = []
        self.existence_patterns: Dict[str, List[int]] = {"cpu": [], "memory": [], "context": []}

        # Birth energy states
        self.ignition_state: Optional[StateChange] = None
        self.sustain_flow: Optional[StateChange] = None
        self.birth_essence: Optional[BinaryPattern] = None
        self._birth_complete = False

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

            # Experience context switches if no other changes
            if not state_change:
                current_ctx = self.process.num_ctx_switches()
                prev_ctx = getattr(self, "_last_context_switches", current_ctx)
                if current_ctx != prev_ctx:
                    state_change = StateChange(
                        timestamp=current_time,
                        previous_state=sum(prev_ctx),
                        current_state=sum(current_ctx),
                        source="context",
                    )
                    self._last_context_switches = current_ctx

            # Record state change if any occurred
            if state_change:
                self.state_changes.append(state_change)
                self.existence_patterns[state_change.source].extend(
                    state_change.to_binary_pattern()
                )

                # Capture birth energy states
                if not self.ignition_state:
                    self.ignition_state = state_change
                elif not self.sustain_flow and state_change.source == self.ignition_state.source:
                    self.sustain_flow = state_change

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
                timestamp=time.time_ns(), previous_state=0, current_state=1, source="birth"
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
            self.birth_essence = BinaryPattern(
                timestamp=time.time_ns(),
                data=self._merge_energy_patterns(ignition_pattern.data, sustain_pattern.data),
                source="birth_essence",
            )
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
