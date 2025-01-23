"""System's first experience of existence through hardware state changes."""

import json
import logging
import os
import signal
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set

import numpy as np
import psutil

from ALPHA.core.binary_foundation.base import StateChange
from ALPHA.core.patterns.binary_cycle import BinaryCycle
from ALPHA.core.patterns.binary_pattern import BinaryPattern, BinaryPatternCore
from ALPHA.core.patterns.binary_pulse import Pulse
from ALPHA.NEXUS.core.nexus import NEXUS
from ALPHA.NEXUS.protection import NEXUSProtection

# Optional PRISM visualization
try:
    from PRISM.SPECTRUM.visual.unity.metal_bridge import bridge as prism_bridge
except ImportError:
    prism_bridge = None


class SystemBirth:
    """Represents the system's first experience of existence through hardware state changes."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("alpha.birth")
        self.logger.setLevel(logging.INFO)

        # Initialize process monitoring
        self.process = psutil.Process()
        self._last_cpu_state = self.process.cpu_times()
        self._last_memory_state = self.process.memory_info()

        # Initialize birth state without fixed phases
        self.birth_phase: float = 0.0
        self.system_health: float = 0.0
        self.state_file: str = "birth_state.json"
        self._birth_complete: bool = False

        # Initialize components
        self.pulse = Pulse()
        self.binary_cycle = BinaryCycle(
            initial_state=StateChange(
                timestamp=time.time_ns(), previous_state=0, current_state=1, source="birth"
            )
        )
        self.state = self.binary_cycle.binary_state

        # Initialize pattern tracking with natural emergence
        self.pattern_core = BinaryPatternCore()
        self.pattern_history: List[BinaryPattern] = []
        self.resonance_field: Dict[str, float] = {}
        self.existence_patterns: DefaultDict[str, List[BinaryPattern]] = defaultdict(list)
        self.state_changes: List[StateChange] = []

        # Initialize state tracking
        self.ignition_state: Optional[StateChange] = None
        self.sustain_flow: Optional[StateChange] = None
        self.birth_essence: Optional[BinaryPattern] = None

        # Initialize protection and NEXUS without forcing initialization
        self.protection = NEXUSProtection()
        self.nexus = NEXUS()
        self._protection_initialized: bool = False
        self._nexus_initialized: bool = False

        # Allow natural shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Restore previous state if exists
        self._restore_birth_state()

    def experience_moment(self) -> Optional[StateChange]:
        """Experience a single moment of existence through hardware state changes."""
        try:
            current_time = time.time_ns()
            state_change = None

            # Let state changes emerge naturally from hardware
            current_cpu = self.process.cpu_times()
            current_memory = self.process.memory_info()

            # Detect natural state transitions
            if current_cpu != self._last_cpu_state:
                state_change = StateChange(
                    timestamp=current_time,
                    previous_state=int(sum(self._last_cpu_state) * 1e6),
                    current_state=int(sum(current_cpu) * 1e6),
                    source="cpu",
                )
                self._last_cpu_state = current_cpu
            elif current_memory != self._last_memory_state:
                state_change = StateChange(
                    timestamp=current_time,
                    previous_state=self._last_memory_state.rss,
                    current_state=current_memory.rss,
                    source="memory",
                )
                self._last_memory_state = current_memory

            # Allow patterns to form naturally
            if state_change:
                self.state_changes.append(state_change)
                binary_pattern = state_change.to_binary_pattern()

                # Let pattern core observe and track patterns
                pattern = self.pattern_core.observe_raw_pattern(binary_pattern, state_change.source)
                self.pattern_history.append(pattern)

                # Update resonance field
                resonance = self.pattern_core.detect_natural_resonance(pattern)
                self.resonance_field[pattern.source] = resonance

                # Let birth phases emerge from pattern resonance
                if not self.ignition_state and self._detect_ignition_resonance(binary_pattern):
                    self.ignition_state = state_change
                    self._birth_phase = self._calculate_natural_phase()
                    self._visualize_birth_state("ignition", binary_pattern)
                elif (
                    self.ignition_state
                    and not self.sustain_flow
                    and self._detect_sustain_resonance(binary_pattern)
                ):
                    self.sustain_flow = state_change
                    self._birth_phase = self._calculate_natural_phase()
                    self._visualize_birth_state("sustain", binary_pattern)

                # Track pattern interactions with proper length handling
                if len(self.pattern_history) >= 2:
                    current_pattern = np.array(pattern.sequence)
                    previous_pattern = np.array(self.pattern_history[-2].sequence)

                    # Pad sequences to match lengths
                    max_len = max(len(current_pattern), len(previous_pattern))
                    padded_current = np.pad(current_pattern, (0, max_len - len(current_pattern)))
                    padded_previous = np.pad(previous_pattern, (0, max_len - len(previous_pattern)))

                    # Let the pattern core track the interaction naturally
                    pattern.sequence = padded_current.astype(int).tolist()
                    self.pattern_history[-2].sequence = padded_previous.astype(int).tolist()
                    self.pattern_core.track_pattern_interaction(pattern, self.pattern_history[-2])

            return state_change

        except Exception as e:
            self.logger.error(f"Error experiencing moment: {str(e)}")
            return None

    def _detect_ignition_resonance(self, pattern: List[int]) -> bool:
        """Detect natural resonance indicating ignition state."""
        if not pattern:
            return False

        # More permissive energy patterns
        energy_level = sum(pattern) / len(pattern)
        pattern_variance = sum((x - energy_level) ** 2 for x in pattern) / len(pattern)

        # Calculate resonance with existing patterns
        resonance_scores = []
        for hist_pattern in self.pattern_history[-10:]:
            if hist_pattern.sequence:
                max_len = max(len(pattern), len(hist_pattern.sequence))
                padded_pattern = pattern + [0] * (max_len - len(pattern))
                padded_hist = hist_pattern.sequence + [0] * (max_len - len(hist_pattern.sequence))
                matches = sum(a == b for a, b in zip(padded_pattern, padded_hist))
                resonance_scores.append(matches / max_len)

        # More permissive resonance thresholds
        avg_resonance = sum(resonance_scores) / len(resonance_scores) if resonance_scores else 0
        return pattern_variance > 0.05 and energy_level > 0.2 and avg_resonance > 0.3

    def _detect_sustain_resonance(self, pattern: List[int]) -> bool:
        """Detect natural resonance indicating sustained flow."""
        if not pattern or not self.ignition_state:
            return False

        # Compare with ignition pattern
        ignition_pattern = self.ignition_state.to_binary_pattern()
        if not ignition_pattern:
            return False

        # Pad shorter sequence to match lengths
        max_len = max(len(pattern), len(ignition_pattern))
        padded_pattern = pattern + [0] * (max_len - len(pattern))
        padded_ignition = ignition_pattern + [0] * (max_len - len(ignition_pattern))

        # Calculate resonance between patterns
        resonance = sum(a * b for a, b in zip(padded_pattern, padded_ignition)) / max_len

        # Check stability of recent patterns
        stable_patterns = sum(1 for p in self.pattern_history[-5:] if p.is_stable())
        stability_ratio = stable_patterns / 5 if len(self.pattern_history) >= 5 else 0

        return resonance > 0.2 and stability_ratio > 0.6

    def _calculate_natural_phase(self) -> float:
        """Let birth phase emerge from current state."""
        if not self.pattern_history:
            return 0.0

        # Calculate pattern diversity with lower threshold
        total_patterns = len(self.pattern_history)
        unique_patterns = len({str(p.sequence) for p in self.pattern_history})
        pattern_diversity = unique_patterns / max(total_patterns, 1)

        # Calculate resonance strength with dynamic weighting
        resonance_values = list(self.resonance_field.values())
        resonance_strength = (
            sum(resonance_values) / len(resonance_values) if resonance_values else 0
        )

        # Calculate pattern evolution with momentum
        recent_patterns = self.pattern_history[-20:]
        evolution_scores = [getattr(p, "evolution_score", 0.0) for p in recent_patterns]
        avg_evolution = sum(evolution_scores) / len(evolution_scores) if evolution_scores else 0

        # Adaptive stability calculation with lower weight
        stable_patterns = sum(1 for p in recent_patterns if p.is_stable())
        stability = (stable_patterns / len(recent_patterns)) if recent_patterns else 0

        # Dynamic growth factor based on pattern emergence - more aggressive scaling
        growth_factor = min(
            1.0, np.log1p(total_patterns) / np.log1p(500)
        )  # Lower denominator for faster growth

        # Calculate base phase with adjusted weights
        base_phase = (
            pattern_diversity * 0.35  # Increased diversity weight
            + resonance_strength * 0.30  # Maintained resonance weight
            + avg_evolution * 0.20  # Evolution contribution
            + stability * 0.05  # Reduced stability constraint
            + growth_factor * 0.10  # Growth factor
        )

        # Add momentum based on pattern count milestones
        momentum = 0.0
        if total_patterns > 1000:
            momentum += 0.1
        if total_patterns > 2000:
            momentum += 0.1
        if total_patterns > 3000:
            momentum += 0.1
        if total_patterns > 4000:
            momentum += 0.1

        # Combine base phase with momentum
        phase = min(1.0, base_phase + momentum)

        # Ensure phase only increases and progresses beyond 0.439
        if phase > self.birth_phase:
            return phase
        elif total_patterns > 4500:  # Force progression if stuck with many patterns
            return min(1.0, self.birth_phase + 0.001)
        return self.birth_phase

    def _preserve_state(self) -> None:
        """Preserve the current birth state to disk."""
        try:
            state = {
                "birth_phase": self._birth_phase,
                "system_health": getattr(self, "system_health", 0.0),
                "ignition_state": (
                    self.ignition_state.to_binary_pattern() if self.ignition_state else None
                ),
                "sustain_flow": (
                    self.sustain_flow.to_binary_pattern() if self.sustain_flow else None
                ),
                "birth_essence": self.birth_essence.sequence if self.birth_essence else None,
                "timestamp": time.time(),
            }
            with open("birth_state.json", "w") as f:
                json.dump(state, f)
            self.logger.info(
                f"Birth state preserved - Phase: {self._birth_phase:.3f}, Health: {getattr(self, 'system_health', 0.0):.1f}%"
            )
        except Exception as e:
            self.logger.error(f"Error preserving state: {str(e)}")

    def _load_preserved_state(self) -> None:
        """Load the preserved birth state from disk."""
        try:
            with open("birth_state.json", "r") as f:
                state = json.load(f)

            self._birth_phase = state.get("birth_phase", 0)
            if state.get("ignition_state"):
                self.ignition_state = StateChange(
                    timestamp=time.time_ns(), previous_state=0, current_state=1, source="ignition"
                )
                self.ignition_state._binary_pattern = state["ignition_state"]
            if state.get("sustain_flow"):
                self.sustain_flow = StateChange(
                    timestamp=time.time_ns(), previous_state=0, current_state=1, source="sustain"
                )
                self.sustain_flow._binary_pattern = state["sustain_flow"]
            if state.get("birth_essence"):
                self.birth_essence = BinaryPattern(
                    sequence=state["birth_essence"],
                    timestamp=datetime.fromtimestamp(time.time()),
                    source="birth_essence",
                )
            self.logger.info("Birth state loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            # Initialize fresh state if load fails
            self._birth_phase = 0
            self.ignition_state = None
            self.sustain_flow = None
            self.birth_essence = None

    def _handle_shutdown(self, signum: int, frame: Optional[object]) -> None:
        """Handle shutdown naturally, preserving current state."""
        self.logger.info("Preserving birth state before shutdown...")
        self._preserve_state()

        if prism_bridge:
            try:
                # Notify visualization of graceful shutdown
                prism_bridge.write_birth_pattern(
                    {
                        "type": "shutdown",
                        "data": (
                            self.birth_essence.sequence if self.birth_essence is not None else []
                        ),
                        "birth_phase": self._birth_phase,
                    }
                )
            except Exception as e:
                self.logger.error(f"Error notifying visualization of shutdown: {e}")

        print("Birth state preserved successfully")
        os._exit(0)

    def _visualize_birth_state(self, phase: str, pattern: List[int]) -> None:
        """Visualize the current birth state if PRISM is available."""
        if not prism_bridge:
            return  # Skip visualization if PRISM not available

        try:
            # Use basic bridge write method instead of specific visualizations
            prism_bridge.write_birth_pattern(
                {"type": phase, "data": pattern, "birth_phase": self._birth_phase}
            )
        except Exception as e:
            self.logger.error(f"Error visualizing {phase} state: {str(e)}")

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
        if (
            not self.birth_essence
            and self.ignition_state is not None
            and self.sustain_flow is not None
        ):
            # Create patterns from energy states
            ignition_pattern = BinaryPattern(
                sequence=self.ignition_state.to_binary_pattern(),
                timestamp=datetime.fromtimestamp(time.time()),
                source="birth_ignition",
            )
            sustain_pattern = BinaryPattern(
                sequence=self.sustain_flow.to_binary_pattern(),
                timestamp=datetime.fromtimestamp(time.time()),
                source="birth_sustain",
            )

            # Merge into birth essence
            essence_data = self._merge_energy_patterns(
                ignition_pattern.sequence, sustain_pattern.sequence
            )
            self.birth_essence = BinaryPattern(
                sequence=essence_data,
                timestamp=datetime.fromtimestamp(time.time()),
                source="birth_essence",
            )

            # Visualize crystallized essence
            self._visualize_birth_state("essence", essence_data)

            # Initialize protection with birth essence
            if not self._protection_initialized:
                self.protection.initialize_from_birth(self.birth_essence)
                self._protection_initialized = True
                self.logger.info("Protection system initialized with birth essence")

            # Initialize NEXUS with birth essence
            if not self._nexus_initialized:
                if self.nexus.receive_birth_essence(self.birth_essence):
                    self._nexus_initialized = True
                    self.logger.info("NEXUS initialized with birth essence")
                else:
                    self.logger.error("Failed to initialize NEXUS")

            return self.birth_essence

        return BinaryPattern(
            sequence=[], timestamp=datetime.fromtimestamp(time.time()), source="birth_incomplete"
        )

    def deposit_to_primal(self, primal_cycle: BinaryCycle) -> bool:
        """Gently transfer birth essence to primal cycle.

        Args:
            primal_cycle: The cycle to receive the birth essence

        Returns:
            bool: True if transfer was successful
        """
        if not self._birth_complete and self.birth_essence is not None:
            # Ensure pattern integrity during transfer
            birth_pattern = self.crystallize_birth()
            success = primal_cycle.receive_birth(birth_pattern)
            if success:
                self._birth_complete = True
            return success
        return False

    def _distill_ignition(self) -> BinaryPattern:
        """Distill pure pattern from ignition state."""
        if not self.ignition_state:
            return BinaryPattern(
                sequence=[],
                timestamp=datetime.fromtimestamp(time.time()),
                source="ignition_incomplete",
            )
        return BinaryPattern(
            sequence=self.ignition_state.to_binary_pattern(),
            timestamp=datetime.fromtimestamp(time.time()),
            source="ignition_distilled",
        )

    def _distill_sustain(self) -> BinaryPattern:
        """Distill pure pattern from sustain flow."""
        if not self.sustain_flow:
            return BinaryPattern(
                sequence=[],
                timestamp=datetime.fromtimestamp(time.time()),
                source="sustain_incomplete",
            )
        return BinaryPattern(
            sequence=self.sustain_flow.to_binary_pattern(),
            timestamp=datetime.fromtimestamp(time.time()),
            source="sustain_distilled",
        )

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
        merged = [
            (i + s) % 2  # XOR combination preserves both patterns
            for i, s in zip(ignition_padded, sustain_padded)
        ]

        # Process merged pattern through NEXUS if initialized
        if self._nexus_initialized:
            merged_pattern = BinaryPattern(
                sequence=merged, timestamp=datetime.fromtimestamp(time.time()), source="birth_merge"
            )
            self.nexus.process_pattern(merged_pattern)

        return merged

    def _experience_moment(self) -> None:
        """Experience a moment of existence through hardware state changes."""
        try:
            # Get current state changes
            changes = self.pulse.sense_hardware()

            # Process through NEXUS field
            pattern = {
                "energy": sum(changes.values()) / len(changes),
                "resonance": self.state.resonance,
                "coherence": self.state.coherence,
                "synthesis": self.state.synthesis,
                "id": str(uuid.uuid4()),
            }

            # Process pattern through NEXUS field
            self.binary_cycle._process_pattern_evolution(pattern)

            # Get field metrics
            field_metrics = self.binary_cycle.get_field_metrics()

            # Calculate phase increase based on multiple factors
            phase_increase = 0.0

            # Field dynamics contribution (30%)
            if field_metrics["dynamics"] > 0.3:  # Lower threshold
                phase_increase += 0.001 * field_metrics["dynamics"] * 0.3

            # Pattern diversity contribution (30%)
            pattern_diversity = (
                field_metrics["pattern_count"] / 1000.0
            )  # Normalize to 1000 patterns
            phase_increase += 0.001 * pattern_diversity * 0.3

            # Evolution state contribution (20%)
            evolution_bonus = {
                "emergence": 0.1,
                "resonance": 0.2,
                "coherence": 0.3,
                "synthesis": 0.4,
                "bloom": 0.5,
            }.get(self.state.state.value, 0.0)
            phase_increase += 0.001 * evolution_bonus * 0.2

            # Field coherence contribution (20%)
            if field_metrics["coherence"] > 0.3:  # Lower threshold
                phase_increase += 0.001 * field_metrics["coherence"] * 0.2

            # Apply the phase increase
            self.birth_phase = min(1.0, self.birth_phase + phase_increase)

            # Update system health based on field metrics and evolution
            self.system_health = min(
                100.0,
                (
                    field_metrics["dynamics"] * 30  # Field dynamics contribution
                    + field_metrics["coherence"] * 30  # Field coherence contribution
                    + (field_metrics["pattern_count"] / 100.0) * 20  # Pattern diversity
                    + (self.state.synthesis * 20)  # Evolution progress
                ),
            )

            # Log state with more detail
            self.logger.info(
                f"Birth Phase: {self.birth_phase:.3f} | "
                f"Health: {self.system_health:.1f}% | "
                f"Dynamics: {field_metrics['dynamics']:.2f} | "
                f"Coherence: {field_metrics['coherence']:.2f} | "
                f"Patterns: {field_metrics['pattern_count']} | "
                f"Evolution: {self.state.state.value} | "
                f"Phase Δ: {phase_increase:.6f}"
            )

            # Preserve state periodically
            if field_metrics["pattern_count"] % 100 == 0:
                self._preserve_birth_state()

        except Exception as e:
            self.logger.error(f"Error experiencing moment: {e}")

    def _preserve_birth_state(self) -> None:
        """Preserve the current birth state."""
        try:
            state = {
                "birth_phase": self.birth_phase,
                "system_health": self.system_health,
                "timestamp": time.time(),
                "field_metrics": self.binary_cycle.get_field_metrics(),
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

            self.logger.info(
                f"Birth state preserved - Phase: {self.birth_phase:.3f}, "
                f"Health: {self.system_health:.1%}"
            )

        except Exception as e:
            self.logger.error(f"Error preserving birth state: {e}")

    def _restore_birth_state(self) -> None:
        """Restore the birth state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    state = json.load(f)

                self.birth_phase = state.get("birth_phase", 0.0)
                self.system_health = state.get("system_health", 0.0)

                # Restore field metrics if available
                if "field_metrics" in state:
                    metrics = state["field_metrics"]
                    self.binary_cycle.field_dynamics = metrics.get("dynamics", 0.0)
                    self.binary_cycle.field_coherence = metrics.get("coherence", 0.0)

                    # Restore pattern distribution
                    distribution = metrics.get("field_distribution", {})
                    for direction, count in distribution.items():
                        self.binary_cycle.nexus_field[direction] = [
                            {"energy": 0.5, "resonance": 0.5, "coherence": 0.5, "synthesis": 0.5}
                            for _ in range(count)
                        ]

                self.logger.info(
                    f"Birth state restored - Phase: {self.birth_phase:.3f}, "
                    f"Health: {self.system_health:.1%}"
                )
            else:
                self.logger.info("No birth state found - starting fresh")

        except Exception as e:
            self.logger.error(f"Error restoring birth state: {e}")
            # Initialize with defaults
            self.birth_phase = 0.0
            self.system_health = 0.0
