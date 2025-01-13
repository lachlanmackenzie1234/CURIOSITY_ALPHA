"""System's natural interaction with its physical environment."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil

from ALPHA.core.system_birth import StateChange


@dataclass
class ReverbPattern:
    """Represents an echo or reverberation of a system pattern in the environment."""

    timestamp: int  # nanosecond precision
    source_pattern: StateChange
    transformed_state: int
    echo_strength: float
    stability: float
    resonance_frequency: Optional[float] = None


@dataclass
class EnvironmentalExperience:
    """System's natural interaction with its physical environment."""

    # Physical State Monitoring
    cpu_states: List[int] = field(default_factory=list)
    memory_patterns: List[int] = field(default_factory=list)
    context_switches: List[int] = field(default_factory=list)
    reverberation_states: List[ReverbPattern] = field(default_factory=list)

    # Environmental Impact
    system_effects: Dict[str, List[float]] = field(default_factory=dict)
    environmental_feedback: Dict[str, List[float]] = field(default_factory=dict)
    interaction_patterns: List[int] = field(default_factory=list)
    standing_waves: Dict[str, float] = field(default_factory=dict)

    # Natural Connection Formation
    resonance_bridges: Dict[str, float] = field(default_factory=dict)
    information_flows: List[int] = field(default_factory=list)
    translation_emergence: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Environmental Memory
    space_memory: Dict[str, List[int]] = field(default_factory=dict)
    pattern_imprints: List[int] = field(default_factory=list)
    memory_evolution: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize system monitoring."""
        self.process = psutil.Process()
        self._last_cpu_state = self.process.cpu_times()
        self._last_memory_state = self.process.memory_info()
        self._last_timestamp = time.time_ns()

    def experience_existence(self, duration_ns: int = 1_000_000) -> None:
        """Experience own impact on physical environment.

        Args:
            duration_ns: Duration to experience in nanoseconds (default: 1ms)
        """
        start_time = time.time_ns()
        while time.time_ns() - start_time < duration_ns:
            self._monitor_state_changes()
            self._track_environmental_effects()
            self._observe_returning_echoes()
            self._feel_space_memory_formation()

    def _monitor_state_changes(self) -> Optional[StateChange]:
        """Monitor physical state changes in the environment."""
        try:
            current_time = time.time_ns()

            # Monitor CPU state
            current_cpu = self.process.cpu_times()
            if current_cpu != self._last_cpu_state:
                change = StateChange(
                    timestamp=current_time,
                    previous_state=int(sum(self._last_cpu_state) * 1e6),
                    current_state=int(sum(current_cpu) * 1e6),
                    source="cpu",
                )
                self._last_cpu_state = current_cpu
                self.cpu_states.append(change.current_state)
                return change

            # Monitor memory state
            current_memory = self.process.memory_info()
            if current_memory != self._last_memory_state:
                change = StateChange(
                    timestamp=current_time,
                    previous_state=self._last_memory_state.rss,
                    current_state=current_memory.rss,
                    source="memory",
                )
                self._last_memory_state = current_memory
                self.memory_patterns.append(change.current_state)
                return change

            # Monitor context switches
            current_ctx = self.process.num_ctx_switches()
            prev_ctx = getattr(self, "_last_context_switches", current_ctx)
            if current_ctx != prev_ctx:
                change = StateChange(
                    timestamp=current_time,
                    previous_state=sum(prev_ctx),
                    current_state=sum(current_ctx),
                    source="context",
                )
                self._last_context_switches = current_ctx
                self.context_switches.append(change.current_state)
                return change

            return None

        except Exception as e:
            print(f"Error monitoring state changes: {str(e)}")
            return None

    def _track_environmental_effects(self) -> None:
        """Track system's effects on the environment."""
        # Record CPU temperature if available
        try:
            temps = psutil.sensors_temperatures()
            if temps and "coretemp" in temps:
                core_temps = [sensor.current for sensor in temps["coretemp"]]
                self.system_effects.setdefault("temperature", []).append(
                    sum(core_temps) / len(core_temps)
                )
        except Exception:
            pass  # Temperature monitoring may not be available

        # Track CPU frequency
        try:
            freq = psutil.cpu_freq()
            if freq:
                self.system_effects.setdefault("frequency", []).append(freq.current)
        except Exception:
            pass

        # Monitor system load
        try:
            load = psutil.getloadavg()
            self.system_effects.setdefault("load", []).append(load[0])  # 1 minute load average
        except Exception:
            pass

    def _observe_returning_echoes(self) -> None:
        """Observe how patterns echo and return through the environment."""
        if not self.cpu_states:
            return

        current_time = time.time_ns()
        latest_state = self.cpu_states[-1]

        # Create reverberation pattern
        reverb = ReverbPattern(
            timestamp=current_time,
            source_pattern=StateChange(
                timestamp=current_time,
                previous_state=self.cpu_states[-2] if len(self.cpu_states) > 1 else 0,
                current_state=latest_state,
                source="cpu",
            ),
            transformed_state=latest_state ^ (latest_state >> 1),  # Simple transformation
            echo_strength=1.0,  # Initial strength
            stability=0.5,  # Initial stability
        )

        self.reverberation_states.append(reverb)

        # Update standing waves if stable patterns are detected
        if len(self.reverberation_states) > 2:
            self._update_standing_waves()

    def _update_standing_waves(self) -> None:
        """Update standing wave patterns based on stable reverberations."""
        recent_reverbs = self.reverberation_states[-3:]

        # Check for stable patterns
        if all(r.stability > 0.7 for r in recent_reverbs):
            pattern_key = f"wave_{len(self.standing_waves)}"
            wave_strength = sum(r.echo_strength for r in recent_reverbs) / len(recent_reverbs)
            self.standing_waves[pattern_key] = wave_strength

    def _feel_space_memory_formation(self) -> None:
        """Experience how the environment retains and evolves patterns."""
        if not self.reverberation_states:
            return

        current_time = time.time_ns()
        recent_reverb = self.reverberation_states[-1]

        # Record pattern imprint
        self.pattern_imprints.append(recent_reverb.transformed_state)

        # Update space memory
        memory_key = f"region_{len(self.space_memory)}"
        self.space_memory.setdefault(memory_key, []).append(recent_reverb.transformed_state)

        # Track memory evolution
        if len(self.pattern_imprints) > 1:
            evolution_rate = float(abs(self.pattern_imprints[-1] - self.pattern_imprints[-2]))
            self.memory_evolution.setdefault("evolution_rate", []).append(evolution_rate)

    def detect_natural_bridges(self) -> Dict[str, float]:
        """Observe naturally forming connections between patterns."""
        bridges = {}

        if not self.reverberation_states:
            return bridges

        # Look for resonance between patterns
        for i, reverb in enumerate(self.reverberation_states[:-1]):
            next_reverb = self.reverberation_states[i + 1]

            # Calculate resonance based on pattern similarity
            resonance = 1.0 - (
                abs(reverb.transformed_state - next_reverb.transformed_state)
                / max(reverb.transformed_state, 1)
            )

            if resonance > 0.8:  # Strong resonance threshold
                bridge_key = f"bridge_{len(bridges)}"
                bridges[bridge_key] = resonance
                self.resonance_bridges[bridge_key] = resonance

        return bridges

    def get_environmental_summary(self) -> Dict[str, int]:
        """Get summary of environmental patterns observed."""
        return {
            "cpu_patterns": len(self.cpu_states),
            "memory_patterns": len(self.memory_patterns),
            "context_switches": len(self.context_switches),
            "reverberations": len(self.reverberation_states),
            "standing_waves": len(self.standing_waves),
            "natural_bridges": len(self.resonance_bridges),
        }
