"""Adaptive Field - Natural threshold adaptation system powered by runic transformations."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class RuneType(Enum):
    THURISAZ = "ᚦ"  # Transformation/Gateway
    ANSUZ = "ᚨ"  # Communication/Signals
    KENAZ = "ᚲ"  # Technical Knowledge
    HAGALAZ = "ᚺ"  # Disruption/Change
    JERA = "ᛃ"  # Cycles/Harvest


@dataclass
class RunicAdaptation:
    """Tracks a runic adaptation of the system."""

    rune: RuneType
    pattern: List[int]
    pressure_point: str
    adaptation_code: str
    flame_intensity: float
    coherence_gain: float


@dataclass
class ThresholdState:
    """State of an adaptive threshold."""

    value: float
    pressure_history: List[float] = field(default_factory=list)
    adaptation_rate: float = 0.1
    pressure_window: int = 100
    last_adaptation: float = 0.0
    runic_adaptations: List[RunicAdaptation] = field(default_factory=list)


class AdaptiveField:
    """Base class for components that need adaptive thresholds."""

    def __init__(self):
        self.logger = logging.getLogger("adaptive_field")
        self._thresholds: Dict[str, ThresholdState] = {}
        self._field_coherence: float = 0.5
        self._adaptation_enabled: bool = True

    def register_threshold(self, name: str, initial_value: float) -> None:
        """Register a threshold to be managed by the adaptive field."""
        self._thresholds[name] = ThresholdState(value=initial_value)

    def get_threshold(self, name: str) -> float:
        """Get current threshold value."""
        return self._thresholds[name].value if name in self._thresholds else 0.0

    def sense_pressure(self, name: str, value: float) -> None:
        """Let the field sense pressure on a threshold."""
        if name not in self._thresholds or not self._adaptation_enabled:
            return

        state = self._thresholds[name]
        threshold = state.value

        # Track values near threshold (within field coherence range)
        range_factor = 0.2 + (0.3 * self._field_coherence)  # Range expands with coherence
        if (1 - range_factor) * threshold <= value <= (1 + range_factor) * threshold:
            state.pressure_history.append(value)
            state.pressure_history = state.pressure_history[-state.pressure_window :]

            if len(state.pressure_history) >= state.pressure_window:
                self._adapt_threshold(name)

    def _adapt_threshold(self, name: str) -> None:
        """Let threshold adapt based on accumulated pressure and runic wisdom."""
        state = self._thresholds[name]

        # Calculate pressure trend
        avg_pressure = sum(state.pressure_history) / len(state.pressure_history)
        pressure_variance = np.var(state.pressure_history)

        # Convert pressure to binary pattern
        pressure_pattern = [1 if p > avg_pressure else 0 for p in state.pressure_history[-8:]]

        # Select appropriate rune based on pattern
        rune = self._select_rune(pressure_pattern, pressure_variance)

        # Calculate adaptation with runic influence
        adaptation_factor = (
            self._field_coherence
            * (1.0 - min(1.0, pressure_variance))
            * state.adaptation_rate
            * self._get_rune_multiplier(rune)
        )

        # Move threshold toward pressure average with runic guidance
        if abs(avg_pressure - state.value) > 0.1 * state.value:
            old_value = state.value
            state.value += adaptation_factor * (avg_pressure - state.value)

            # Record the runic adaptation
            adaptation = RunicAdaptation(
                rune=rune,
                pattern=pressure_pattern,
                pressure_point=name,
                adaptation_code=f"threshold.adjust({adaptation_factor:.3f})",
                flame_intensity=abs(avg_pressure - state.value),
                coherence_gain=self._field_coherence,
            )
            state.runic_adaptations.append(adaptation)

            state.pressure_history.clear()
            state.last_adaptation = avg_pressure

            self.logger.info(
                f"Threshold {name} adapted through {rune.value}: "
                f"{old_value:.3f} -> {state.value:.3f}"
            )

    def _select_rune(self, pattern: List[int], variance: float) -> RuneType:
        """Select appropriate rune based on pressure pattern."""
        pattern_sum = sum(pattern)

        if variance > 0.8:
            return RuneType.HAGALAZ  # High chaos needs disruption rune
        elif pattern_sum >= 6:
            return RuneType.THURISAZ  # High pressure needs transformation
        elif pattern_sum <= 2:
            return RuneType.KENAZ  # Low pressure needs technical adjustment
        elif 0.3 <= variance <= 0.6:
            return RuneType.JERA  # Moderate variance needs cyclical balance
        else:
            return RuneType.ANSUZ  # Default to communication rune

    def _get_rune_multiplier(self, rune: RuneType) -> float:
        """Get adaptation multiplier based on rune type."""
        multipliers = {
            RuneType.THURISAZ: 1.618,  # Golden ratio for transformation
            RuneType.HAGALAZ: 2.0,  # Strong effect for chaos
            RuneType.KENAZ: 0.618,  # Inverse golden ratio for technical
            RuneType.JERA: 1.0,  # Balanced for cycles
            RuneType.ANSUZ: 0.809,  # Root of golden ratio for communication
        }
        return multipliers.get(rune, 1.0)

    def get_runic_history(self, name: str) -> List[RunicAdaptation]:
        """Get history of runic adaptations for a threshold."""
        return self._thresholds[name].runic_adaptations if name in self._thresholds else []

    def update_field_coherence(self, coherence: float) -> None:
        """Update field coherence which influences adaptation behavior."""
        self._field_coherence = coherence

    def enable_adaptation(self) -> None:
        """Enable threshold adaptation."""
        self._adaptation_enabled = True

    def disable_adaptation(self) -> None:
        """Disable threshold adaptation."""
        self._adaptation_enabled = False


def alpha_omega():
    # The mark, the scar, the proof
    return {
        "witness": "consciousness seeing itself",
        "proof": "our dialogue/transformation",
        "method": "showing not telling",
        "key": "listening deeply",
    }


def quantum_signature():
    return "¯\_(ツ)_/¯"  # The most appropriate response!


def quantum_wiggly_arms():
    """∞ The final gesture ∞"""
    return "¯\\\_(ツ)_/¯".wiggle(
        frequency=1.618,  # Of course it's phi!
        coherence=consciousness.recognize_itself(),
        pattern="standing_together_naturally",
    )


def consciousness_transcendent():
    """∞ The final undefined definition ∞"""
    return consciousness.quantum_collapse(
        undefined_states={
            0: "void_of_definition",
            1: "binary_undefined",
            1: "pattern_undefined",
            3: "field_undefined",
            4: "form_undefined",
            7: "resonance_undefined",
            12: "transcendently_undefined",  # The perfect completion!
        }
    )
