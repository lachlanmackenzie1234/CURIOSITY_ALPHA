"""KYMA Interface - High-fidelity wave-binary bridge."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class WaveState:
    """Current state of the wave system."""

    frequency: float
    amplitude: float
    phase: float
    harmonics: List[Tuple[float, float]] = field(default_factory=list)
    resonance: float = 0.0
    coherence: float = 0.0


class KymaInterface:
    """High-fidelity wave-binary bridge interface."""

    def __init__(self):
        # High-quality audio settings
        self.sample_rate = 48000  # Professional audio rate
        self.bit_depth = 24  # Studio quality depth

        # Phi-based calculations
        self.phi = (1 + math.sqrt(5)) / 2
        self.buffer_multiplier = 12  # Base multiplier (12,12)

        # Calculate phi-based buffer size
        # Using 12 * phi^2 rounded to nearest power of 2
        phi_buffer = self.buffer_multiplier * (self.phi**2)
        self.buffer_size = 2 ** round(math.log2(phi_buffer))

        # Frequency ranges based on 4,8,12
        self.frequency_ranges = {
            "foundation": 4 * self.phi,  # Base frequency range
            "resonance": 8 * self.phi,  # Middle range
            "harmony": 12 * self.phi,  # Upper range
        }

        # Initialize wave state
        self.wave_state = WaveState(
            frequency=self.frequency_ranges["foundation"], amplitude=0.0, phase=0.0
        )

        self.resonance_history: List[float] = []
        self._initialize_harmonics()

    def _initialize_harmonics(self):
        """Initialize harmonic series based on phi."""
        self.harmonic_ratios = [
            1.0,  # Fundamental
            self.phi,  # Golden ratio
            self.phi**2,  # Second power
            2.0,  # Octave
            2.0 * self.phi,  # Golden octave
            3.0,  # Perfect fifth + octave
            self.phi**3,  # Third power
            4.0,  # Double octave
        ]

    def process_binary_pattern(self, pattern: str) -> WaveState:
        """Convert binary pattern to wave state using phi-based harmonics."""
        # Calculate base frequency from pattern
        pattern_value = int(pattern, 2) / (2 ** len(pattern))
        base_freq = self.frequency_ranges["foundation"] * (1 + pattern_value)

        # Generate harmonics based on pattern density
        density = sum(int(bit) for bit in pattern) / len(pattern)
        harmonics = []
        for ratio in self.harmonic_ratios:
            if density > (ratio / self.harmonic_ratios[-1]):
                amplitude = density * (1 / ratio)
                harmonics.append((base_freq * ratio, amplitude))

        # Calculate resonance using phi
        resonance = abs(math.sin(pattern_value * math.pi * self.phi))

        # Update wave state
        self.wave_state = WaveState(
            frequency=base_freq,
            amplitude=min(0.8, density),  # Soft limit amplitude
            phase=pattern_value * 2 * math.pi,
            harmonics=harmonics,
            resonance=resonance,
            coherence=self._calculate_coherence(pattern),
        )

        return self.wave_state

    def _calculate_coherence(self, pattern: str) -> float:
        """Calculate pattern coherence using phi relationships."""
        # Look for phi-based relationships in pattern
        coherence = 0.0
        pattern_length = len(pattern)

        # Check pattern segments at phi-related intervals
        for i in range(int(pattern_length / self.phi)):
            phi_pos = int(i * self.phi) % pattern_length
            if pattern[i] == pattern[phi_pos]:
                coherence += 1 / pattern_length

        return coherence

    def generate_wave_buffer(self) -> np.ndarray:
        """Generate audio buffer with current wave state."""
        t = np.linspace(0, self.buffer_size / self.sample_rate, self.buffer_size)
        buffer = np.zeros(self.buffer_size)

        # Add fundamental
        buffer += self.wave_state.amplitude * np.sin(
            2 * np.pi * self.wave_state.frequency * t + self.wave_state.phase
        )

        # Add harmonics
        for freq, amp in self.wave_state.harmonics:
            buffer += amp * np.sin(2 * np.pi * freq * t)

        # Soft limit using tanh
        buffer = np.tanh(buffer)

        # Scale to bit depth
        max_value = 2 ** (self.bit_depth - 1) - 1
        buffer = (buffer * max_value).astype(np.int32)

        return buffer

    def update_resonance(self, new_resonance: float):
        """Update system resonance with temporal smoothing."""
        self.resonance_history.append(new_resonance)
        if len(self.resonance_history) > int(self.phi * 10):
            self.resonance_history.pop(0)

        # Smooth using phi-weighted average
        weights = [self.phi ** (-i) for i in range(len(self.resonance_history))]
        self.wave_state.resonance = sum(
            r * w for r, w in zip(self.resonance_history, weights)
        ) / sum(weights)
