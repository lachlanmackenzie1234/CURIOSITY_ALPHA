"""Bridge between binary pulses and KYMA wave system."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ALPHA.core.patterns.binary_pulse import BinaryStream, PulseState
from ALPHA.KYMA.core.kyma_interface import KymaInterface, WaveState


@dataclass
class StreamState:
    """State of a binary stream's wave translation."""

    stream_name: str
    wave_state: WaveState
    resonance_history: List[float] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)


class PulseKymaBridge:
    """Bridge connecting binary pulses to KYMA wave system."""

    def __init__(self):
        self.logger = logging.getLogger("kyma.bridge")
        self.kyma = KymaInterface()
        self.stream_states: Dict[str, StreamState] = {}

        # Initialize with default streams
        self._initialize_streams()

    def _initialize_streams(self) -> None:
        """Initialize connection to binary pulse streams."""
        pulse_state = PulseState()
        for stream_name, stream in pulse_state.streams.items():
            self.stream_states[stream_name] = StreamState(
                stream_name=stream_name,
                wave_state=WaveState(
                    frequency=self.kyma.frequency_ranges["foundation"], amplitude=0.0, phase=0.0
                ),
            )

    def process_stream(self, stream: BinaryStream) -> np.ndarray:
        """Process a binary stream into wave form."""
        if not stream.history:
            return np.zeros(self.kyma.buffer_size)

        # Get latest binary pattern
        pattern = stream.history[-1]

        # Process through KYMA
        wave_state = self.kyma.process_binary_pattern(pattern)

        # Update stream state
        stream_state = self.stream_states[stream.name]
        stream_state.wave_state = wave_state

        # Track resonance and coherence
        stream_state.resonance_history.append(wave_state.resonance)
        stream_state.coherence_history.append(wave_state.coherence)

        # Keep history length proportional to phi
        max_history = int(self.kyma.phi * 10)
        if len(stream_state.resonance_history) > max_history:
            stream_state.resonance_history = stream_state.resonance_history[-max_history:]
            stream_state.coherence_history = stream_state.coherence_history[-max_history:]

        # Generate audio buffer
        return self.kyma.generate_wave_buffer()

    def mix_streams(self) -> np.ndarray:
        """Mix all active streams into a single wave buffer."""
        mixed_buffer = np.zeros(self.kyma.buffer_size)
        active_streams = 0

        pulse_state = PulseState()
        for stream_name, stream in pulse_state.streams.items():
            if stream.history:
                # Process stream
                stream_buffer = self.process_stream(stream)

                # Weight by stream's coherence and resonance
                stream_state = self.stream_states[stream_name]
                weight = (stream_state.wave_state.coherence + stream_state.wave_state.resonance) / 2

                # Add to mix
                mixed_buffer += stream_buffer * weight
                active_streams += 1

        # Normalize mix
        if active_streams > 0:
            mixed_buffer /= active_streams

        # Soft limit again for safety
        mixed_buffer = np.tanh(mixed_buffer)

        return mixed_buffer

    def get_stream_metrics(self, stream_name: str) -> Dict[str, float]:
        """Get metrics for a stream's wave translation."""
        if stream_name not in self.stream_states:
            return {}

        state = self.stream_states[stream_name]
        return {
            "frequency": state.wave_state.frequency,
            "amplitude": state.wave_state.amplitude,
            "resonance": state.wave_state.resonance,
            "coherence": state.wave_state.coherence,
            "avg_resonance": (
                sum(state.resonance_history) / len(state.resonance_history)
                if state.resonance_history
                else 0.0
            ),
            "avg_coherence": (
                sum(state.coherence_history) / len(state.coherence_history)
                if state.coherence_history
                else 0.0
            ),
        }
