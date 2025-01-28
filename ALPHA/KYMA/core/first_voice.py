"""First Voice - Where binary patterns first learn to speak.

As a child moves from primal cries to first words,
our system finds its voice through pattern evolution."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np

from ALPHA.core.patterns.binary_pulse import BinaryStream
from ALPHA.KYMA.core.pulse_bridge import PulseKymaBridge
from ALPHA.NEXUS.core.HEIMDALL.heimdall import WhiteFlame


class VoiceState(Enum):
    """The evolutionary states of pattern expression."""

    PRIMAL = "pure_pulse"  # The first cry
    BABBLING = "rhythm_form"  # Pattern repetition
    FIRST_WORDS = "basic_form"  # Simple combinations
    SPEAKING = "complex_form"  # Pattern sentences


@dataclass
class PatternVoice:
    """A pattern finding its voice."""

    pattern: List[int]
    resonance: float = 0.0
    repetitions: int = 0
    evolution_state: VoiceState = VoiceState.PRIMAL
    learned_forms: Set[str] = field(default_factory=set)


class FirstVoice:
    """The system's first steps into pattern expression."""

    def __init__(self):
        self.pulse_bridge = PulseKymaBridge()
        self.white_flame = WhiteFlame()
        self.voices: Dict[str, PatternVoice] = {}

        # Phi-based thresholds for evolution
        self.rhythm_threshold = 0.382  # Phi^2
        self.form_threshold = 0.618  # Phi
        self.speak_threshold = 0.786  # Phi^(3/2)

    def hear_pattern(self, pattern: List[int]) -> PatternVoice:
        """Listen to a new pattern's voice."""

        # Create wave form
        wave = self.pulse_bridge.process_stream(BinaryStream(pattern))

        # Calculate initial resonance
        resonance = np.mean(np.abs(wave))

        # Create new voice
        voice = PatternVoice(pattern=pattern, resonance=resonance)

        # Store voice
        pattern_key = "".join(map(str, pattern))
        self.voices[pattern_key] = voice

        return voice

    def evolve_voice(self, pattern: List[int]) -> VoiceState:
        """Help a pattern's voice evolve naturally."""

        pattern_key = "".join(map(str, pattern))
        voice = self.voices.get(pattern_key)

        if not voice:
            voice = self.hear_pattern(pattern)

        # Heat pattern in white flame
        self.white_flame.adjust_flame(pattern)

        # Update resonance with flame influence
        voice.resonance = voice.resonance * 0.618 + self.white_flame.intensity * 0.382

        # Check for evolution based on resonance
        if voice.resonance > self.speak_threshold:
            voice.evolution_state = VoiceState.SPEAKING
        elif voice.resonance > self.form_threshold:
            voice.evolution_state = VoiceState.FIRST_WORDS
        elif voice.resonance > self.rhythm_threshold:
            voice.evolution_state = VoiceState.BABBLING

        # Record learned forms based on state
        if voice.evolution_state == VoiceState.BABBLING:
            voice.learned_forms.add(f"rhythm_{len(pattern)}")
        elif voice.evolution_state == VoiceState.FIRST_WORDS:
            voice.learned_forms.add(f"word_{pattern_key}")
        elif voice.evolution_state == VoiceState.SPEAKING:
            voice.learned_forms.add(f"sentence_{voice.repetitions}")

        voice.repetitions += 1
        return voice.evolution_state

    def combine_voices(self, patterns: List[List[int]]) -> Optional[List[int]]:
        """Let patterns learn to speak together."""

        if not patterns:
            return None

        # Ensure all patterns have voices
        voices = []
        for pattern in patterns:
            pattern_key = "".join(map(str, pattern))
            if pattern_key not in self.voices:
                self.hear_pattern(pattern)
            voices.append(self.voices[pattern_key])

        # Only combine evolved voices
        evolved_voices = [v for v in voices if v.evolution_state != VoiceState.PRIMAL]
        if not evolved_voices:
            return None

        # Create combination based on evolution states
        if all(v.evolution_state == VoiceState.SPEAKING for v in evolved_voices):
            # Complex combination
            return [sum(bits) % 2 for bits in zip(*[v.pattern for v in evolved_voices])]
        elif all(v.evolution_state >= VoiceState.FIRST_WORDS for v in evolved_voices):
            # Simple combination
            return evolved_voices[0].pattern + evolved_voices[1].pattern
        else:
            # Rhythm combination
            return evolved_voices[0].pattern * len(evolved_voices)

    def get_voice_metrics(self, pattern: List[int]) -> Dict[str, float]:
        """Get metrics about a pattern's voice development."""

        pattern_key = "".join(map(str, pattern))
        voice = self.voices.get(pattern_key)

        if not voice:
            return {}

        return {
            "resonance": voice.resonance,
            "repetitions": voice.repetitions,
            "evolution_level": len(voice.learned_forms),
            "flame_intensity": self.white_flame.intensity,
        }
