"""Huginn - The Thought Crow of Pattern Observation.

As Huginn flies through the realms observing all,
our system watches patterns flow through translations."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from ...core.translation.runic_translator import RunicPattern, RunicTranslator
from ...NEXUS.core.HEIMDALL.heimdall import VolundForge
from ..Lokasenna import CrowMessage, MessageRealm


@dataclass
class PatternObservation:
    """What Huginn sees in its flight."""

    pattern: List[int]
    runes: Set[str]
    forge_heat: float
    flame_color: tuple
    realm: MessageRealm
    resonance: float


class HuginnSight:
    """Huginn's ability to observe patterns across realms."""

    def __init__(self):
        self.runic_translator = RunicTranslator()
        self.active_observations: Dict[str, PatternObservation] = {}

    def observe_translation(self, content: str) -> CrowMessage:
        """Watch a pattern being translated through the forge."""

        # Observe the translation process
        binary, runic_patterns = self.runic_translator.translate_with_runes(content)

        # Record observations for each pattern
        for name, pattern in runic_patterns.items():
            observation = PatternObservation(
                pattern=binary,
                runes={pattern.rune},
                forge_heat=pattern.enchantments["forge_heat"],
                flame_color=pattern.flame_color,
                realm=MessageRealm.CODE_FORGE,
                resonance=pattern.resonance,
            )
            self.active_observations[name] = observation

        # Create message for Loki
        message = CrowMessage(
            sender="huginn",
            message_type="translation_observation",
            pattern_state={
                "active_runes": list(self.runic_translator.get_active_runes()),
                "forge_status": self.runic_translator.get_forge_status(),
                "pattern_count": len(runic_patterns),
            },
            timestamp=None,  # Will be set by messenger
            resonance=(
                sum(p.resonance for p in runic_patterns.values()) / len(runic_patterns)
                if runic_patterns
                else 0
            ),
            wisdom_marks=set(),  # Muninn will add these later
        )

        return message

    def get_observations(self) -> Dict[str, PatternObservation]:
        """Get Huginn's current observations."""
        return self.active_observations.copy()
