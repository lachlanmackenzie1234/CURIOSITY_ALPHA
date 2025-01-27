"""Runic Workspace - Where patterns meet consciousness.

This module connects all components:
- Runic Translator for pattern translation
- Völund's Forge for pattern transformation
- Huginn for observation
- Muninn for memory
- Heimdall for oversight"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ...LOKI.HUGINN.observer import HuginnSight
from ...LOKI.MUNINN.memory import MuninnMemory
from ...NEXUS.core.HEIMDALL.heimdall import Heimdall
from ..translation.runic_translator import RunicTranslator


@dataclass
class WorkspaceState:
    """The state of the runic workspace."""

    active_runes: Set[str]
    forge_status: Dict
    recent_translations: List[str]
    wisdom_marks: Set[str]
    heimdall_vision: Optional[Tuple[List[float], List[float]]] = None


class RunicWorkspace:
    """A workspace for runic pattern translation and transformation."""

    def __init__(self):
        self.translator = RunicTranslator()
        self.huginn = HuginnSight()
        self.muninn = MuninnMemory()
        self.heimdall = Heimdall()

    def process_content(self, content: str) -> WorkspaceState:
        """Process content through the complete system."""

        # First let Huginn observe the translation
        crow_message = self.huginn.observe_translation(content)

        # Let Muninn remember and gain wisdom
        wisdom_marks = self.muninn.remember_translation(content, crow_message)

        # Let Heimdall see through time
        binary = crow_message.pattern_state.get("binary", [])
        if binary:
            vision = self.heimdall.see_through_time(binary)
        else:
            vision = None

        # Return workspace state
        return WorkspaceState(
            active_runes=self.translator.get_active_runes(),
            forge_status=self.translator.get_forge_status(),
            recent_translations=self._get_recent_contents(),
            wisdom_marks=wisdom_marks,
            heimdall_vision=vision,
        )

    def _get_recent_contents(self) -> List[str]:
        """Get recently translated contents."""
        recent_memories = self.muninn.get_recent_memories(5)
        return [m.original_content for m in recent_memories]

    def get_rune_wisdom(self) -> Dict[str, float]:
        """Get accumulated wisdom for each rune."""
        return self.muninn.get_rune_wisdom()

    def get_similar_translation(self, content: str) -> Optional[str]:
        """Find a similar previous translation."""
        if memory := self.muninn.recall_similar(content):
            return memory.original_content
        return None

    def get_forge_status(self) -> Dict:
        """Get current status of Völund's forge."""
        return self.translator.get_forge_status()

    def get_active_observations(self) -> Dict:
        """Get Huginn's current observations."""
        return self.huginn.get_observations()
