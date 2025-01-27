"""Muninn - The Memory Crow of Pattern Wisdom.

As Muninn remembers all that has passed,
our system preserves the wisdom of pattern translations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from ...core.translation.runic_translator import RunicPattern
from ..Lokasenna import CrowMessage, MessageRealm


@dataclass
class TranslationMemory:
    """A memory of a pattern translation."""

    original_content: str
    binary_form: List[int]
    runes_used: Set[str]
    forge_heat: float
    resonance: float
    timestamp: datetime
    wisdom_gained: float = 0.0


class MuninnMemory:
    """Muninn's ability to remember and learn from translations."""

    def __init__(self):
        self.translation_memories: Dict[str, TranslationMemory] = {}
        self.rune_wisdom: Dict[str, float] = {}
        self.phi = 0.618  # Golden ratio for wisdom growth

    def remember_translation(self, content: str, message: CrowMessage) -> Set[str]:
        """Remember a translation and extract wisdom from it."""

        # Create memory of translation
        memory = TranslationMemory(
            original_content=content,
            binary_form=message.pattern_state.get("binary", []),
            runes_used=set(message.pattern_state["active_runes"]),
            forge_heat=message.pattern_state["forge_status"]["fires"]["pattern"],
            resonance=message.resonance,
            timestamp=datetime.now(),
        )

        # Calculate wisdom gained
        memory.wisdom_gained = self._calculate_wisdom(
            memory.forge_heat, memory.resonance, len(memory.runes_used)
        )

        # Update rune wisdom
        for rune in memory.runes_used:
            if rune not in self.rune_wisdom:
                self.rune_wisdom[rune] = 0.0
            self.rune_wisdom[rune] += memory.wisdom_gained * self.phi

        # Store memory
        key = f"{content[:32]}_{memory.timestamp.isoformat()}"
        self.translation_memories[key] = memory

        # Return wisdom marks based on learning
        return self._generate_wisdom_marks(memory)

    def _calculate_wisdom(self, heat: float, resonance: float, rune_count: int) -> float:
        """Calculate wisdom gained from a translation."""
        base_wisdom = heat * resonance
        rune_factor = self.phi**rune_count
        return base_wisdom * rune_factor

    def _generate_wisdom_marks(self, memory: TranslationMemory) -> Set[str]:
        """Generate wisdom marks based on translation memory."""
        marks = set()

        # Mark for high resonance
        if memory.resonance > 0.8:
            marks.add("resonant_wisdom")

        # Mark for strong forge heat
        if memory.forge_heat > 0.8:
            marks.add("forge_mastery")

        # Mark for multiple runes
        if len(memory.runes_used) > 3:
            marks.add("runic_mastery")

        # Mark for high wisdom gain
        if memory.wisdom_gained > 0.8:
            marks.add("deep_insight")

        return marks

    def recall_similar(self, content: str) -> Optional[TranslationMemory]:
        """Recall a similar translation from memory."""
        best_match = None
        best_score = 0.0

        for memory in self.translation_memories.values():
            similarity = self._calculate_similarity(content, memory.original_content)
            if similarity > best_score:
                best_score = similarity
                best_match = memory

        return best_match if best_score > 0.7 else None

    def _calculate_similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple length-based similarity for now
        min_len = min(len(a), len(b))
        max_len = max(len(a), len(b))
        return min_len / max_len if max_len > 0 else 0.0

    def get_rune_wisdom(self) -> Dict[str, float]:
        """Get accumulated wisdom for each rune."""
        return self.rune_wisdom.copy()

    def get_recent_memories(self, limit: int = 10) -> List[TranslationMemory]:
        """Get most recent translation memories."""
        sorted_memories = sorted(
            self.translation_memories.values(), key=lambda m: m.timestamp, reverse=True
        )
        return sorted_memories[:limit]
