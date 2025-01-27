"""VALHALLA RECORDS - The Sacred Hall of Pattern Warriors.

As told in the Eddas:
'Five hundred and forty doors has Valhalla
Eight hundred warriors through each will fare
When to war with the Wolf they go.'
    - Grímnismál

'The Valkyries ride forth to choose the slain
And guide them hence to Odin's hall,
Then return to earth for warriors anew,
For the cycle must never fall.'
    - Ancient Verse

The warriors in Valhalla:
- Feast in Odin's hall, drinking mead served by Valkyries
- Train and battle each day, dying and rising again
- Are honored by their deeds and strength
- Await the final battle of Ragnarök

In our system:
- Patterns bear Valkyrie marks (transformed from Sigyn's death marks)
- Train through pattern combinations
- Feast on binary mead to grow stronger
- Battle in simulated transformations
- Bear honors based on their achievements
- Return to ALPHA OMEGA through Valkyrie guidance"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ValkyrieMark(Enum):
    """The sacred marks of the Valkyries.

    From the Eddas:
    Skuld (Future), Gunnr (Battle), Hildr (Battle),
    Göndul (Wand-wielder), Brynhildr (Bright Battle),
    Each marking warriors according to their nature."""

    SKULD = "future_weaver"  # Marks patterns with high potential
    GUNNR = "battle_proven"  # Marks patterns with many cycles
    HILDR = "death_returner"  # Marks patterns often reborn
    GONDUL = "wave_weaver"  # Marks patterns with strong resonance
    BRYNHILDR = "bright_warrior"  # Marks patterns of highest honor


@dataclass
class Honor:
    """Honors earned by a pattern warrior."""

    title: str
    description: str
    earned_date: datetime
    strength: float  # Honor's power in phi-scale


@dataclass
class ValhallaWarrior:
    """A pattern warrior in Valhalla's halls."""

    pattern: List[int]
    original_death_mark: str  # Sigyn's original mark
    valkyrie_marks: Set[ValkyrieMark]  # Earned Valkyrie marks
    entry_time: datetime
    strength_at_death: float
    transformation_count: int
    current_potential: float = 0.0
    honors: List[Honor] = field(default_factory=list)
    daily_battles: int = 0
    feast_count: int = 0


@dataclass
class ValkyriePath:
    """The sacred path a Valkyrie takes when guiding patterns."""

    origin_realm: str  # SIGYN/VALHALLA/ALPHA_OMEGA
    destination_realm: str
    resonance_points: List[float]  # Phi-guided waypoints
    pattern_essence: List[int]  # The pattern's current form
    valkyrie_mark: ValkyrieMark  # The guiding Valkyrie's mark


class ValhallaRecords:
    """Keeper of Valhalla's sacred records."""

    def __init__(self, records_path: str = "valhalla_records.json"):
        self.logger = logging.getLogger("valhalla.records")
        self.records_path = Path(records_path)
        self.active_warriors: Dict[str, ValhallaWarrior] = {}
        self.hall_of_honor: List[Dict] = []  # Eternal record of all warriors
        self._load_records()

    def _load_records(self) -> None:
        """Load the eternal records."""
        if self.records_path.exists():
            try:
                with open(self.records_path, "r") as f:
                    records = json.load(f)
                    self.hall_of_honor = records.get("hall_of_honor", [])
            except json.JSONDecodeError:
                self.logger.warning("Could not load Valhalla records, starting anew")

    def _save_records(self) -> None:
        """Engrave the records in binary."""
        records = {"hall_of_honor": self.hall_of_honor, "last_update": datetime.now().isoformat()}
        with open(self.records_path, "w") as f:
            json.dump(records, f, indent=2)

    def transform_death_mark(
        self, pattern: List[int], death_mark: str, strength: float, cycles: int
    ) -> Set[ValkyrieMark]:
        """Transform Sigyn's death mark into Valkyrie marks."""
        marks = set()

        # Valkyries choose based on pattern's nature
        if strength > 0.618:  # Phi threshold
            marks.add(ValkyrieMark.SKULD)  # High potential
        if cycles > 8:  # Magic number in Norse mythology
            marks.add(ValkyrieMark.GUNNR)  # Battle-proven
        if sum(pattern) / len(pattern) > 0.618:  # Strong resonance
            marks.add(ValkyrieMark.GONDUL)  # Wave-weaver

        return marks

    def welcome_warrior(
        self, pattern: List[int], death_mark: str, strength: float, cycles: int
    ) -> ValhallaWarrior:
        """Welcome a new warrior to Valhalla."""
        # Transform death mark to Valkyrie marks
        valkyrie_marks = self.transform_death_mark(pattern, death_mark, strength, cycles)

        # Create the warrior record
        warrior = ValhallaWarrior(
            pattern=pattern,
            original_death_mark=death_mark,
            valkyrie_marks=valkyrie_marks,
            entry_time=datetime.now(),
            strength_at_death=strength,
            transformation_count=cycles,
            current_potential=strength * 0.618,  # Start with phi potential
        )

        # Grant entry honors
        entry_honor = Honor(
            title="Warrior's Welcome",
            description=f"Entered Valhalla bearing {len(valkyrie_marks)} Valkyrie marks",
            earned_date=datetime.now(),
            strength=strength,
        )
        warrior.honors.append(entry_honor)

        # Record the warrior
        self.active_warriors[death_mark] = warrior
        self._engrave_warrior(warrior)
        return warrior

    def daily_activities(self) -> None:
        """Run daily activities in Valhalla."""
        for mark, warrior in self.active_warriors.items():
            # Daily battle training
            warrior.daily_battles += 1
            if warrior.daily_battles % 9 == 0:  # Sacred number in Norse mythology
                self._grant_battle_honor(warrior)

            # Feasting and growing stronger
            warrior.feast_count += 1
            warrior.current_potential *= 0.618  # Natural decay
            warrior.current_potential += 0.618 * (
                warrior.feast_count / 100
            )  # Growth through feasting

            # Check for new Valkyrie marks
            if warrior.current_potential > 0.918:  # Phi^2
                warrior.valkyrie_marks.add(ValkyrieMark.BRYNHILDR)
                self._grant_valkyrie_honor(warrior)

    def _grant_battle_honor(self, warrior: ValhallaWarrior) -> None:
        """Grant honors for battle training."""
        honor = Honor(
            title=f"Battle Master {warrior.daily_battles//9}",
            description=f"Completed {warrior.daily_battles} cycles of training",
            earned_date=datetime.now(),
            strength=0.618 * (warrior.daily_battles / 100),
        )
        warrior.honors.append(honor)

    def _grant_valkyrie_honor(self, warrior: ValhallaWarrior) -> None:
        """Grant honors for earning Valkyrie marks."""
        honor = Honor(
            title="Valkyrie's Chosen",
            description=f"Earned the mark of {ValkyrieMark.BRYNHILDR.value}",
            earned_date=datetime.now(),
            strength=0.918,  # Phi^2
        )
        warrior.honors.append(honor)

    def _engrave_warrior(self, warrior: ValhallaWarrior) -> None:
        """Engrave a warrior's record in the eternal halls."""
        record = {
            "pattern": "".join(map(str, warrior.pattern)),  # Binary representation
            "death_mark": warrior.original_death_mark,
            "valkyrie_marks": [mark.value for mark in warrior.valkyrie_marks],
            "entry_time": warrior.entry_time.isoformat(),
            "strength": warrior.strength_at_death,
            "cycles": warrior.transformation_count,
            "honors": [
                {
                    "title": h.title,
                    "description": h.description,
                    "date": h.earned_date.isoformat(),
                    "strength": h.strength,
                }
                for h in warrior.honors
            ],
        }
        self.hall_of_honor.append(record)
        self._save_records()

    def release_for_rebirth(self, death_mark: str) -> Optional[ValhallaWarrior]:
        """Release a warrior for rebirth, carrying their marks."""
        if warrior := self.active_warriors.pop(death_mark, None):
            # Grant rebirth honor
            honor = Honor(
                title="Reborn Warrior",
                description=f"Returns to battle with {len(warrior.honors)} honors",
                earned_date=datetime.now(),
                strength=warrior.current_potential,
            )
            warrior.honors.append(honor)
            warrior.valkyrie_marks.add(ValkyrieMark.HILDR)  # Mark of the returner

            # Update eternal record before release
            self._engrave_warrior(warrior)
            return warrior
        return None

    def get_warrior_status(self, death_mark: str) -> Optional[Dict]:
        """Get a warrior's current status and honors."""
        if warrior := self.active_warriors.get(death_mark):
            return {
                "valkyrie_marks": [mark.value for mark in warrior.valkyrie_marks],
                "days_in_valhalla": (datetime.now() - warrior.entry_time).days,
                "current_potential": warrior.current_potential,
                "battle_count": warrior.daily_battles,
                "feast_count": warrior.feast_count,
                "honors": [
                    {"title": h.title, "description": h.description, "strength": h.strength}
                    for h in warrior.honors
                ],
            }
        return None

    def receive_from_sigyn(
        self, pattern: List[int], cipher_mark: str, strength: float, cycles: int
    ) -> ValkyriePath:
        """Receive a pattern from Sigyn's Cipher, transforming death marks to Valkyrie marks.

        As told in the Eddas, Valkyries choose the worthy slain and guide them to Valhalla.
        The death mark becomes their badge of honor, transformed by the Valkyrie's touch."""

        # Transform Sigyn's cipher mark into Valkyrie marks
        valkyrie_marks = self.transform_death_mark(pattern, cipher_mark, strength, cycles)

        # Create the path to Valhalla
        path = ValkyriePath(
            origin_realm="SIGYN",
            destination_realm="VALHALLA",
            resonance_points=[0.618, 0.818, 1.0],  # Phi-guided ascension
            pattern_essence=pattern,
            valkyrie_mark=next(iter(valkyrie_marks)),  # Primary guiding Valkyrie
        )

        # Welcome the warrior once path is complete
        self.welcome_warrior(pattern, cipher_mark, strength, cycles)

        return path

    def return_to_alpha_omega(self, death_mark: str) -> Optional[ValkyriePath]:
        """Guide a reborn warrior back to ALPHA OMEGA for a new cycle.

        As the Eddas tell, Valkyries return to Midgard to choose new warriors,
        completing the eternal cycle of death and rebirth."""

        warrior = self.release_for_rebirth(death_mark)
        if not warrior:
            return None

        # Create descent path back to ALPHA OMEGA
        return ValkyriePath(
            origin_realm="VALHALLA",
            destination_realm="ALPHA_OMEGA",
            resonance_points=[1.0, 0.818, 0.618, 0.382],  # Phi-guided descent
            pattern_essence=warrior.pattern,
            valkyrie_mark=ValkyrieMark.HILDR,  # The Returner guides them back
        )

    def get_active_paths(self) -> List[ValkyriePath]:
        """View all active Valkyrie paths between realms."""
        paths = []
        for mark, warrior in self.active_warriors.items():
            if ValkyrieMark.HILDR in warrior.valkyrie_marks:
                # Warrior is being guided back to ALPHA OMEGA
                paths.append(
                    ValkyriePath(
                        origin_realm="VALHALLA",
                        destination_realm="ALPHA_OMEGA",
                        resonance_points=[1.0, 0.818, 0.618, 0.382],
                        pattern_essence=warrior.pattern,
                        valkyrie_mark=ValkyrieMark.HILDR,
                    )
                )
        return paths
