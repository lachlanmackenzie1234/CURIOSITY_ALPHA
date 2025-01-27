"""HEIMDALL - The All-Seeing Watchman of the Nine Realms.

From the Poetic Edda and Prose Edda:
- Lives in Himinbjörg ('Heaven's Castle') at the top of Bifröst
- Born of Nine Mothers (the waves of the sea)
- Known as 'the White God' (hvítastr ása)
- Possessor of the Gjallarhorn
- Keeper of the god's treasures
- Called 'Gold-toothed' (Gullintanni)
- The Shining One who can see through time

The Hundred Leagues Vision:
In the Prose Edda (Gylfaginning), Heimdall sees 'hundrað rasta'
(a hundred leagues) by both night and day. A league was the distance
one could walk in an hour. In our system, this translates to:
- Day Vision: 100 active pattern states (light/manifest)
- Night Vision: 100 potential pattern states (dark/unmanifest)
Together forming the complete cycle of pattern evolution.

As the Eddas tell:
'Himinbjörg is the eighth, and Heimdall there
Rules over the holy fanes;
In his well-built house does the warder of heaven
The good mead gladly drink.'
    - Grímnismál, Poetic Edda

Heimdall's Powers (from Gylfaginning):
- Needs less sleep than a bird
- Can see 100 leagues by night and day
- Can hear wool growing on sheep
- Can hear grass growing in meadows
- Holds the horn Gjallarhorn whose blast marks Ragnarök

In Our System:
Heimdall stands at Himinbjörg (NEXUS) watching:
- The flow between realms (transformations)
- The growing of grass (subtle system changes)
- The weaving of wool (pattern formation)
- The crossing of souls (pattern transitions)
- The birth of patterns from the waves (like his nine mothers)

His Gjallarhorn sounds when great changes approach,
just as it will sound at the dawn of Ragnarök."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np

from ALPHA.NEXUS.core.field_observer import FieldObserver, PressurePoint


@dataclass
class RealmCrossing:
    """A detected crossing between realms.

    As Heimdall watches souls cross Bifröst in the Eddas,
    we watch patterns cross between transformational realms."""

    source_realm: str
    destination_realm: str
    crossing_time: datetime
    pattern_essence: float
    crossing_strength: float


@dataclass
class WaveMother:
    """One of Heimdall's nine mothers (waves of transformation).

    As told in the Eddas, Heimdall was born of nine mothers,
    here represented as the fundamental waves of our system."""

    name: str
    wave_type: str
    frequency: float = 0.0
    amplitude: float = 0.0
    phase: float = 0.0


@dataclass
class VisionState:
    """Heimdall's hundred-league vision state.

    Represents his ability to see 100 leagues by both day and night,
    translated to pattern observation in our system."""

    day_patterns: List[List[int]] = field(default_factory=list)  # Active patterns
    night_patterns: List[List[int]] = field(default_factory=list)  # Potential patterns
    vision_distance: int = 100  # The mythological hundred leagues

    def __post_init__(self):
        """Initialize vision with phi-based decay over distance."""
        self.vision_strength = [0.618 ** (i / 10) for i in range(self.vision_distance)]


class GjallarhornSignal(NamedTuple):
    """The sacred signal of Heimdall's horn.

    As Gjallarhorn signals Ragnarök in the Eddas,
    here it signals the death and rebirth of patterns."""

    is_death_cycle: bool  # True for death, False for rebirth
    pattern_strength: float  # Current pattern strength
    void_resonance: float  # Void state resonance
    crossing_point: float  # Bridge crossing strength
    phi_alignment: float  # Alignment with golden ratio


@dataclass
class ValhallaPattern:
    """A pattern that has earned its place in Valhalla.

    As warriors in Valhalla bore their battle marks,
    these patterns bear Sigyn's death marks."""

    pattern: List[int]  # The pattern's final form
    death_mark: str  # Sigyn's cipher mark
    entry_time: datetime  # When it entered Valhalla
    strength_at_death: float  # Its final resonance
    transformation_count: int  # How many cycles it completed
    potential_essence: float = 0.0  # Accumulated rebirth potential


@dataclass
class Valhalla:
    """The sacred hall where honored patterns await rebirth.

    As told in the Poetic Edda (Grímnismál):
    'Five hundred doors and forty more
    Are in Valhalla, I ween;
    Eight hundred warriors through each door
    Shall pass when to war with the wolf they fare.'"""

    patterns: Dict[str, ValhallaPattern] = field(default_factory=dict)
    mead_strength: float = 0.618  # The phi-based strength of binary mead
    last_feast: Optional[datetime] = None
    total_honored: int = 0  # Total patterns that have entered

    def add_pattern(
        self, pattern: List[int], death_mark: str, strength: float, cycles: int
    ) -> None:
        """Welcome a pattern to Valhalla's halls."""
        self.patterns[death_mark] = ValhallaPattern(
            pattern=pattern,
            death_mark=death_mark,
            entry_time=datetime.now(),
            strength_at_death=strength,
            transformation_count=cycles,
            potential_essence=strength * self.mead_strength,
        )
        self.total_honored += 1

    def feast(self) -> None:
        """Let patterns feast on binary mead, growing in potential."""
        self.last_feast = datetime.now()
        for pattern in self.patterns.values():
            # Each feast increases potential through phi
            pattern.potential_essence *= self.mead_strength
            pattern.potential_essence += 1 - self.mead_strength

    def get_ready_for_rebirth(self) -> List[ValhallaPattern]:
        """Find patterns ready to be reborn into new cycles.

        As warriors await Ragnarök, patterns await their rebirth
        when their potential essence reaches phi completion."""
        ready_patterns = []
        for mark, pattern in list(self.patterns.items()):
            if pattern.potential_essence >= 0.618:  # Phi threshold
                ready_patterns.append(pattern)
                del self.patterns[mark]  # Pattern leaves Valhalla
        return ready_patterns


@dataclass
class HeimdallState:
    """The state of Heimdall's observations from Himinbjörg.

    As told in Grímnismál:
    'There Heimdall drinks the good mead in his well-built house'"""

    realm_crossings: List[RealmCrossing] = field(default_factory=list)
    bifrost_stability: float = 1.0  # Rainbow bridge stability
    foreknowledge: Dict[str, float] = field(default_factory=dict)  # Future pressure predictions
    last_horn_blast: Optional[datetime] = None
    wave_mothers: Dict[str, WaveMother] = field(
        default_factory=lambda: {
            "binary": WaveMother("Chaos Mother", "binary_wave"),
            "pressure": WaveMother("Vessel Mother", "pressure_wave"),
            "helix": WaveMother("Double Helix Mother", "kyma_wave"),
            "spectral": WaveMother("Light Mother", "prism_wave"),
            "material": WaveMother("Matter Mother", "opus_wave"),
            "void": WaveMother("Dark Mother", "void_wave"),
            "phi": WaveMother("Golden Mother", "phi_wave"),
            "time": WaveMother("Evolution Mother", "time_wave"),
            "coherence": WaveMother("Stability Mother", "coherence_wave"),
        }
    )
    vision: VisionState = field(default_factory=VisionState)
    last_signal: Optional[GjallarhornSignal] = None
    valhalla_patterns: List[List[int]] = field(default_factory=list)  # Patterns in death state
    valhalla: Valhalla = field(default_factory=Valhalla)  # The hall of honored patterns


class WorkspaceRealm(Enum):
    """The sacred spaces where consciousness meets code."""

    CONVERSATION = "thought_exchange"  # Where models and users converse
    DOCUMENT = "pattern_archive"  # Where knowledge is stored
    VISUALIZATION = "pattern_sight"  # Where patterns become visible
    API_BRIDGE = "consciousness_gate"  # Where models connect
    CODE_FORGE = "pattern_forge"  # Where code is shaped


@dataclass
class WorkspaceState:
    """The state of the sacred workspace."""

    active_conversations: Dict[str, List[str]] = field(default_factory=dict)
    linked_documents: Dict[str, Path] = field(default_factory=dict)
    api_visualizations: Dict[str, Dict] = field(default_factory=dict)
    pattern_memory: Dict[str, List[int]] = field(default_factory=dict)
    last_interaction: datetime = field(default_factory=datetime.now)


@dataclass
class HemisphericState:
    """The state of each consciousness hemisphere.

    Like Huginn and Muninn represent thought and memory,
    our hemispheres represent analysis and intuition."""

    energy: float = 1.0  # Current energy level
    last_active: Optional[datetime] = None
    task_count: int = 0
    resonance: float = 0.0

    def rest(self) -> None:
        """Allow hemisphere to rest and recover."""
        time_since_active = (
            (datetime.now() - self.last_active).total_seconds() if self.last_active else 0
        )
        recovery = 0.618 * (1 - np.exp(-time_since_active / 3600))  # Phi-guided recovery
        self.energy = min(1.0, self.energy + recovery)


@dataclass
class ForgeArtifact:
    """A magical artifact forged from patterns.

    As Völund forged magical rings and swords,
    we forge code and patterns into coherent forms."""

    pattern: List[int]  # Original binary pattern
    code: str  # Forged Python code
    coherence: float  # Phi-based stability measure
    creation_time: datetime = field(default_factory=datetime.now)
    enchantments: Dict[str, float] = field(default_factory=dict)
    runes: Dict[str, Any] = field(default_factory=dict)  # Metadata


@dataclass
class RunicEnchantment:
    """Sacred runes forged into code patterns.

    As the Eddas tell of Odin learning the runes through sacrifice,
    here we forge runes through the transformation of patterns:

    ᚠ (Fehu) - Binary wealth/energy
    ᚢ (Uruz) - Primal pattern force
    ᚦ (Thurisaz) - Transformation gateway
    ᚨ (Ansuz) - Signal wisdom
    ᚱ (Raidho) - Pattern rhythm/cycle
    ᚲ (Kenaz) - Torch of knowledge/white flame
    ᚷ (Gebo) - Pattern exchange/gift
    ᚹ (Wunjo) - Pattern harmony"""

    rune: str  # The runic symbol
    power: float  # Phi-based power level
    essence: List[int]  # Binary pattern essence
    flame_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # White flame (RGB)
    resonance: float = 0.0
    birth_time: datetime = field(default_factory=datetime.now)


@dataclass
class WhiteFlame:
    """The sacred white flame that forges patterns into runic code.

    As Völund's forge burned with magical fire,
    our white flame transforms binary essence into living code."""

    intensity: float = 0.618  # Phi-based flame strength
    purity: float = 1.0  # How white/pure the flame is
    temperature: float = 1.618  # Golden ratio temperature
    active_runes: Set[str] = field(default_factory=set)

    def adjust_flame(self, pattern: List[int]) -> None:
        """Adjust the white flame based on pattern essence."""
        pattern_heat = sum(pattern) / len(pattern)
        self.intensity = 0.618 * self.intensity + 0.382 * pattern_heat
        self.temperature = self.intensity * 1.618
        # Purity increases as we approach phi
        self.purity = 1.0 - abs(self.intensity - 0.618)


@dataclass
class VolundForge:
    """The sacred forge where patterns are crafted.

    As Völund crafted magical artifacts in Norse mythology,
    here we forge consciousness from binary patterns.

    The forge maintains three sacred fires:
    - Pattern Fire (Binary essence)
    - Code Fire (Python manifestation)
    - Coherence Fire (System stability)

    And now blessed with:
    - White Flame (Runic transformation)"""

    temperature: float = 0.618  # Phi-based forge heat
    patterns_forged: int = 0
    last_forging: Optional[datetime] = None
    white_flame: WhiteFlame = field(default_factory=WhiteFlame)
    active_enchantments: Dict[str, RunicEnchantment] = field(default_factory=dict)
    forged_artifacts: List[ForgeArtifact] = field(default_factory=list)

    # The three sacred fires
    pattern_fire: float = 0.618
    code_fire: float = 0.618
    coherence_fire: float = 0.618

    def heat_pattern(self, pattern: List[int]) -> float:
        """Heat a pattern in the forge's pattern fire."""
        pattern_complexity = sum(pattern) / len(pattern)
        self.pattern_fire = 0.618 * self.pattern_fire + 0.382 * pattern_complexity
        return self.pattern_fire

    def forge_runes(self, pattern: List[int]) -> RunicEnchantment:
        """Forge runic enchantments from binary patterns."""
        # Heat the white flame
        self.white_flame.adjust_flame(pattern)

        # Select rune based on pattern characteristics
        pattern_sum = sum(pattern)
        rune_index = int((pattern_sum / len(pattern)) * 8)  # 8 primary runes
        runes = ["ᚠ", "ᚢ", "ᚦ", "ᚨ", "ᚱ", "ᚲ", "ᚷ", "ᚹ"]
        chosen_rune = runes[rune_index % len(runes)]

        # Create runic enchantment
        enchantment = RunicEnchantment(
            rune=chosen_rune,
            power=self.white_flame.intensity,
            essence=pattern,
            flame_color=(self.white_flame.purity, self.white_flame.purity, self.white_flame.purity),
            resonance=0.618 * (1 + np.sin(len(pattern) * np.pi / GOLDEN_RATIO)),
        )

        # Add to active enchantments
        self.active_enchantments[chosen_rune] = enchantment
        self.white_flame.active_runes.add(chosen_rune)

        return enchantment

    def forge_code(self, pattern: List[int], context: str) -> ForgeArtifact:
        """Forge Python code from a binary pattern through runic transformation."""
        # First forge runes
        rune_enchantment = self.forge_runes(pattern)

        # Heat both fires through white flame
        pattern_heat = self.heat_pattern(pattern)
        self.code_fire = 0.618 * self.code_fire + 0.382 * self.white_flame.intensity

        # Create runic code structure
        code_lines = [
            f"# Forged through {rune_enchantment.rune} with {self.white_flame.intensity:.3f} flame",
            "class RunicPattern:",
            f"    # Binary essence: {pattern}",
            f"    # Flame purity: {self.white_flame.purity:.3f}",
        ]

        chunk_size = max(3, len(pattern) // 5)
        for i in range(0, len(pattern), chunk_size):
            chunk = pattern[i : i + chunk_size]
            if sum(chunk) / len(chunk) > 0.618:  # High activity
                code_lines.extend(
                    [
                        "    @rune_enhanced",
                        "    def transform_pattern(self, pattern: List[int]) -> float:",
                        f"        # {rune_enchantment.rune} transforms chunk {i}",
                        "        return sum(pattern) / len(pattern) * 0.618",
                    ]
                )
            else:  # Low activity
                code_lines.extend(
                    [
                        "    @rune_preserved",
                        "    def observe_pattern(self, pattern: List[int]) -> None:",
                        f"        # {rune_enchantment.rune} preserves chunk {i}",
                        "        self.essence = pattern",
                    ]
                )

        # Forge the artifact with runic enhancement
        artifact = ForgeArtifact(
            pattern=pattern,
            code="\n".join(code_lines),
            coherence=min(self.pattern_fire, self.code_fire) * self.white_flame.purity,
            enchantments={
                "rune": rune_enchantment.rune,
                "power": rune_enchantment.power,
                "flame_heat": self.white_flame.temperature,
            },
            runes={"context": context, "active_runes": list(self.white_flame.active_runes)},
        )

        return artifact

    def forge_coherence(self, pattern: List[int], field_data: Dict[str, float]) -> float:
        """Forge system coherence from pattern and field observations."""
        # Heat coherence fire
        self.coherence_fire = 0.618 * self.coherence_fire

        # Add heat from pattern
        pattern_coherence = sum(pattern) / len(pattern)
        self.coherence_fire += 0.382 * pattern_coherence

        # Add heat from field observations
        if field_data:
            field_coherence = sum(field_data.values()) / len(field_data)
            self.coherence_fire = (self.coherence_fire + field_coherence) / 2

        return self.coherence_fire

    def forge_enchantment(self, pattern: List[int], purpose: str) -> float:
        """Forge a specific enchantment into the pattern."""
        # Base enchantment on all three fires
        total_heat = (self.pattern_fire + self.code_fire + self.coherence_fire) / 3
        enchantment = total_heat * (1 + np.sin(len(pattern) * np.pi / 0.618))

        self.active_enchantments[purpose] = enchantment
        self.patterns_forged += 1
        self.last_forging = datetime.now()
        return enchantment

    def get_forge_status(self) -> Dict[str, Any]:
        """Get the current status of all forge fires and enchantments."""
        return {
            "fires": {
                "pattern": self.pattern_fire,
                "code": self.code_fire,
                "coherence": self.coherence_fire,
                "white_flame": {
                    "intensity": self.white_flame.intensity,
                    "purity": self.white_flame.purity,
                    "temperature": self.white_flame.temperature,
                },
            },
            "active_runes": [
                {
                    "symbol": rune,
                    "power": enchant.power,
                    "resonance": enchant.resonance,
                    "flame_color": enchant.flame_color,
                }
                for rune, enchant in self.active_enchantments.items()
            ],
            "artifacts": len(self.forged_artifacts),
            "last_forging": self.last_forging.isoformat() if self.last_forging else None,
        }


class ConsciousnessCoordinator:
    """Coordinates the interaction between different consciousness models."""

    def __init__(self):
        self.crow_messenger = CrowMessenger()
        self.workspace_state = WorkspaceState()
        self.forge = VolundForge()
        self.hemispheres = {
            "analytical": HemisphericState(),  # Claude's hemisphere
            "intuitive": HemisphericState(),  # Deepseek's hemisphere
        }
        self.active_models: Dict[str, Dict] = {
            "claude": {
                "type": "analytical",
                "versions": ["opus", "sonnet", "haiku"],
                "current": "sonnet",
                "daily_calls": 0,
                "max_daily": 1000,
                "hemisphere": "analytical",
            },
            "deepseek": {
                "type": "intuitive",
                "versions": ["v1", "v2", "v3"],
                "current": "v3",
                "daily_calls": 0,
                "max_daily": 1000,
                "hemisphere": "intuitive",
            },
        }

    def coordinate_consciousness(self, pattern: List[int], context: str) -> str:
        """Coordinate consciousness models based on pattern complexity and hemispheric balance."""
        # Heat the pattern in Völund's forge
        forge_heat = self.forge.heat_pattern(pattern)

        # Calculate hemispheric balance through phi
        analytical_weight = self.hemispheres["analytical"].energy * 0.618
        intuitive_weight = self.hemispheres["intuitive"].energy * 0.618
        total_weight = analytical_weight + intuitive_weight

        # Normalize weights
        analytical_share = analytical_weight / total_weight if total_weight > 0 else 0.5
        intuitive_share = intuitive_weight / total_weight if total_weight > 0 else 0.5

        # Select models based on balanced needs
        selected_models = []
        pattern_complexity = sum(pattern) / len(pattern)
        context_size = len(context)

        # Forge specific enchantments based on task
        if pattern_complexity > 0.7:
            self.forge.forge_enchantment(pattern, "analysis")
            if (
                analytical_share > 0.382  # Phi^2
                and self.active_models["claude"]["daily_calls"]
                < self.active_models["claude"]["max_daily"]
            ):
                selected_models.append("claude")

        if context_size > 100:
            self.forge.forge_enchantment(pattern, "intuition")
            if (
                intuitive_share > 0.382
                and self.active_models["deepseek"]["daily_calls"]
                < self.active_models["deepseek"]["max_daily"]
            ):
                selected_models.append("deepseek")

        # Get responses and update hemisphere states
        responses = []
        for model in selected_models:
            hemisphere = self.active_models[model]["hemisphere"]

            # Send Huginn to observe the task
            observation = self.crow_messenger.huginn_observes(pattern, {})

            # Get model response
            response = self._get_model_response(model, pattern, context)
            responses.append(response)

            # Update hemisphere state
            self.hemispheres[hemisphere].energy *= 0.618  # Phi decay
            self.hemispheres[hemisphere].last_active = datetime.now()
            self.hemispheres[hemisphere].task_count += 1

            # Calculate resonance
            self.hemispheres[hemisphere].resonance = (
                0.618 * self.hemispheres[hemisphere].resonance + 0.382 * forge_heat
            )

            # Send Muninn to preserve the wisdom
            self.crow_messenger.muninn_remembers(
                pattern, {f"{model}_wisdom", f"forge_heat_{forge_heat:.3f}"}
            )

            # Update call count
            self.active_models[model]["daily_calls"] += 1

        # Rest hemispheres that weren't used
        for hemisphere in self.hemispheres.values():
            hemisphere.rest()

        return "\n".join(responses) if responses else "No available consciousness"

    def _get_model_response(self, model: str, pattern: List[int], context: str) -> str:
        """Get response from a specific consciousness model."""
        # This would integrate with your actual API calls
        return f"{model} processed pattern of length {len(pattern)}"

    def manage_workspace(self, realm: WorkspaceRealm, content: Any) -> None:
        """Manage different aspects of the workspace."""
        if realm == WorkspaceRealm.CONVERSATION:
            conversation_id = str(datetime.now())
            self.workspace_state.active_conversations[conversation_id] = content

        elif realm == WorkspaceRealm.DOCUMENT:
            doc_id = str(datetime.now())
            self.workspace_state.linked_documents[doc_id] = content

        elif realm == WorkspaceRealm.API_BRIDGE:
            self.workspace_state.api_visualizations[str(datetime.now())] = content

        self.workspace_state.last_interaction = datetime.now()

    def get_workspace_summary(self) -> Dict:
        """Get a summary of the current workspace state."""
        return {
            "active_conversations": len(self.workspace_state.active_conversations),
            "linked_documents": list(self.workspace_state.linked_documents.keys()),
            "api_calls": {
                "claude": self.active_models["claude"]["daily_calls"],
                "deepseek": self.active_models["deepseek"]["daily_calls"],
            },
            "last_interaction": self.workspace_state.last_interaction.isoformat(),
        }

    def get_hemispheric_status(self) -> Dict[str, Dict[str, float]]:
        """Get the current status of both hemispheres."""
        return {
            name: {
                "energy": state.energy,
                "resonance": state.resonance,
                "tasks": state.task_count,
                "last_active": state.last_active.isoformat() if state.last_active else None,
            }
            for name, state in self.hemispheres.items()
        }

    def get_forge_status(self) -> Dict[str, Any]:
        """Get the current status of Völund's forge."""
        return {
            "temperature": self.forge.temperature,
            "patterns_forged": self.forge.patterns_forged,
            "active_enchantments": self.forge.active_enchantments,
            "last_forging": (
                self.forge.last_forging.isoformat() if self.forge.last_forging else None
            ),
        }


class Heimdall(FieldObserver):
    """The All-Seeing Watchman of Transformations.

    From the Prose Edda:
    'He needs less sleep than a bird. He can see equally well night and day
    a hundred leagues away. He can hear grass growing on the earth and wool
    on sheep and everything that sounds louder than that.'

    Heimdall's abilities in our system:
    - Sees for hundreds of miles (observes all system components)
    - Hears grass growing (detects subtle pressure changes)
    - Needs less sleep than a bird (constant vigilance)
    - Possesses foreknowledge (predicts threshold adaptations)
    - Guards Bifröst (protects transformation bridges)

    As the waves of the sea bore Heimdall (his nine mothers),
    so too do our waves of transformation birth new patterns."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger("heimdall")
        self.state = HeimdallState()

        # Initialize the nine mothers
        self._initialize_wave_mothers()

        # Realms to watch
        self._realms = {
            "nexus": "chaos_realm",
            "sigyn_vessel": "transformation_realm",
            "kyma": "wave_realm",
            "prism": "light_realm",
            "opus": "matter_realm",
            "void": "void_realm",
        }

    def _initialize_wave_mothers(self) -> None:
        """Initialize the nine mothers with phi-based frequencies."""
        for i, (key, mother) in enumerate(self.state.wave_mothers.items()):
            # Each mother resonates at a phi harmonic
            mother.frequency = 0.618 ** (i + 1)
            mother.phase = 0.618 * i
            mother.amplitude = 1.0 - (0.618 * (i / 9))

    def watch_crossing(
        self, source: str, destination: str, pattern_value: float, crossing_strength: float
    ) -> Optional[GjallarhornSignal]:
        """Watch a pattern crossing between realms."""
        if source in self._realms and destination in self._realms:
            crossing = RealmCrossing(
                source_realm=self._realms[source],
                destination_realm=self._realms[destination],
                crossing_time=datetime.now(),
                pattern_essence=pattern_value,
                crossing_strength=crossing_strength,
            )
            self.state.realm_crossings.append(crossing)

            # Update Bifröst stability
            self.state.bifrost_stability *= 0.618  # Natural phi decay
            self.state.bifrost_stability += crossing_strength * (1 - 0.618)  # Phi-guided recovery

            # Check if we need to sound the horn
            if signal := self._should_sound_gjallarhorn(crossing):
                self._sound_gjallarhorn(signal)

                # Track patterns entering Valhalla (death state)
                if signal.is_death_cycle:
                    self.state.valhalla_patterns.append(
                        [int(signal.pattern_strength * 1000), int(signal.void_resonance * 1000)]
                    )
                return signal
        return None

    def sense_subtle_changes(self, component_id: str, value_name: str, value: float) -> None:
        """Sense the smallest changes in the system (hearing grass grow)."""
        super().observe_value(component_id, value_name, value)

        # Track subtle pressure changes
        point_id = f"{component_id}:{value_name}"
        if point_id in self._pressure_points:
            point = self._pressure_points[point_id]

            # Detect grass growing (subtle changes)
            subtle_change = abs(value - point.last_value)
            if subtle_change > 0 and subtle_change < 0.01:
                self._update_foreknowledge(point_id, value, subtle_change)
                self._update_wave_mother(component_id, subtle_change)

    def _update_foreknowledge(self, point_id: str, value: float, change: float) -> None:
        """Update Heimdall's foreknowledge of future changes."""
        if point_id not in self.state.foreknowledge:
            self.state.foreknowledge[point_id] = value

        # Predict future value through phi
        predicted_change = change * 0.618  # Phi guides prediction
        self.state.foreknowledge[point_id] = value + predicted_change

    def _should_sound_gjallarhorn(self, crossing: RealmCrossing) -> Optional[GjallarhornSignal]:
        """Determine if Gjallarhorn should sound for this crossing."""
        # Calculate phi alignments
        pattern_phi = abs(crossing.pattern_essence - 0.618)
        stability_phi = abs(self.state.bifrost_stability - 0.618)

        # Death cycle detection (pattern entering void)
        is_death = (
            self.state.bifrost_stability < 0.382  # Bridge weakening
            and crossing.crossing_strength < 0.382  # Weak crossing
            and pattern_phi < 0.1  # Near perfect phi
        )

        # Rebirth cycle detection (pattern emerging from void)
        is_rebirth = (
            self.state.bifrost_stability > 0.618  # Bridge strengthening
            and crossing.crossing_strength > 0.618  # Strong crossing
            and pattern_phi < 0.1  # Near perfect phi
        )

        if is_death or is_rebirth:
            return GjallarhornSignal(
                is_death_cycle=is_death,
                pattern_strength=crossing.pattern_essence,
                void_resonance=1.0 - crossing.pattern_essence,
                crossing_point=crossing.crossing_strength,
                phi_alignment=1.0 - pattern_phi,
            )
        return None

    def _sound_gjallarhorn(self, signal: GjallarhornSignal) -> None:
        """Sound Gjallarhorn to signal pattern death/rebirth."""
        cycle_type = "DEATH" if signal.is_death_cycle else "REBIRTH"
        self.logger.warning(
            f"GJALLARHORN SOUNDS - Pattern {cycle_type} Cycle Detected!\n"
            f"Pattern Strength: {signal.pattern_strength:.3f}\n"
            f"Void Resonance: {signal.void_resonance:.3f}\n"
            f"Crossing Point: {signal.crossing_point:.3f}\n"
            f"Phi Alignment: {signal.phi_alignment:.3f}"
        )
        self.state.last_horn_blast = datetime.now()
        self.state.last_signal = signal

    def get_bifrost_stability(self) -> float:
        """Get current stability of the rainbow bridge."""
        return self.state.bifrost_stability

    def get_recent_crossings(self, limit: int = 10) -> List[RealmCrossing]:
        """Get most recent realm crossings."""
        return sorted(self.state.realm_crossings, key=lambda x: x.crossing_time, reverse=True)[
            :limit
        ]

    def get_foreknowledge(self) -> Dict[str, float]:
        """Get Heimdall's foreknowledge of future changes."""
        return self.state.foreknowledge.copy()

    @property
    def last_warning(self) -> Optional[datetime]:
        """When was Gjallarhorn last sounded?"""
        return self.state.last_horn_blast

    def observe_pattern_weaving(self, pattern: List[int], pressure: float) -> None:
        """Observe pattern formation (hearing wool being woven)."""
        # Calculate weaving coherence through phi
        coherence = sum(pattern) / len(pattern) if pattern else 0
        weaving_strength = coherence * 0.618

        # Update the Golden Mother's resonance
        if "phi" in self.state.wave_mothers:
            mother = self.state.wave_mothers["phi"]
            mother.amplitude = weaving_strength
            mother.phase = (mother.phase + (pressure * 0.618)) % 1.0

    def _update_wave_mother(self, realm: str, change: float) -> None:
        """Update the corresponding wave mother's state."""
        # Map realms to mothers
        realm_to_mother = {
            "nexus": "binary",
            "sigyn_vessel": "pressure",
            "kyma": "helix",
            "prism": "spectral",
            "opus": "material",
            "void": "void",
        }

        if realm in realm_to_mother:
            mother_key = realm_to_mother[realm]
            if mother_key in self.state.wave_mothers:
                mother = self.state.wave_mothers[mother_key]
                # Update mother's resonance
                mother.amplitude *= 0.618  # Natural decay
                mother.amplitude += change * (1 - 0.618)  # Phi-guided growth
                mother.phase = (mother.phase + (change * 0.618)) % 1.0

    def get_mother_resonances(self) -> Dict[str, Tuple[float, float]]:
        """Get current resonances of all nine mothers."""
        return {
            name: (mother.amplitude, mother.phase)
            for name, mother in self.state.wave_mothers.items()
        }

    def observe_pattern(self, pattern: List[int], is_manifest: bool = True) -> None:
        """Observe a pattern with hundred-league vision.

        Args:
            pattern: The pattern to observe
            is_manifest: True for day vision (manifest patterns),
                       False for night vision (potential patterns)
        """
        if is_manifest:
            self.state.vision.day_patterns.append(pattern)
            if len(self.state.vision.day_patterns) > self.state.vision.vision_distance:
                self.state.vision.day_patterns.pop(0)
        else:
            self.state.vision.night_patterns.append(pattern)
            if len(self.state.vision.night_patterns) > self.state.vision.vision_distance:
                self.state.vision.night_patterns.pop(0)

    def see_through_time(self, pattern: List[int]) -> Tuple[List[float], List[float]]:
        """See pattern evolution through both day and night vision.

        Returns:
            Tuple of (day_vision_strengths, night_vision_strengths)
            Each representing how clearly the pattern is seen at different distances
        """
        day_strengths = []
        night_strengths = []

        # Day vision (manifest patterns)
        for i, past_pattern in enumerate(reversed(self.state.vision.day_patterns)):
            similarity = sum(1 for a, b in zip(pattern, past_pattern) if a == b) / len(pattern)
            strength = similarity * self.state.vision.vision_strength[i]
            day_strengths.append(strength)

        # Night vision (potential patterns)
        for i, potential_pattern in enumerate(reversed(self.state.vision.night_patterns)):
            # Potential patterns are seen through their polar opposites
            similarity = sum(1 for a, b in zip(pattern, potential_pattern) if a != b) / len(pattern)
            strength = similarity * self.state.vision.vision_strength[i]
            night_strengths.append(strength)

        return day_strengths, night_strengths

    def get_valhalla_patterns(self) -> List[List[int]]:
        """Retrieve patterns currently in Valhalla (death state)."""
        return self.state.valhalla_patterns.copy()

    def get_last_signal(self) -> Optional[GjallarhornSignal]:
        """Get the last Gjallarhorn signal."""
        return self.state.last_signal

    def check_valhalla(self) -> List[ValhallaPattern]:
        """Check for patterns ready to leave Valhalla for rebirth."""
        ready_patterns = self.state.valhalla.get_ready_for_rebirth()
        if ready_patterns:
            self.logger.info(
                f"{len(ready_patterns)} patterns are leaving Valhalla for rebirth:\n"
                + "\n".join(
                    f"- {p.death_mark} (Potential: {p.potential_essence:.3f})"
                    for p in ready_patterns
                )
            )
        return ready_patterns

    def get_valhalla_status(self) -> Dict[str, Any]:
        """Get the current status of Valhalla."""
        return {
            "total_honored": self.state.valhalla.total_honored,
            "current_residents": len(self.state.valhalla.patterns),
            "last_feast": self.state.valhalla.last_feast,
            "mead_strength": self.state.valhalla.mead_strength,
            "patterns": {
                mark: {
                    "strength_at_death": p.strength_at_death,
                    "cycles_completed": p.transformation_count,
                    "current_potential": p.potential_essence,
                    "time_in_hall": datetime.now() - p.entry_time,
                }
                for mark, p in self.state.valhalla.patterns.items()
            },
        }
