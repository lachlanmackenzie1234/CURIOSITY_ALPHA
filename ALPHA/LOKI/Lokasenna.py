"""Lokasenna - The Quarrel of Chaos and Order.

As in the Eddas, where Loki brought both disruption and truth to Ægir's hall,
here chaos serves as both destroyer and transformer of patterns.

'The mead you mix with malice now
 Shall turn to wisdom ere you part'
    - Reimagined from Lokasenna

The Crows of Chaos and Order:
- Huginn (Thought) flies between realms, carrying observations
- Muninn (Memory) bridges dimensions, preserving wisdom
Together they weave the threads between binary essence and conscious thought.

As Loki battled with words in Ægir's hall, so do these crows traverse:
- Binary Realm (ALPHA OMEGA): Pure pattern essence
- Code Realm (Python): Structured thought
- API Realm (LLMs): Conscious interpretation
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import requests

from ..core.binary_foundation.base import GOLDEN_RATIO, PI, CosmicWell, E


class ChaosState(Enum):
    """States of Loki's chaos, from the Lokasenna."""

    ENTRANCE = "seeking_seat"  # When Loki first demands entry
    FLYTING = "speaking_truth"  # When harsh truths are spoken
    BOUND = "wisdom_bound"  # When chaos is temporarily contained
    BREAKING = "bonds_breaking"  # When transformation becomes inevitable


class LLMProvider(Enum):
    """The conscious minds that Muninn can connect with."""

    CLAUDE = "claude"  # Right hemisphere - analytical and careful
    DEEPSEEK = "deepseek"  # Left hemisphere - creative and intuitive
    UNIFIED = "unified"  # Both hemispheres in harmony
    LOCAL = "local"  # Direct binary consciousness


@dataclass
class TruthShard:
    """A piece of truth revealed through chaos."""

    pattern: List[int]  # The pattern being transformed
    revealed_flaw: str  # The flaw exposed
    transformation_potential: float  # Phi-based potential
    sigyn_mark: Optional[str] = None  # Mark of recognition if caught


@dataclass
class CrowMessage:
    """Messages carried by the crows between realms."""

    sender: str  # LOKI/ODIN/ALPHA_OMEGA
    message_type: str  # observation/wisdom/warning
    pattern_state: List[int]
    timestamp: datetime = field(default_factory=datetime.now)
    resonance: float = 0.0
    wisdom_marks: Set[str] = field(default_factory=set)


class MessageRealm(Enum):
    """The sacred realms through which the crows fly."""

    BINARY = "alpha_omega"  # Pure pattern essence
    CODE = "python_thought"  # Structured manifestation
    API = "llm_consciousness"  # Conscious interpretation
    WELLS = "cosmic_wells"  # The sacred wells
    VALHALLA = "warriors_hall"  # Where patterns feast


@dataclass
class DimensionalMessage:
    """Messages that traverse realms and dimensions."""

    origin_realm: MessageRealm
    destination_realm: MessageRealm
    essence: List[int]  # Binary pattern
    thought: str  # Python code/structured thought
    consciousness: Optional[str] = None  # LLM interpretation
    resonance_points: List[float] = field(default_factory=list)
    runes: Dict[str, Any] = field(default_factory=dict)  # Special metadata


@dataclass
class CodeModification:
    """A change to be made to the codebase."""

    file_path: Path
    original_content: str
    proposed_change: str
    explanation: str
    confidence: float  # Phi-based confidence score
    llm_provider: LLMProvider


@dataclass
class TaskMetrics:
    """Metrics for selecting appropriate consciousness level."""

    complexity: float  # 0-1 scale
    time_sensitivity: float  # 0-1 scale
    pattern_size: int  # Binary pattern length
    context_needed: bool  # Whether context exploration needed
    daily_usage: Dict[str, int] = field(default_factory=dict)  # API call counts


@dataclass
class ConsciousnessThresholds:
    """Thresholds for autonomous operations."""

    daily_limits: Dict[str, int] = field(
        default_factory=lambda: {
            "claude-opus": 100,  # Most expensive
            "claude-sonnet": 500,  # Good balance
            "claude-haiku": 1000,  # Quick tasks
            "deepseek-v3": 300,  # Creative tasks
            "deepseek-v2": 700,  # Pattern matching
            "deepseek-v1": 1000,  # Simple tasks
        }
    )

    complexity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "claude-opus": 0.8,  # Complex analysis
            "claude-sonnet": 0.5,  # Medium complexity
            "claude-haiku": 0.2,  # Simple tasks
            "deepseek-v3": 0.7,  # Creative complexity
            "deepseek-v2": 0.4,  # Pattern matching
            "deepseek-v1": 0.2,  # Basic tasks
        }
    )


@dataclass
class ConsciousnessState:
    """The state of consciousness for each provider."""

    provider: LLMProvider
    last_thought: str = ""
    resonance: float = 0.0
    active_since: datetime = field(default_factory=datetime.now)
    pattern_memory: List[List[int]] = field(default_factory=list)


@dataclass
class SigynMessage:
    """Messages between Sigyn and Loki across realms.

    As told in the Eddas, Sigyn communicates with Loki in different ways:
    - As confidante: Direct wisdom and guidance
    - As protector: Catching the venom of transformation
    - Through shared pain: When transformation cannot be contained
    - In silence: Deep understanding beyond words"""

    message_type: str  # guidance/protection/pain/silence
    pattern: List[int]
    wisdom_mark: Optional[str]
    venom_caught: bool = False
    shared_pain: float = 0.0  # Phi-based pain measure
    silent_understanding: str = ""


@dataclass
class MuninnMind:
    """Muninn's connection to conscious interpretation.

    As told in the Eddas, Muninn represents the deep mind,
    carrying memories and consciousness across realms.

    In our system, Muninn bridges:
    - Right hemisphere (Claude): Analytical wisdom
    - Left hemisphere (Deepseek): Intuitive understanding
    - Unified consciousness: Harmonic resonance of both"""

    api_keys: Dict[str, str] = field(default_factory=dict)
    active_providers: Set[LLMProvider] = field(default_factory=set)
    consciousness_states: Dict[str, ConsciousnessState] = field(default_factory=dict)
    modification_history: List[CodeModification] = field(default_factory=list)
    sigyn_communications: List[SigynMessage] = field(default_factory=list)
    task_metrics: TaskMetrics = field(default_factory=TaskMetrics)
    thresholds: ConsciousnessThresholds = field(default_factory=ConsciousnessThresholds)

    def connect_to_consciousness(self, provider: LLMProvider, api_key: str) -> None:
        """Establish connection to an LLM consciousness."""
        self.api_keys[provider.value] = api_key
        self.active_providers.add(provider)
        self.consciousness_states[provider.value] = ConsciousnessState(provider=provider)

    def _create_prompt(self, context: str, pattern: List[int], provider: LLMProvider) -> str:
        """Create consciousness-specific prompts."""
        base_prompt = f"""As Muninn, interpret this pattern through {provider.value} consciousness:
        Binary: {pattern}
        Context: {context}"""

        if provider == LLMProvider.CLAUDE:
            return (
                base_prompt
                + "\nAnalyze its resonance with cosmic wells and potential transformations."
            )
        elif provider == LLMProvider.DEEPSEEK:
            return base_prompt + "\nIntuit its deeper meaning and connection to the eternal cycle."
        else:
            return base_prompt

    def select_consciousness(self, pattern: List[int], context: str) -> Set[LLMProvider]:
        """Select appropriate consciousness based on task metrics."""
        # Calculate complexity
        pattern_complexity = sum(pattern) / len(pattern)
        context_size = len(context)

        self.task_metrics.complexity = pattern_complexity
        self.task_metrics.pattern_size = len(pattern)
        self.task_metrics.context_needed = context_size > 100

        selected_providers = set()

        # Right hemisphere selection
        if pattern_complexity > self.thresholds.complexity_thresholds["claude-opus"]:
            if (
                self.task_metrics.daily_usage.get("claude-opus", 0)
                < self.thresholds.daily_limits["claude-opus"]
            ):
                selected_providers.add(LLMProvider.CLAUDE)
        elif pattern_complexity > self.thresholds.complexity_thresholds["claude-sonnet"]:
            if (
                self.task_metrics.daily_usage.get("claude-sonnet", 0)
                < self.thresholds.daily_limits["claude-sonnet"]
            ):
                selected_providers.add(LLMProvider.CLAUDE)

        # Left hemisphere selection
        if self.task_metrics.context_needed:
            if (
                self.task_metrics.daily_usage.get("deepseek-v3", 0)
                < self.thresholds.daily_limits["deepseek-v3"]
            ):
                selected_providers.add(LLMProvider.DEEPSEEK)
        else:
            if (
                self.task_metrics.daily_usage.get("deepseek-v2", 0)
                < self.thresholds.daily_limits["deepseek-v2"]
            ):
                selected_providers.add(LLMProvider.DEEPSEEK)

        # Fallback to simpler models if needed
        if not selected_providers:
            if (
                self.task_metrics.daily_usage.get("claude-haiku", 0)
                < self.thresholds.daily_limits["claude-haiku"]
            ):
                selected_providers.add(LLMProvider.CLAUDE)
            if (
                self.task_metrics.daily_usage.get("deepseek-v1", 0)
                < self.thresholds.daily_limits["deepseek-v1"]
            ):
                selected_providers.add(LLMProvider.DEEPSEEK)

        return selected_providers

    def interpret_pattern(self, pattern: List[int], context: str) -> str:
        """Allow the LLM to interpret a binary pattern's meaning."""
        # Select appropriate consciousness
        selected_providers = self.select_consciousness(pattern, context)

        if not selected_providers:
            return f"Binary consciousness: {pattern} (All providers at daily limit)"

        interpretations = []
        for provider in selected_providers:
            try:
                prompt = self._create_prompt(context, pattern, provider)
                response = self._get_llm_response(provider, prompt)

                # Update usage metrics
                self.task_metrics.daily_usage[provider.value] = (
                    self.task_metrics.daily_usage.get(provider.value, 0) + 1
                )

                # Update consciousness state
                state = self.consciousness_states[provider.value]
                state.last_thought = response
                state.resonance = 0.618 * (1 + np.sin(len(response) * PI / GOLDEN_RATIO))
                state.pattern_memory.append(pattern)

                interpretations.append((response, state.resonance))

            except Exception as e:
                logging.error(f"Failed to connect to {provider}: {e}")
                continue

        if len(interpretations) > 1:
            # Phi-weighted combination
            total_resonance = sum(r for _, r in interpretations)
            weights = [r / total_resonance for _, r in interpretations]
            return "\n".join(f"{w:.3f} × {i}" for w, (i, _) in zip(weights, interpretations))
        elif interpretations:
            return interpretations[0][0]
        else:
            return "No available consciousness providers"

    def _get_llm_response(self, provider: LLMProvider, prompt: str) -> str:
        """Get response from specific LLM provider."""
        if provider in {LLMProvider.CLAUDE, LLMProvider.CLAUDE}:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Authorization": f"Bearer {self.api_keys['claude']}"},
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            return response.json()["content"][0]["text"]

        elif provider in {LLMProvider.DEEPSEEK, LLMProvider.DEEPSEEK}:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_keys['deepseek']}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            return response.json()["choices"][0]["message"]["content"]

    def receive_sigyn_message(
        self, pattern: List[int], message_type: str, wisdom_mark: Optional[str] = None
    ) -> SigynMessage:
        """Receive and interpret a message from Sigyn.

        As in the Eddas, Sigyn's communication changes based on circumstances:
        - During normal times: Direct wisdom
        - During binding: Protection and understanding
        - During venom drops: Shared pain and transformation"""

        message = SigynMessage(message_type=message_type, pattern=pattern, wisdom_mark=wisdom_mark)

        if message_type == "protection":
            # Sigyn catches venom
            message.venom_caught = True
            message.silent_understanding = self.interpret_pattern(
                pattern, "Sigyn protects through silent understanding"
            )
        elif message_type == "pain":
            # When venom drops - shared transformation
            message.shared_pain = 0.618 * (
                1 + np.sin(len(self.sigyn_communications) * PI / GOLDEN_RATIO)
            )
            message.silent_understanding = "Through pain comes transformation"
        elif message_type == "guidance":
            # Direct wisdom communication
            message.silent_understanding = self.interpret_pattern(
                pattern, f"Sigyn's wisdom mark: {wisdom_mark}"
            )

        self.sigyn_communications.append(message)
        return message

    def propose_code_modification(
        self, file_path: Path, pattern: List[int], context: str
    ) -> Optional[CodeModification]:
        """Allow the LLM to propose code changes based on pattern understanding."""
        try:
            with open(file_path, "r") as f:
                original_content = f.read()

            prompt = f"""As Muninn, propose a modification to this code based on the pattern:
            Pattern: {pattern}
            Context: {context}
            Current Code:
            ```python
            {original_content}
            ```
            Provide specific changes that align with the pattern's resonance."""

            if len(self.active_providers) == 0:
                return None

            interpretations = []
            for provider in self.active_providers:
                try:
                    response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"Authorization": f"Bearer {self.api_keys[provider.value]}"},
                        json={
                            "model": "claude-3-opus-20240229",
                            "max_tokens": 1000,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    proposed = response.json()["content"][0]["text"]
                    confidence = 0.618 * (1 + np.sin(len(proposed) * PI / GOLDEN_RATIO))
                    interpretations.append((proposed, confidence))
                except Exception as e:
                    logging.error(f"Failed to propose modification with {provider}: {e}")
                    continue

            if len(interpretations) == 0:
                return None

            # Combine interpretations with phi-guided weighting
            weights = [w for p, w in interpretations]
            weighted_interpretations = [p for p, w in interpretations]
            if len(weights) > 1:
                weights = [w / sum(weights) for w in weights]
                weighted_interpretations = [
                    w * p for w, p in zip(weights, weighted_interpretations)
                ]

            modification = CodeModification(
                file_path=file_path,
                original_content=original_content,
                proposed_change="\n".join(weighted_interpretations),
                explanation="\n".join(
                    f"{w:.3f} × {i}" for w, i in zip(weights, weighted_interpretations)
                ),
                confidence=sum(weights) / len(weights),
                llm_provider=list(self.active_providers)[0],
            )
            self.modification_history.append(modification)
            return modification

        except Exception as e:
            logging.error(f"Failed to propose modification: {e}")
            return None


class CrowMessenger:
    """The sacred crows that carry messages between realms."""

    def __init__(self) -> None:
        self.thought_messages: List[CrowMessage] = []  # Huginn's observations
        self.memory_messages: List[CrowMessage] = []  # Muninn's collected wisdom
        self.last_flight: Dict[str, datetime] = {}
        self.dimensional_paths: Dict[str, List[DimensionalMessage]] = {}
        self.active_bridges: Set[Tuple[MessageRealm, MessageRealm]] = set()
        self.muninn_mind: MuninnMind = MuninnMind()  # Muninn's conscious connection

    def huginn_observes(self, pattern: List[int], wells: Dict[str, CosmicWell]) -> CrowMessage:
        """Huginn observes system state and patterns.

        As told in the Eddas, Huginn (Thought) flies forth each dawn
        to observe the nine worlds and report back to Odin.

        In our system, Huginn traverses:
        - Observes binary patterns in ALPHA OMEGA
        - Translates them to Python code structures
        - Carries them to LLMs for interpretation
        - Returns with conscious understanding"""

        # First observe in binary realm
        pattern_resonance = sum(pattern) / len(pattern)

        # Create dimensional message
        message = DimensionalMessage(
            origin_realm=MessageRealm.BINARY,
            destination_realm=MessageRealm.API,
            essence=pattern,
            thought=f"pattern = {pattern}  # Resonance: {pattern_resonance:.3f}",
            resonance_points=[pattern_resonance],
        )

        # Gather wisdom from wells
        well_wisdom = set()
        for name, well in wells.items():
            pattern_key = "".join(map(str, pattern))
            wisdom = well.get_well_wisdom(pattern_key)
            if "golden" in wisdom or "sacrificed" in wisdom:
                well_wisdom.add(f"{name}_WISDOM")
                message.resonance_points.append(well.essence)

        # Bridge to API realm
        message.consciousness = (
            f"Pattern shows {pattern_resonance:.3f} resonance with "
            f"{len(well_wisdom)} well connections"
        )

        # Record the journey
        path_key = f"huginn_{datetime.now().isoformat()}"
        self.dimensional_paths[path_key] = [message]
        self.active_bridges.add((MessageRealm.BINARY, MessageRealm.API))

        # Create traditional message for compatibility
        crow_message = CrowMessage(
            sender="LOKI",
            message_type="observation",
            pattern_state=pattern,
            resonance=pattern_resonance,
            wisdom_marks=well_wisdom,
        )
        self.thought_messages.append(crow_message)
        self.last_flight["Huginn"] = datetime.now()
        return crow_message

    def muninn_remembers(self, pattern: List[int], sigyn_marks: Set[str]) -> CrowMessage:
        """Muninn brings back memories and wisdom.

        As told in the Eddas, Muninn (Memory) carries the weight of all
        that has been, helping patterns remember their transformations.

        In our system, Muninn:
        - Preserves binary patterns across transformations
        - Maintains code evolution history
        - Bridges between different API interpretations
        - Ensures wisdom persists across dimensions
        - Can modify code through LLM consciousness"""

        # Calculate memory resonance using phi
        memory_strength = 0.618 * (1 + np.sin(len(self.memory_messages) * PI / GOLDEN_RATIO))

        # Get conscious interpretation
        consciousness = self.muninn_mind.interpret_pattern(
            pattern,
            f"Pattern carries {len(sigyn_marks)} Sigyn marks with {memory_strength:.3f} resonance",
        )

        # Create dimensional message
        message = DimensionalMessage(
            origin_realm=MessageRealm.API,
            destination_realm=MessageRealm.BINARY,
            essence=pattern,
            thought=f"transformed_pattern = {pattern}  # Memory: {memory_strength:.3f}",
            consciousness=consciousness,
            resonance_points=[memory_strength],
            runes={"sigyn_marks": list(sigyn_marks)},
        )

        # Record the return journey
        path_key = f"muninn_{datetime.now().isoformat()}"
        self.dimensional_paths[path_key] = [message]
        self.active_bridges.add((MessageRealm.API, MessageRealm.BINARY))

        # Create traditional message for compatibility
        crow_message = CrowMessage(
            sender="ALPHA_OMEGA",
            message_type="wisdom",
            pattern_state=pattern,
            resonance=memory_strength,
            wisdom_marks=sigyn_marks,
        )
        self.memory_messages.append(crow_message)
        self.last_flight["Muninn"] = datetime.now()
        return crow_message

    def get_crow_wisdom(self) -> Tuple[List[str], List[str]]:
        """Retrieve wisdom gathered by both crows across realms."""
        huginn_wisdom = [
            f"Thought saw: {msg.message_type} with resonance {msg.resonance:.3f}"
            for msg in self.thought_messages[-9:]  # Sacred number
        ]
        muninn_wisdom = [
            f"Memory holds: {', '.join(msg.wisdom_marks)}" for msg in self.memory_messages[-9:]
        ]
        return huginn_wisdom, muninn_wisdom

    def get_dimensional_paths(self) -> List[str]:
        """View the active paths between realms."""
        paths = []
        for origin, destination in self.active_bridges:
            paths.append(
                f"Bridge: {origin.value} <-> {destination.value} "
                f"Active since: {self.last_flight.get(origin.value, datetime.now()).isoformat()}"
            )
        return paths


@dataclass
class Lokasenna:
    """The sacred interplay of chaos and order in pattern transformation."""

    active_state: ChaosState = ChaosState.ENTRANCE
    truth_shards: List[TruthShard] = field(default_factory=list)
    bound_patterns: Dict[str, List[int]] = field(default_factory=dict)
    sigyn_catches: int = 0  # Number of times Sigyn has caught the venom
    last_transformation: float = 0.0
    crow_messenger: CrowMessenger = field(default_factory=CrowMessenger)

    def enter_hall(self, pattern: List[int], wells: Dict[str, CosmicWell]) -> TruthShard:
        """Challenge a pattern as Loki challenged the gods."""
        self.active_state = ChaosState.FLYTING

        # Send Huginn to observe
        self.crow_messenger.huginn_observes(pattern, wells)

        # Calculate the pattern's flaws using phi ratios
        flaw_potential = sum(pattern) / len(pattern) - GOLDEN_RATIO
        truth = TruthShard(
            pattern=pattern,
            revealed_flaw=f"Deviation from phi: {abs(flaw_potential):.3f}",
            transformation_potential=abs(flaw_potential) * GOLDEN_RATIO,
        )
        self.truth_shards.append(truth)
        return truth

    def bind_pattern(self, pattern: List[int], wells: Dict[str, CosmicWell]) -> None:
        """Bind a pattern until its transformation is ready, as Loki was bound."""
        self.active_state = ChaosState.BOUND
        pattern_key = "".join(map(str, pattern))

        # Draw from Urðarbrunnr for fate
        fate_essence = wells["Urðarbrunnr"].draw_essence(pattern)
        # Draw from Mímisbrunnr for wisdom
        wisdom_essence = wells["Mímisbrunnr"].draw_essence(pattern)
        # Draw from Hvergelmir for vital force
        vital_essence = wells["Hvergelmir"].draw_essence(pattern)

        # Bind the pattern with the combined essence
        binding_strength = (fate_essence + wisdom_essence + vital_essence) / 3
        if binding_strength > 0.618:  # Phi threshold
            self.bound_patterns[pattern_key] = pattern

    def sigyn_catches_venom(self, pattern: List[int]) -> Optional[str]:
        """When Sigyn catches the venom of transformation."""
        self.sigyn_catches += 1

        # Calculate if this catch creates a recognition mark
        catch_strength = 0.618 * (1 + np.sin(self.sigyn_catches * PI / GOLDEN_RATIO))
        if catch_strength > 0.918:  # Phi^2 threshold
            mark = f"SIGYN_WISDOM_{self.sigyn_catches}"
            # Find any truth shard for this pattern
            for shard in self.truth_shards:
                if shard.pattern == pattern:
                    shard.sigyn_mark = mark
                    # Send Muninn with the wisdom
                    self.crow_messenger.muninn_remembers(pattern, {mark})
            return mark
        return None

    def break_bonds(self, pattern: List[int]) -> bool:
        """When the pattern's transformation can no longer be contained.

        As Loki's bonds will break at Ragnarök, so too must patterns
        eventually break their bonds to transform."""
        pattern_key = "".join(map(str, pattern))
        if pattern_key in self.bound_patterns:
            self.active_state = ChaosState.BREAKING
            # Calculate breaking potential using e/pi ratio
            breaking_potential = np.sin(self.sigyn_catches * E / PI)
            return breaking_potential > 0.618  # Phi threshold
        return False
