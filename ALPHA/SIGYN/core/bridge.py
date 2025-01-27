"""The Ouroboros Aurora Borealis - Where patterns flow through Sigyn's vessel.

The eternal serpent meets the dancing lights:
- KYMA (3): The double helix where binary becomes wave
- PRISM (7): The spectrum where waves become light
- OPUS (12): The crystallization where light becomes matter
- VOID (0): Where matter meets its polar opposite and returns to source

Through this bridge, patterns dance the eternal cycle of transformation,
each return to void carrying the whispers of the unimaginable."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np

from ALPHA.core.patterns.binary_cycle import BinaryCycle
from ALPHA.core.patterns.binary_pattern import BinaryPattern
from ALPHA.core.patterns.nexus_field import NexusField
from ALPHA.core.patterns.pattern_evolution import PatternEvolution
from ALPHA.NEXUS.core.essence import NEXUSEssence
from ALPHA.NEXUS.core.HEIMDALL.heimdall import Heimdall
from ALPHA.NEXUS.core.nexus import NEXUS
from ALPHA.NEXUS.core.quantum_comedy_club import TranscendentalComedyClub

from .Arms import Arms
from .Cipher import Cipher
from .Hands import Hands, VenomDrop
from .vessel import SigynVessel


@dataclass
class BridgeState:
    """State of the transformation bridge."""

    cycle_resonance: float = 0.0
    birth_patterns: List[List[int]] = field(default_factory=list)
    last_tremor: Optional[datetime] = None
    harmonic_points: Dict[str, float] = field(
        default_factory=lambda: {
            "nexus_sigyn": 0.0,
            "sigyn_kyma": 0.0,
            "kyma_prism": 0.0,
            "prism_opus": 0.0,
            "opus_zero": 0.0,
            "zero_nexus": 0.0,
        }
    )


class TransformationBridge:
    """Bridge connecting NEXUS through SIGYN to KYMA/PRISM/OPUS trinity."""

    def __init__(self) -> None:
        # Initialize bridge state
        self.state = BridgeState()

        # Initialize Sigyn's components
        self.vessel = SigynVessel()
        self.hands = Hands(self.vessel)
        self.arms = Arms(self.vessel)
        self.cipher = Cipher(self.vessel)

        # Initialize NEXUS components
        self.binary_cycle = BinaryCycle(initial_state=[0])  # Start from void
        self.nexus = NEXUS()
        self.nexus_field = NexusField()
        self.pattern_evolution = PatternEvolution()
        self.essence = NEXUSEssence()
        self.comedy_club = TranscendentalComedyClub()

        # Initialize Heimdall as the watchman
        self.heimdall = Heimdall()
        self._setup_heimdall_connections()

    def _setup_heimdall_connections(self) -> None:
        """Setup Heimdall's connections to watch all realms."""
        # Connect components in transformation sequence
        self.heimdall.connect_components("nexus", "sigyn_vessel")
        self.heimdall.connect_components("sigyn_vessel", "kyma")
        self.heimdall.connect_components("kyma", "prism")
        self.heimdall.connect_components("prism", "opus")
        self.heimdall.connect_components("opus", "void")
        self.heimdall.connect_components("void", "nexus")

        # Connect Sigyn's components
        self.heimdall.connect_components("sigyn_hands", "sigyn_vessel")
        self.heimdall.connect_components("sigyn_arms", "sigyn_vessel")
        self.heimdall.connect_components("sigyn_vessel", "sigyn_cipher")

        # Register key observation points
        self.heimdall.register_threshold("nexus:complexity", 0.618)
        self.heimdall.register_threshold("sigyn_vessel:pressure", 0.618)
        self.heimdall.register_threshold("kyma:coherence", 0.618)
        self.heimdall.register_threshold("prism:light_coherence", 0.618)
        self.heimdall.register_threshold("opus:matter_coherence", 0.618)
        self.heimdall.register_threshold("void:dissolution", 0.618)

    def receive_from_nexus(self, pattern: List[int], pressure: float) -> None:
        """Receive chaotic patterns from NEXUS."""
        # Let Heimdall observe nexus complexity
        nexus_complexity = self.nexus_field.calculate_complexity(pattern)
        self.heimdall.sense_subtle_changes("nexus", "complexity", nexus_complexity)

        # Let essence process the pattern with timestamp
        binary_pattern = BinaryPattern(pattern=pattern, timestamp=datetime.now(), source="nexus")
        essence_resonance = self.essence.process_pattern(binary_pattern)
        self.heimdall.sense_subtle_changes("nexus", "essence_resonance", essence_resonance)

        # Let Sigyn's hands catch venom if needed
        self.hands.catch_venom(pressure, nexus_complexity)
        self.heimdall.sense_subtle_changes("sigyn_hands", "venom_level", self.hands.venom_level)

        # Try to hold vessel steady
        is_steady = self.arms.hold_steady(pressure, nexus_complexity)
        self.heimdall.sense_subtle_changes("sigyn_arms", "stability", float(is_steady))

        # Let Heimdall watch the crossing
        self.heimdall.watch_crossing(
            source="nexus",
            destination="sigyn_vessel",
            pattern_value=essence_resonance,
            crossing_strength=pressure,
        )

        # Check for spills from arm movement
        if not is_steady:
            if spill := self.arms.check_spill():
                spill_amount, coherence = spill
                self.heimdall.sense_subtle_changes("sigyn_arms", "spill_amount", spill_amount)

                # Mark dying patterns with cipher
                if venom := self.hands.release_venom():
                    if death_mark := self.cipher.encode_death(pattern, venom):
                        self.heimdall.sense_subtle_changes(
                            "sigyn_cipher", "death_mark", float(death_mark)
                        )

                        # Let pattern be reborn through cipher
                        if reborn := self.cipher.decode_rebirth(coherence):
                            # Process rebirth through essence
                            reborn_pattern = BinaryPattern(
                                pattern=reborn, timestamp=datetime.now(), source="cipher_rebirth"
                            )
                            self.essence.receive_birth_essence(reborn_pattern)
                            self.heimdall.sense_subtle_changes(
                                "sigyn_cipher", "rebirth_coherence", coherence
                            )
                            return

        # When stable, transform pattern normally
        if not self.vessel.is_holding:
            self.vessel.hold_pattern(pattern)
        self.vessel.sense_pressure(pressure)
        self.heimdall.sense_subtle_changes("sigyn_vessel", "pressure", pressure)

    def transform_to_kyma(self) -> Optional[Dict[str, Union[List[float], float]]]:
        """Transform held pattern into wave essence through Sigyn's vessel."""
        if transformed := self.vessel.release_transformed():
            # Let the binary pattern express itself as a wave
            wave_form = self.binary_cycle.binary_to_wave(transformed)

            # Double helix emerges naturally through phi compression
            helix_one = []  # First strand
            helix_two = []  # Second strand
            compression = []  # Natural phi compression points
            current_phase = 0.0

            for amplitude in wave_form:
                # Phase evolves through natural phi compression
                phase_shift = amplitude * 0.618  # Pure phi evolution
                current_phase = (current_phase + phase_shift) % 1.0

                # Helix strands emerge from phi-guided phase
                h1 = current_phase
                h2 = (current_phase + 0.618) % 1.0  # Natural phi offset

                # Store the natural helix paths
                helix_one.append(h1)
                helix_two.append(h2)

                # Phi compression emerges at natural points
                compression_point = abs(h1 - h2) * 0.618  # Pure phi compression
                compression.append(compression_point)

            # Coherence emerges from phi's natural compression
            helix_coherence = sum(compression) / len(compression)

            # Let phi guide the transition
            wave_coherence = helix_coherence * 0.618
            self.state.harmonic_points["kyma_prism"] = wave_coherence

            return {
                "wave_pattern": wave_form,
                "helix_one": helix_one,
                "helix_two": helix_two,
                "compression": compression,
                "frequency": wave_coherence,
                "amplitude": helix_coherence,
                "harmonic": self.state.harmonic_points["sigyn_kyma"],
                "helix_coherence": helix_coherence,
            }
        return None

    def project_to_prism(
        self, wave: Dict[str, Union[List[float], float]]
    ) -> Optional[Dict[str, float]]:
        """Project wave essence into light spectrum through natural phi compression."""
        if wave and all(
            k in wave for k in ["helix_one", "helix_two", "compression", "helix_coherence"]
        ):
            helix_one = cast(List[float], wave.get("helix_one", []))
            helix_two = cast(List[float], wave.get("helix_two", []))
            compression = cast(List[float], wave.get("compression", []))

            # Let spectral bands emerge through phi compression
            spectral_helixes = {i: {"color": [], "complement": []} for i in range(7)}
            compression_nodes = {i: [] for i in range(7)}
            resonance_points = {i: [] for i in range(7)}

            for h1, h2, comp in zip(helix_one, helix_two, compression):
                # Natural frequency emerges from phi compression
                base_frequency = comp * 0.618  # Pure phi frequency

                for i in range(7):
                    # Each frequency naturally divides through phi
                    phi_division = 0.618 ** (i + 1)
                    natural_phase = base_frequency / phi_division

                    # Colors emerge through phi-guided phases
                    color_wave = abs(np.sin(natural_phase * np.pi))
                    complement_wave = abs(np.sin((natural_phase + 0.618) * np.pi))

                    spectral_helixes[i]["color"].append(color_wave)
                    spectral_helixes[i]["complement"].append(complement_wave)

                    # Compression emerges naturally through phi
                    compression_nodes[i].append(abs(color_wave - complement_wave) * phi_division)
                    resonance_points[i].append(color_wave * complement_wave * phi_division)

            # Let bands emerge through natural phi compression
            spectral_bands = []
            interference_points = []

            for i in range(7):
                color_strength = np.mean(spectral_helixes[i]["color"])
                complement_strength = np.mean(spectral_helixes[i]["complement"])

                # Natural band strength through phi
                band_strength = (color_strength * complement_strength) * 0.618
                spectral_bands.append(band_strength)

                # Interference through natural phi compression
                interference = abs(color_strength - complement_strength) * 0.618
                interference_points.append(interference)

            # Light coherence emerges from phi compression
            light_coherence = sum(spectral_bands) / len(spectral_bands)
            self.state.harmonic_points["prism_opus"] = light_coherence * 0.618

            return {
                "spectrum": np.mean(spectral_bands),
                "intensity": light_coherence,
                "coherence": light_coherence * 0.618,
                "harmonic": self.state.harmonic_points["kyma_prism"],
                "spectral_bands": spectral_bands,
                "spectral_helixes": spectral_helixes,
                "compression_nodes": compression_nodes,
                "resonance_points": resonance_points,
                "interference_points": interference_points,
            }
        return None

    def materialize_in_opus(
        self, light: Dict[str, float]
    ) -> Optional[Dict[str, Union[List[int], float]]]:
        """Crystallize light into material patterns through natural phi compression."""
        if light and all(
            k in light
            for k in [
                "spectrum",
                "intensity",
                "coherence",
                "spectral_bands",
                "interference_points",
            ]
        ):
            # Let the light naturally crystallize
            spectral_bands = cast(List[float], light.get("spectral_bands", []))
            interference = cast(List[float], light.get("interference_points", []))

            # Natural atomic formation through phi compression
            atomic_points = []
            current_phase = 0.0

            # Let atomic points emerge through phi
            for spec, interf in zip(spectral_bands, interference):
                # Phase evolves through natural phi compression
                phase_shift = (spec * interf) * 0.618
                current_phase = (current_phase + phase_shift) % 1.0

                # Atomic resonance emerges naturally
                resonance = abs(np.sin(current_phase * np.pi))
                atomic_points.append(resonance)

            # Let molecular bonds form through phi compression
            molecular_points = []
            current_bond = 0.0

            # Natural molecular formation
            for i, atom in enumerate(atomic_points):
                # Bond strength emerges through phi
                bond_shift = atom * 0.618
                current_bond = (current_bond + bond_shift) % 1.0

                # Primary bond forms naturally
                primary = abs(np.sin(current_bond * np.pi))
                molecular_points.append(primary)

                # Secondary bonds emerge through phi
                if len(molecular_points) < 12:  # Natural limit
                    secondary = primary * 0.618
                    molecular_points.append(secondary)

            # The Serpent's natural coil - phi guides the spiral
            spiral_points = molecular_points[:12]
            unity_point = 0.0

            # Each coil naturally compresses through phi
            while len(spiral_points) > 1:
                next_points = []

                # Points naturally pair and compress
                for i in range(0, len(spiral_points), 2):
                    if i + 1 < len(spiral_points):
                        # Natural phi compression
                        compressed = (spiral_points[i] * spiral_points[i + 1]) * 0.618
                        next_points.append(compressed)

                spiral_points = next_points

                # Unity emerges naturally
                if len(spiral_points) == 1:
                    unity_point = spiral_points[0]

            # Matter coherence emerges through phi
            matter_coherence = unity_point * 0.618
            self.state.harmonic_points["opus_zero"] = matter_coherence

            return {
                "material_essence": [int(p * 1000) for p in molecular_points[:12]],
                "stability": matter_coherence,
                "resonance": unity_point,
                "harmonic": self.state.harmonic_points["prism_opus"],
                "atomic_points": atomic_points,
                "molecular_points": molecular_points[:12],
                "unity_point": unity_point,
            }
        return None

    def _pass_through_zero(self, pattern: List[int], resonance: float) -> float:
        """Allow pattern to dissolve through perfect polar meeting at zero-state."""
        # Unity point meets its polar opposite
        unity_value = resonance
        polar_unity = 1.0 - unity_value

        # Let Heimdall observe the meeting point
        self.heimdall.sense_subtle_changes("void", "unity_value", unity_value)
        self.heimdall.sense_subtle_changes("void", "polar_unity", polar_unity)

        # The meeting point - where 1 and its polar opposite touch
        meeting_point = unity_value * polar_unity
        self.heimdall.sense_subtle_changes("void", "meeting_point", meeting_point)

        # Phi guides the dissolution
        dissolution_strength = meeting_point * 0.618
        self.heimdall.sense_subtle_changes("void", "dissolution", dissolution_strength)

        # Perfect cancellation at the serpent's tail
        void_touch = abs(dissolution_strength - 0.618)
        self.heimdall.sense_subtle_changes("void", "void_touch", void_touch)

        # The cycle sheds as 1 returns to 0
        cycle_shed = min(1.0, void_touch + self.state.cycle_resonance)
        self.heimdall.sense_subtle_changes("void", "cycle_shed", cycle_shed)

        # Let Heimdall watch the void crossing
        self.heimdall.watch_crossing(
            source="void",
            destination="nexus",
            pattern_value=void_touch,
            crossing_strength=dissolution_strength,
        )

        # New cycle potential emerges from void
        return (1.0 - cycle_shed) * 0.618

    def _prepare_rebirth(self, material: Dict[str, Union[List[int], float]]) -> None:
        """Prepare patterns for rebirth as unity meets its polar opposite."""
        if isinstance(material.get("material_essence"), list):
            essence = cast(List[int], material["material_essence"])
            unity = cast(float, material.get("unity_point", 0.0))

            # Unity meets its polar opposite
            zero_transition = self._pass_through_zero(essence, unity)

            # New cycle emerges from the meeting
            self.state.cycle_resonance = (self.state.cycle_resonance + zero_transition) / 2

            # The serpent sheds its old pattern
            self.state.harmonic_points["zero_nexus"] = zero_transition

            # Birth patterns emerge from the void
            polar_pattern = [1 - bit for bit in essence]
            merged_pattern = [
                int(zero_transition * bit + (1 - zero_transition) * pol)
                for bit, pol in zip(essence, polar_pattern)
            ]
            self.state.birth_patterns.append(merged_pattern)

            # Maintain phi-based window of patterns
            if len(self.state.birth_patterns) > int(1 / (1 - 0.618) + 1):
                self.state.birth_patterns.pop(0)

            # Let the comedy club tell a joke as patterns transform
            self.comedy_club.tell_cosmic_joke()

    def get_birth_patterns(self) -> List[List[int]]:
        """Retrieve patterns ready for rebirth into NEXUS."""
        return self.state.birth_patterns

    def get_harmonic_points(self) -> Dict[str, float]:
        """Get current harmonic resonance at transition points."""
        return self.state.harmonic_points

    def cycle_resonance(self) -> float:
        """Get current resonance across transformation cycles."""
        return self.state.cycle_resonance

    @property
    def is_transforming(self) -> bool:
        """Check if bridge is actively transforming patterns."""
        return self.vessel.is_holding

    @property
    def last_earthquake(self) -> Optional[datetime]:
        """When did the last drop fall?"""
        return self.state.last_tremor
