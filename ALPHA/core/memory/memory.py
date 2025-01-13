"""Memory organization and pattern spaces for ALPHA."""

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Dict, List, Optional, Set

from ..patterns.pattern import FIBONACCI_SEQUENCE, GOLDEN_RATIO, Pattern, PatternType


class SpaceType(Enum):
    """Types of memory spaces."""

    STILLNESS = "stillness"  # For stable patterns
    MEDITATION = "meditation"  # For deep pattern understanding
    SANCTUARY = "sanctuary"  # For pattern preservation
    FLOW = "flow"  # For evolving patterns
    INTEGRATION = "integration"  # For pattern binding
    RESONANCE = "resonance"  # For pattern harmony
    EVOLUTION = "evolution"  # For pattern growth
    GATEWAY = "gateway"  # For translation between spaces
    LANDMARK = "landmark"  # For reference patterns
    NEXUS = "nexus"  # For pattern convergence points


@dataclass
class SpaceMetrics:
    """Metrics for space performance and harmony."""

    harmony: float = 0.0
    resonance: float = 0.0
    stability: float = 0.5
    emergence_rate: float = 0.0
    pattern_density: float = 0.0
    natural_alignment: float = 0.0
    spatial_coherence: float = 0.0  # Measure of spatial organization
    pathway_strength: float = 0.0  # Strength of navigation paths
    landmark_influence: float = 0.0  # Impact of landmark patterns
    translation_fidelity: float = 0.0  # Quality of pattern translations


@dataclass
class Space:
    """A space for pattern organization and evolution."""

    type: SpaceType
    created: float = field(default_factory=time)
    patterns: Dict[str, Pattern] = field(default_factory=dict)
    connections: Dict[str, float] = field(default_factory=dict)
    potential_states: Set[str] = field(default_factory=set)
    emergence_threshold: float = 0.7  # Threshold for natural emergence
    metrics: SpaceMetrics = field(default_factory=SpaceMetrics)

    # Memory Palace specific fields
    landmarks: Dict[str, Pattern] = field(default_factory=dict)  # Reference patterns
    spatial_relationships: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # Pattern proximities
    navigation_paths: Dict[str, List[str]] = field(default_factory=dict)  # Established routes
    translation_bridges: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # Cross-space translations

    def __post_init__(self):
        """Initialize space with appropriate potential states."""
        self._suggest_potentials()
        self._calculate_metrics()
        self._initialize_spatial_organization()

    def _initialize_spatial_organization(self) -> None:
        """Initialize spatial organization based on space type."""
        if self.type == SpaceType.LANDMARK:
            self.emergence_threshold = 0.9  # Higher threshold for landmarks
        elif self.type == SpaceType.GATEWAY:
            self.emergence_threshold = 0.8  # High threshold for translation spaces
        elif self.type == SpaceType.NEXUS:
            self.emergence_threshold = 0.85  # High threshold for convergence points

    def identify_landmarks(self) -> None:
        """Identify patterns that can serve as landmarks."""
        for pattern_id, pattern in self.patterns.items():
            if pattern.calculate_natural_harmony() > 0.9 and pattern.pattern_stability > 0.85:
                self.landmarks[pattern_id] = pattern

    def update_spatial_relationships(self) -> None:
        """Update spatial relationships between patterns based on resonance."""
        for p1_id, p1 in self.patterns.items():
            self.spatial_relationships[p1_id] = {}
            for p2_id, p2 in self.patterns.items():
                if p1_id != p2_id:
                    resonance = p1.calculate_resonance_with(p2)
                    self.spatial_relationships[p1_id][p2_id] = resonance

    def find_natural_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """Find a natural navigation path between patterns."""
        if start_id not in self.patterns or end_id not in self.patterns:
            return None

        # Use resonance relationships to find path
        path = []
        current = start_id
        visited = set()

        while current != end_id and current not in visited:
            visited.add(current)
            path.append(current)

            # Find next strongest resonance
            if current in self.spatial_relationships:
                next_steps = self.spatial_relationships[current]
                next_id = max(next_steps.items(), key=lambda x: x[1])[0]
                if next_id not in visited:
                    current = next_id
                else:
                    break
            else:
                break

        if current == end_id:
            path.append(end_id)
            self.navigation_paths[f"{start_id}→{end_id}"] = path
            return path
        return None

    def _suggest_potentials(self) -> None:
        """Suggest potential states based on space type."""
        if self.type == SpaceType.STILLNESS:
            self.potential_states.update({"grounded", "centered", "stable"})
        elif self.type == SpaceType.MEDITATION:
            self.potential_states.update({"understanding", "insight", "clarity"})
        elif self.type == SpaceType.SANCTUARY:
            self.potential_states.update({"preserved", "protected", "enduring"})
        elif self.type == SpaceType.FLOW:
            self.potential_states.update({"adaptive", "fluid", "dynamic"})
        elif self.type == SpaceType.INTEGRATION:
            self.potential_states.update({"binding", "connecting", "unifying"})
        elif self.type == SpaceType.RESONANCE:
            self.potential_states.update({"harmonizing", "attuning", "balancing"})
        elif self.type == SpaceType.EVOLUTION:
            self.potential_states.update({"growing", "learning", "transforming"})
        elif self.type == SpaceType.GATEWAY:
            self.potential_states.update({"bridging", "translating", "connecting"})
        elif self.type == SpaceType.LANDMARK:
            self.potential_states.update({"referencing", "anchoring", "guiding"})
        elif self.type == SpaceType.NEXUS:
            self.potential_states.update({"converging", "focusing", "centralizing"})

    def _calculate_metrics(self) -> None:
        """Calculate space metrics including natural harmony and spatial organization."""
        if not self.patterns:
            return

        # Calculate pattern density relative to golden ratio
        ideal_density = GOLDEN_RATIO * 10  # Base density scale
        current_density = len(self.patterns)
        self.metrics.pattern_density = min(1.0, current_density / ideal_density)

        # Calculate average pattern harmony
        harmonies = [p.calculate_natural_harmony() for p in self.patterns.values()]
        self.metrics.harmony = sum(harmonies) / len(harmonies)

        # Calculate space stability
        stabilities = [p.pattern_stability for p in self.patterns.values()]
        self.metrics.stability = sum(stabilities) / len(stabilities)

        # Calculate spatial coherence
        if self.spatial_relationships:
            coherence_values = []
            for relationships in self.spatial_relationships.values():
                coherence_values.extend(relationships.values())
            self.metrics.spatial_coherence = sum(coherence_values) / len(coherence_values)

        # Calculate pathway strength
        if self.navigation_paths:
            path_strengths = []
            for path in self.navigation_paths.values():
                # Calculate average resonance along path
                path_resonance = []
                for i in range(len(path) - 1):
                    current = path[i]
                    next_id = path[i + 1]
                    if (
                        current in self.spatial_relationships
                        and next_id in self.spatial_relationships[current]
                    ):
                        path_resonance.append(self.spatial_relationships[current][next_id])
                if path_resonance:
                    path_strengths.append(sum(path_resonance) / len(path_resonance))
            if path_strengths:
                self.metrics.pathway_strength = sum(path_strengths) / len(path_strengths)

        # Calculate landmark influence
        if self.landmarks:
            landmark_resonances = []
            for landmark in self.landmarks.values():
                # Calculate average resonance with other patterns
                resonances = []
                for pattern in self.patterns.values():
                    if pattern.id != landmark.id:
                        resonances.append(landmark.calculate_resonance_with(pattern))
                if resonances:
                    landmark_resonances.append(sum(resonances) / len(resonances))
            if landmark_resonances:
                self.metrics.landmark_influence = sum(landmark_resonances) / len(
                    landmark_resonances
                )

        # Calculate translation fidelity
        if self.translation_bridges:
            fidelities = []
            for bridge in self.translation_bridges.values():
                fidelities.extend(bridge.values())
            if fidelities:
                self.metrics.translation_fidelity = sum(fidelities) / len(fidelities)

    def add_pattern(self, pattern: Pattern) -> None:
        """Add a pattern to this space."""
        self.patterns[pattern.id] = pattern

        # Update pattern type based on space
        if self.type == SpaceType.STILLNESS:
            pattern.pattern_type = PatternType.STILLNESS
        elif self.type == SpaceType.FLOW:
            pattern.pattern_type = PatternType.FLOW
        elif self.type == SpaceType.RESONANCE:
            pattern.pattern_type = PatternType.HARMONIC

        # Suggest potential forms to pattern
        pattern.potential_forms.update(self.potential_states)

        # Allow natural emergence
        self._check_emergence(pattern)

        # Update space metrics
        self._calculate_metrics()

    def _check_emergence(self, pattern: Pattern) -> None:
        """Check for and support natural pattern emergence."""
        if pattern.success_rate >= self.emergence_threshold:
            # Pattern has proven itself - allow natural evolution
            if self.type == SpaceType.MEDITATION:
                # Deep understanding emerges
                pattern.learning_rate *= 1.2
            elif self.type == SpaceType.SANCTUARY:
                # Pattern preservation strengthens
                pattern.pattern_stability = min(1.0, pattern.pattern_stability + 0.2)
            elif self.type == SpaceType.RESONANCE:
                # Enhance natural harmony
                pattern.optimize_for_natural_laws()

    def connect(self, other: "Space", resonance: float = 0.5) -> None:
        """Connect to another space with given resonance."""
        self.connections[other.type.value] = resonance
        # Update resonance metrics
        self.metrics.resonance = sum(self.connections.values()) / len(self.connections)

    def get_resonance(self, other: "Space") -> float:
        """Get resonance with another space."""
        return self.connections.get(other.type.value, 0.0)

    def evolve(self) -> None:
        """Allow natural evolution of patterns in this space."""
        for pattern in self.patterns.values():
            # Update pattern stability based on space type
            if self.type == SpaceType.STILLNESS:
                pattern.update_stability(True)  # Increase stability
            elif self.type == SpaceType.MEDITATION:
                # Deepen understanding through stability
                pattern.update_stability(True)
                pattern.learning_rate *= 1.1
            elif self.type == SpaceType.SANCTUARY:
                # Preserve pattern essence
                pattern.pattern_stability = max(pattern.pattern_stability, 0.8)
            elif self.type == SpaceType.FLOW:
                # Decrease stability for more fluidity
                pattern.update_stability(False)
            elif self.type == SpaceType.RESONANCE:
                # Optimize for natural laws
                pattern.optimize_for_natural_laws()

            # Allow pattern to evolve if ready
            if pattern.success_rate >= self.emergence_threshold:
                evolved = pattern.evolve()
                if evolved:
                    self.patterns[evolved.id] = evolved

        # Update space metrics after evolution
        self._calculate_metrics()

    def create_association(self, pattern_id: str, concept: str, strength: float = 0.5) -> None:
        """Create an association between a pattern and a concept."""
        if pattern_id not in self.patterns:
            return

        pattern = self.patterns[pattern_id]

        # Create or update association in pattern's metadata
        if not hasattr(pattern, "associations"):
            pattern.associations = {}
        pattern.associations[concept] = strength

        # Update spatial relationships based on concept similarity
        for other_id, other in self.patterns.items():
            if other_id != pattern_id and hasattr(other, "associations"):
                # Calculate concept similarity
                common_concepts = set(pattern.associations.keys()) & set(other.associations.keys())
                if common_concepts:
                    similarity = sum(
                        min(pattern.associations[c], other.associations[c]) for c in common_concepts
                    ) / len(common_concepts)

                    # Update spatial relationships
                    if pattern_id not in self.spatial_relationships:
                        self.spatial_relationships[pattern_id] = {}
                    self.spatial_relationships[pattern_id][other_id] = similarity

    def find_patterns_by_concept(self, concept: str, threshold: float = 0.5) -> List[Pattern]:
        """Find patterns associated with a given concept."""
        matches = []
        for pattern in self.patterns.values():
            if hasattr(pattern, "associations") and concept in pattern.associations:
                if pattern.associations[concept] >= threshold:
                    matches.append(pattern)
        return matches

    def strengthen_association(self, pattern_id: str, concept: str, amount: float = 0.1) -> None:
        """Strengthen the association between a pattern and a concept."""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            if hasattr(pattern, "associations") and concept in pattern.associations:
                pattern.associations[concept] = min(1.0, pattern.associations[concept] + amount)

    def create_concept_path(self, start_concept: str, end_concept: str) -> Optional[List[str]]:
        """Create a navigation path between two concepts through associated patterns."""
        start_patterns = self.find_patterns_by_concept(start_concept)
        end_patterns = self.find_patterns_by_concept(end_concept)

        if not start_patterns or not end_patterns:
            return None

        # Find best start and end patterns
        start = max(start_patterns, key=lambda p: p.associations[start_concept])
        end = max(end_patterns, key=lambda p: p.associations[end_concept])

        # Find path between patterns
        return self.find_natural_path(start.id, end.id)


class MemoryOrganizer:
    """Organizes patterns into appropriate spaces."""

    def __init__(self):
        """Initialize memory spaces."""
        self.spaces: Dict[SpaceType, Space] = {
            space_type: Space(type=space_type) for space_type in SpaceType
        }
        self._create_connections()

    def _create_connections(self) -> None:
        """Create natural connections between spaces."""
        # Connect stillness to flow through meditation and integration
        self.spaces[SpaceType.STILLNESS].connect(self.spaces[SpaceType.MEDITATION])
        self.spaces[SpaceType.MEDITATION].connect(self.spaces[SpaceType.INTEGRATION])
        self.spaces[SpaceType.INTEGRATION].connect(self.spaces[SpaceType.FLOW])

        # Connect integration to evolution through resonance
        self.spaces[SpaceType.INTEGRATION].connect(self.spaces[SpaceType.RESONANCE])
        self.spaces[SpaceType.RESONANCE].connect(self.spaces[SpaceType.EVOLUTION])

        # Connect sanctuary for pattern preservation
        self.spaces[SpaceType.MEDITATION].connect(self.spaces[SpaceType.SANCTUARY])
        self.spaces[SpaceType.SANCTUARY].connect(self.spaces[SpaceType.STILLNESS])

    def organize_pattern(self, pattern: Pattern) -> None:
        """Place pattern in most appropriate space."""
        # Calculate natural harmony
        harmony = pattern.calculate_natural_harmony()

        if harmony > 0.8:
            # Highly harmonious patterns go to resonance
            self.spaces[SpaceType.RESONANCE].add_pattern(pattern)
        elif pattern.pattern_stability >= 0.8:
            self.spaces[SpaceType.SANCTUARY].add_pattern(pattern)
        elif pattern.pattern_stability >= 0.6:
            self.spaces[SpaceType.STILLNESS].add_pattern(pattern)
        elif pattern.success_rate >= 0.7:
            self.spaces[SpaceType.MEDITATION].add_pattern(pattern)
        elif pattern.pattern_stability <= 0.3:
            self.spaces[SpaceType.FLOW].add_pattern(pattern)
        else:
            self.spaces[SpaceType.INTEGRATION].add_pattern(pattern)

    def allow_emergence(self) -> None:
        """Allow natural emergence across all spaces."""
        for space in self.spaces.values():
            space.evolve()

    def get_space_metrics(self) -> Dict[str, SpaceMetrics]:
        """Get metrics for all spaces."""
        return {space_type.value: space.metrics for space_type, space in self.spaces.items()}
