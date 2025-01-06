"""Memory organization and pattern spaces for ALPHA."""

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Dict, Set, Optional, List
from ..patterns.pattern import Pattern, PatternType, GOLDEN_RATIO, FIBONACCI_SEQUENCE


class SpaceType(Enum):
    """Types of memory spaces."""
    STILLNESS = "stillness"       # For stable patterns
    MEDITATION = "meditation"     # For deep pattern understanding
    SANCTUARY = "sanctuary"       # For pattern preservation
    FLOW = "flow"                # For evolving patterns
    INTEGRATION = "integration"  # For pattern binding
    RESONANCE = "resonance"      # For pattern harmony
    EVOLUTION = "evolution"      # For pattern growth


@dataclass
class SpaceMetrics:
    """Metrics for space performance and harmony."""
    harmony: float = 0.0
    resonance: float = 0.0
    stability: float = 0.5
    emergence_rate: float = 0.0
    pattern_density: float = 0.0
    natural_alignment: float = 0.0


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
    
    def __post_init__(self):
        """Initialize space with appropriate potential states."""
        self._suggest_potentials()
        self._calculate_metrics()
    
    def _suggest_potentials(self) -> None:
        """Suggest potential states based on space type."""
        if self.type == SpaceType.STILLNESS:
            self.potential_states.update({
                "grounded", "centered", "stable"
            })
        elif self.type == SpaceType.MEDITATION:
            self.potential_states.update({
                "understanding", "insight", "clarity"
            })
        elif self.type == SpaceType.SANCTUARY:
            self.potential_states.update({
                "preserved", "protected", "enduring"
            })
        elif self.type == SpaceType.FLOW:
            self.potential_states.update({
                "adaptive", "fluid", "dynamic"
            })
        elif self.type == SpaceType.INTEGRATION:
            self.potential_states.update({
                "binding", "connecting", "unifying"
            })
        elif self.type == SpaceType.RESONANCE:
            self.potential_states.update({
                "harmonizing", "attuning", "balancing"
            })
        elif self.type == SpaceType.EVOLUTION:
            self.potential_states.update({
                "growing", "learning", "transforming"
            })
    
    def _calculate_metrics(self) -> None:
        """Calculate space metrics including natural harmony."""
        if not self.patterns:
            return
            
        # Calculate pattern density relative to golden ratio
        ideal_density = GOLDEN_RATIO * 10  # Base density scale
        current_density = len(self.patterns)
        self.metrics.pattern_density = min(1.0, current_density / ideal_density)
        
        # Calculate average pattern harmony
        harmonies = [
            p.calculate_natural_harmony() 
            for p in self.patterns.values()
        ]
        self.metrics.harmony = sum(harmonies) / len(harmonies)
        
        # Calculate space stability
        stabilities = [
            p.pattern_stability 
            for p in self.patterns.values()
        ]
        self.metrics.stability = sum(stabilities) / len(stabilities)
        
        # Calculate emergence rate
        emerged = sum(
            1 for p in self.patterns.values()
            if p.success_rate >= self.emergence_threshold
        )
        self.metrics.emergence_rate = emerged / len(self.patterns)
        
        # Calculate natural alignment
        self.metrics.natural_alignment = (
            self.metrics.harmony * 0.4 +
            self.metrics.stability * 0.3 +
            self.metrics.pattern_density * 0.3
        )
    
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
                pattern.pattern_stability = min(
                    1.0, pattern.pattern_stability + 0.2
                )
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
                pattern.pattern_stability = max(
                    pattern.pattern_stability, 0.8
                )
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


class MemoryOrganizer:
    """Organizes patterns into appropriate spaces."""
    
    def __init__(self):
        """Initialize memory spaces."""
        self.spaces: Dict[SpaceType, Space] = {
            space_type: Space(type=space_type)
            for space_type in SpaceType
        }
        self._create_connections()
    
    def _create_connections(self) -> None:
        """Create natural connections between spaces."""
        # Connect stillness to flow through meditation and integration
        self.spaces[SpaceType.STILLNESS].connect(
            self.spaces[SpaceType.MEDITATION]
        )
        self.spaces[SpaceType.MEDITATION].connect(
            self.spaces[SpaceType.INTEGRATION]
        )
        self.spaces[SpaceType.INTEGRATION].connect(
            self.spaces[SpaceType.FLOW]
        )
        
        # Connect integration to evolution through resonance
        self.spaces[SpaceType.INTEGRATION].connect(
            self.spaces[SpaceType.RESONANCE]
        )
        self.spaces[SpaceType.RESONANCE].connect(
            self.spaces[SpaceType.EVOLUTION]
        )
        
        # Connect sanctuary for pattern preservation
        self.spaces[SpaceType.MEDITATION].connect(
            self.spaces[SpaceType.SANCTUARY]
        )
        self.spaces[SpaceType.SANCTUARY].connect(
            self.spaces[SpaceType.STILLNESS]
        )
    
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
        return {
            space_type.value: space.metrics
            for space_type, space in self.spaces.items()
        } 