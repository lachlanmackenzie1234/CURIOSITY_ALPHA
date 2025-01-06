"""Pattern resonance system for enhanced pattern detection and preservation.

This module provides mechanisms for calculating and tracking pattern resonance,
which measures how strongly patterns align with natural mathematical principles
and how they interact with each other.
"""

from dataclasses import dataclass, field
from typing import Dict, Set
import numpy as np
from enum import Enum

from .natural_patterns import (
    NaturalPattern,
    NaturalPatternHierarchy
)


class ResonanceType(Enum):
    """Types of pattern resonance."""
    HARMONIC = "harmonic"      # Patterns that reinforce each other
    DESTRUCTIVE = "destructive"  # Patterns that interfere negatively
    NEUTRAL = "neutral"        # Patterns with minimal interaction
    EMERGENT = "emergent"      # Patterns that create new patterns


@dataclass
class ResonanceProfile:
    """Profile of pattern resonance characteristics."""
    
    resonance_type: ResonanceType
    strength: float = 0.0
    stability: float = 0.0
    harmony: float = 0.0
    influence_radius: int = 0
    affected_patterns: Set[str] = field(default_factory=set)


class PatternResonance:
    """Manages pattern resonance detection and analysis."""
    
    def __init__(self):
        """Initialize the pattern resonance system."""
        self.hierarchy = NaturalPatternHierarchy()
        self.resonance_profiles: Dict[str, ResonanceProfile] = {}
        self.interaction_matrix: Dict[str, Dict[str, float]] = {}
        
        # Resonance thresholds
        self.harmonic_threshold = 0.8
        self.destructive_threshold = 0.3
        self.influence_radius_max = 128
    
    def calculate_resonance(
        self,
        pattern: NaturalPattern,
        context_data: np.ndarray
    ) -> ResonanceProfile:
        """Calculate resonance profile for a pattern in context."""
        try:
            # Calculate base resonance metrics
            strength = self._calculate_resonance_strength(
                pattern, context_data
            )
            stability = self._calculate_resonance_stability(
                pattern, context_data
            )
            harmony = self._calculate_resonance_harmony(
                pattern, context_data
            )
            
            # Determine resonance type
            if harmony > self.harmonic_threshold:
                res_type = ResonanceType.HARMONIC
            elif harmony < self.destructive_threshold:
                res_type = ResonanceType.DESTRUCTIVE
            else:
                res_type = ResonanceType.NEUTRAL
            
            # Calculate influence radius
            radius = self._calculate_influence_radius(
                strength, stability, harmony
            )
            
            return ResonanceProfile(
                resonance_type=res_type,
                strength=strength,
                stability=stability,
                harmony=harmony,
                influence_radius=radius
            )
            
        except Exception as e:
            print(f"Error calculating resonance: {str(e)}")
            return ResonanceProfile(resonance_type=ResonanceType.NEUTRAL)
    
    def analyze_pattern_interactions(
        self,
        patterns: Dict[str, NaturalPattern],
        data: np.ndarray
    ) -> Dict[str, ResonanceProfile]:
        """Analyze interactions between patterns in data."""
        profiles = {}
        try:
            # Calculate individual resonance profiles
            for pattern_id, pattern in patterns.items():
                profile = self.calculate_resonance(pattern, data)
                profiles[pattern_id] = profile
            
            # Analyze pattern interactions
            self.interaction_matrix.clear()
            pattern_ids = list(patterns.keys())
            
            for i, id1 in enumerate(pattern_ids):
                self.interaction_matrix[id1] = {}
                for id2 in pattern_ids[i + 1:]:
                    interaction = self._calculate_pattern_interaction(
                        patterns[id1], patterns[id2]
                    )
                    self.interaction_matrix[id1][id2] = interaction
                    
                    # Update affected patterns
                    if interaction > self.harmonic_threshold:
                        profiles[id1].affected_patterns.add(id2)
                        profiles[id2].affected_patterns.add(id1)
            
            return profiles
            
        except Exception as e:
            print(f"Error analyzing pattern interactions: {str(e)}")
            return profiles
    
    def _calculate_resonance_strength(
        self,
        pattern: NaturalPattern,
        data: np.ndarray
    ) -> float:
        """Calculate the strength of pattern resonance."""
        try:
            # Base strength on pattern confidence
            base_strength = pattern.confidence
            
            # Analyze data structure
            if len(data) >= 4:
                # Calculate structural metrics
                regularity = self._calculate_regularity(data)
                consistency = self._calculate_consistency(data)
                
                # Combine metrics
                strength = (
                    base_strength * 0.4 +
                    regularity * 0.3 +
                    consistency * 0.3
                )
                
                return float(np.clip(strength, 0.0, 1.0))
            
            return base_strength
            
        except Exception:
            return 0.0
    
    def _calculate_resonance_stability(
        self,
        pattern: NaturalPattern,
        data: np.ndarray
    ) -> float:
        """Calculate the stability of pattern resonance."""
        try:
            if len(data) < 4:
                return 0.0
            
            # Calculate stability metrics
            entropy = self.hierarchy._calculate_entropy(data)
            symmetry = self.hierarchy._calculate_symmetry(data)
            
            # More stable patterns have:
            # - Lower entropy (more predictable)
            # - Higher symmetry
            stability = (
                (1 - entropy) * 0.5 +  # Lower entropy is better
                symmetry * 0.5
            )
            
            return float(np.clip(stability, 0.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _calculate_resonance_harmony(
        self,
        pattern: NaturalPattern,
        data: np.ndarray
    ) -> float:
        """Calculate how harmoniously the pattern resonates."""
        try:
            if len(data) < 4:
                return 0.0
            
            # Check alignment with natural principles
            principle_alignment = self.hierarchy._calculate_principle_confidence(
                data, pattern.principle_type
            )
            
            # Check for related patterns
            related_alignment = 0.0
            if pattern.principle_type in self.hierarchy.relationships:
                related_types = self.hierarchy.relationships[
                    pattern.principle_type
                ]
                alignments = []
                for related_type in related_types:
                    alignment = self.hierarchy._calculate_principle_confidence(
                        data, related_type
                    )
                    alignments.append(alignment)
                if alignments:
                    related_alignment = np.mean(alignments)
            
            # Combine alignments
            harmony = (
                principle_alignment * 0.7 +
                related_alignment * 0.3
            )
            
            return float(np.clip(harmony, 0.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _calculate_influence_radius(
        self,
        strength: float,
        stability: float,
        harmony: float
    ) -> int:
        """Calculate the radius of pattern influence."""
        try:
            # Base radius on combined metrics
            base_radius = int(
                self.influence_radius_max * (
                    strength * 0.4 +
                    stability * 0.3 +
                    harmony * 0.3
                )
            )
            
            # Ensure minimum radius of 4 bytes
            return max(4, min(base_radius, self.influence_radius_max))
            
        except Exception:
            return 4
    
    def _calculate_pattern_interaction(
        self,
        pattern1: NaturalPattern,
        pattern2: NaturalPattern
    ) -> float:
        """Calculate interaction strength between two patterns."""
        try:
            # Check if patterns are related
            if (pattern1.principle_type in self.hierarchy.relationships and
                    pattern2.principle_type in
                    self.hierarchy.relationships[pattern1.principle_type]):
                base_interaction = 0.8
            else:
                base_interaction = 0.4
            
            # Adjust based on confidence
            confidence_factor = np.sqrt(
                pattern1.confidence * pattern2.confidence
            )
            
            # Adjust based on resonance
            resonance_factor = np.sqrt(
                pattern1.resonance * pattern2.resonance
            )
            
            interaction = (
                base_interaction * 0.4 +
                confidence_factor * 0.3 +
                resonance_factor * 0.3
            )
            
            return float(np.clip(interaction, 0.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _calculate_regularity(self, data: np.ndarray) -> float:
        """Calculate pattern regularity in data."""
        try:
            if len(data) < 4:
                return 0.0
            
            # Look for repeating subsequences
            max_length = min(len(data) // 2, 16)
            regularities = []
            
            for length in range(2, max_length + 1):
                chunks = [
                    tuple(data[i:i + length])
                    for i in range(0, len(data) - length + 1)
                ]
                unique_ratio = len(set(chunks)) / len(chunks)
                regularities.append(1 - unique_ratio)
            
            return float(np.mean(regularities)) if regularities else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_consistency(self, data: np.ndarray) -> float:
        """Calculate pattern consistency in data."""
        try:
            if len(data) < 4:
                return 0.0
            
            # Calculate first and second order differences
            diff1 = np.diff(data)
            diff2 = np.diff(diff1)
            
            # More consistent patterns have smaller variations
            consistency1 = 1.0 / (1.0 + np.std(diff1))
            consistency2 = 1.0 / (1.0 + np.std(diff2))
            
            # Combine metrics
            consistency = (
                consistency1 * 0.6 +
                consistency2 * 0.4
            )
            
            return float(np.clip(consistency, 0.0, 1.0))
            
        except Exception:
            return 0.0 