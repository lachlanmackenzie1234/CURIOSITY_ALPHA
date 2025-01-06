"""Pattern-first translation system.

This module implements a translation approach that identifies and preserves
patterns before performing any translation, ensuring maximum pattern integrity.
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field

from ..patterns.natural_patterns import (
    NaturalPattern,
    NaturalPatternHierarchy
)
from ..patterns.resonance import PatternResonance
from ..patterns.binary_mapping import (
    PatternMapper,
    BinaryMapping
)


@dataclass
class TranslationUnit:
    """Represents a unit of translation with pattern awareness."""
    
    content: bytes
    patterns: Dict[str, NaturalPattern]
    mappings: Dict[str, BinaryMapping]
    resonance_score: float = 0.0
    preservation_score: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


class PatternFirstTranslator:
    """Implements pattern-first translation strategy."""
    
    def __init__(self):
        """Initialize the pattern-first translator."""
        self.hierarchy = NaturalPatternHierarchy()
        self.resonance = PatternResonance()
        self.mapper = PatternMapper()
        
        # Translation settings
        self.min_pattern_confidence = 0.7
        self.min_resonance_score = 0.6
        self.preservation_threshold = 0.8
        
        # Window settings
        self.pattern_window = 64
        self.overlap_size = 32
        self.context_size = 16
    
    def translate_to_binary(
        self,
        content: str,
        preserve_patterns: bool = True
    ) -> bytes:
        """Translate content to binary with pattern preservation."""
        try:
            # Convert content to numerical form
            data = np.array([ord(c) for c in content], dtype=np.uint8)
            
            # First pass: Identify patterns
            units = self._identify_patterns(data)
            
            # Second pass: Create mappings
            units = self._create_mappings(units)
            
            # Third pass: Optimize pattern preservation
            if preserve_patterns:
                units = self._optimize_preservation(units)
            
            # Final pass: Generate binary
            return self._generate_binary(units)
            
        except Exception as e:
            print(f"Error in pattern-first translation: {str(e)}")
            return content.encode()
    
    def translate_from_binary(
        self,
        binary: bytes,
        preserve_patterns: bool = True
    ) -> Optional[str]:
        """Translate binary back to content with pattern preservation."""
        try:
            # First pass: Extract translation units
            units = self._extract_units(binary)
            
            # Second pass: Recover patterns
            units = self._recover_patterns(units)
            
            # Third pass: Optimize pattern preservation
            if preserve_patterns:
                units = self._optimize_recovery(units)
            
            # Final pass: Generate content
            return self._generate_content(units)
            
        except Exception as e:
            print(f"Error in pattern-first translation: {str(e)}")
            return None
    
    def _identify_patterns(
        self,
        data: np.ndarray
    ) -> List[TranslationUnit]:
        """Identify patterns in content using sliding windows."""
        units = []
        try:
            # Use sliding window with overlap
            for i in range(0, len(data), self.pattern_window - self.overlap_size):
                # Get window of data
                window_end = min(i + self.pattern_window, len(data))
                window = data[i:window_end]
                
                # Get context around window
                context_start = max(0, i - self.context_size)
                context_end = min(len(data), window_end + self.context_size)
                context = data[context_start:context_end]
                
                # Detect patterns in window
                patterns: Dict[str, NaturalPattern] = {}
                pattern = self.hierarchy.detect_natural_pattern(
                    window
                )
                if (pattern and
                        pattern.confidence > self.min_pattern_confidence):
                    pattern_id = f"pattern_{len(patterns)}"
                    patterns[pattern_id] = pattern
                
                # Calculate resonance
                resonance_score = 0.0
                if patterns:
                    profiles = self.resonance.analyze_pattern_interactions(
                        patterns, context
                    )
                    resonance_score = max(
                        p.harmony for p in profiles.values()
                    )
                
                # Create translation unit
                unit = TranslationUnit(
                    content=window.tobytes(),
                    patterns=patterns,
                    mappings={},
                    resonance_score=resonance_score
                )
                units.append(unit)
            
            return units
            
        except Exception as e:
            print(f"Error identifying patterns: {str(e)}")
            return units
    
    def _create_mappings(
        self,
        units: List[TranslationUnit]
    ) -> List[TranslationUnit]:
        """Create binary mappings for patterns in translation units."""
        try:
            for unit in units:
                # Create mappings for each pattern
                for pattern_id, pattern in unit.patterns.items():
                    # Get context data
                    context = np.frombuffer(unit.content, dtype=np.uint8)
                    
                    # Create mapping
                    mapping = self.mapper.map_to_binary(pattern, context)
                    unit.mappings[pattern_id] = mapping
                    
                    # Update preservation score
                    unit.preservation_score = max(
                        unit.preservation_score,
                        mapping.mapping_confidence
                    )
            
            return units
            
        except Exception as e:
            print(f"Error creating mappings: {str(e)}")
            return units
    
    def _optimize_preservation(
        self,
        units: List[TranslationUnit]
    ) -> List[TranslationUnit]:
        """Optimize pattern preservation across translation units."""
        try:
            optimized = []
            i = 0
            while i < len(units):
                unit = units[i]
                
                # Check if unit has strong patterns
                if (unit.patterns and
                        unit.preservation_score >
                        self.preservation_threshold):
                    # Look ahead for related patterns
                    look_ahead = min(3, len(units) - i - 1)
                    related_units = []
                    
                    for j in range(1, look_ahead + 1):
                        next_unit = units[i + j]
                        if self._are_units_related(unit, next_unit):
                            related_units.append(next_unit)
                    
                    if related_units:
                        # Merge related units
                        merged = self._merge_units(
                            [unit] + related_units
                        )
                        optimized.append(merged)
                        i += len(related_units) + 1
                    else:
                        optimized.append(unit)
                        i += 1
                else:
                    optimized.append(unit)
                    i += 1
            
            return optimized
            
        except Exception as e:
            print(f"Error optimizing preservation: {str(e)}")
            return units
    
    def _generate_binary(self, units: List[TranslationUnit]) -> bytes:
        """Generate final binary output from translation units."""
        try:
            result = bytearray()
            
            for unit in units:
                if unit.patterns and unit.preservation_score > self.preservation_threshold:
                    # Use pattern-aware encoding
                    best_mapping = max(
                        unit.mappings.values(),
                        key=lambda m: m.mapping_confidence
                    )
                    result.extend(best_mapping.binary_form)
                else:
                    # Use direct encoding
                    result.extend(unit.content)
            
            return bytes(result)
            
        except Exception as e:
            print(f"Error generating binary: {str(e)}")
            return b''
    
    def _extract_units(self, binary: bytes) -> List[TranslationUnit]:
        """Extract translation units from binary data."""
        units = []
        try:
            # Look for pattern markers
            pos = 0
            while pos < len(binary):
                if (binary[pos] == self.mapper.PATTERN_START and
                        pos + 1 < len(binary)):
                    # Extract pattern-encoded section
                    end_pos = binary.find(self.mapper.PATTERN_END, pos + 1)
                    if end_pos == -1:
                        end_pos = len(binary)
                    
                    section = binary[pos:end_pos + 1]
                    encoding_type = self.mapper.detect_encoding_type(section)
                    
                    # Create unit with pattern
                    pattern = self.mapper.map_from_binary(
                        section, encoding_type
                    )
                    if pattern:
                        unit = TranslationUnit(
                            content=section,
                            patterns={'p0': pattern},
                            mappings={}
                        )
                        units.append(unit)
                    
                    pos = end_pos + 1
                else:
                    # Extract non-pattern section
                    next_pattern = binary.find(
                        self.mapper.PATTERN_START, pos + 1
                    )
                    if next_pattern == -1:
                        next_pattern = len(binary)
                    
                    section = binary[pos:next_pattern]
                    unit = TranslationUnit(
                        content=section,
                        patterns={},
                        mappings={}
                    )
                    units.append(unit)
                    
                    pos = next_pattern
            
            return units
            
        except Exception as e:
            print(f"Error extracting units: {str(e)}")
            return units
    
    def _recover_patterns(
        self,
        units: List[TranslationUnit]
    ) -> List[TranslationUnit]:
        """Recover patterns from binary translation units."""
        try:
            for unit in units:
                if not unit.patterns:
                    # Try to detect patterns in content
                    data = np.frombuffer(unit.content, dtype=np.uint8)
                    pattern = self.hierarchy.detect_natural_pattern(data)
                    if pattern and pattern.confidence > self.min_pattern_confidence:
                        unit.patterns['p0'] = pattern
                        
                        # Create mapping
                        mapping = self.mapper.map_to_binary(pattern, data)
                        unit.mappings['p0'] = mapping
                        unit.preservation_score = mapping.mapping_confidence
            
            return units
            
        except Exception as e:
            print(f"Error recovering patterns: {str(e)}")
            return units
    
    def _optimize_recovery(
        self,
        units: List[TranslationUnit]
    ) -> List[TranslationUnit]:
        """Optimize pattern recovery and preservation."""
        try:
            # Similar to _optimize_preservation but for recovery
            return self._optimize_preservation(units)
            
        except Exception as e:
            print(f"Error optimizing recovery: {str(e)}")
            return units
    
    def _generate_content(
        self,
        units: List[TranslationUnit]
    ) -> Optional[str]:
        """Generate final content from translation units."""
        try:
            result = bytearray()
            
            for unit in units:
                if unit.patterns and unit.preservation_score > self.preservation_threshold:
                    # Use pattern-aware decoding
                    best_pattern = max(
                        unit.patterns.values(),
                        key=lambda p: p.confidence
                    )
                    sequence = (
                        best_pattern.sequence if best_pattern.sequence
                        else [0]
                    )
                    # Convert sequence to bytes
                    byte_data = [
                        int(x) % 256 for x in sequence
                    ]
                    result.extend(bytes(byte_data))
                else:
                    # Use direct content
                    result.extend(unit.content)
            
            return result.decode()
            
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            return None
    
    def _are_units_related(
        self,
        unit1: TranslationUnit,
        unit2: TranslationUnit
    ) -> bool:
        """Check if two translation units are related."""
        try:
            if not (unit1.patterns and unit2.patterns):
                return False
            
            # Check pattern relationships
            for p1 in unit1.patterns.values():
                for p2 in unit2.patterns.values():
                    if p1.principle_type == p2.principle_type:
                        return True
                    
                    if (p1.principle_type in self.hierarchy.relationships and
                            p2.principle_type in
                            self.hierarchy.relationships[p1.principle_type]):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _merge_units(
        self,
        units: List[TranslationUnit]
    ) -> TranslationUnit:
        """Merge related translation units."""
        try:
            if not units:
                return TranslationUnit(
                    content=b'',
                    patterns={},
                    mappings={}
                )
            
            # Combine content
            content = bytearray()
            for unit in units:
                content.extend(unit.content)
            
            # Combine patterns and mappings
            patterns = {}
            mappings = {}
            resonance_score = 0.0
            preservation_score = 0.0
            
            for unit in units:
                patterns.update(unit.patterns)
                mappings.update(unit.mappings)
                resonance_score = max(resonance_score, unit.resonance_score)
                preservation_score = max(
                    preservation_score,
                    unit.preservation_score
                )
            
            return TranslationUnit(
                content=bytes(content),
                patterns=patterns,
                mappings=mappings,
                resonance_score=resonance_score,
                preservation_score=preservation_score
            )
            
        except Exception as e:
            print(f"Error merging units: {str(e)}")
            return units[0] if units else TranslationUnit(
                content=b'',
                patterns={},
                mappings={}
            ) 