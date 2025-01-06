"""Binary pattern mapping system.

This module provides mechanisms for mapping patterns to and from binary
representations while preserving their natural mathematical properties.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from enum import Enum

from .natural_patterns import NaturalPattern, NaturalPatternHierarchy
from .resonance import PatternResonance, ResonanceProfile, ResonanceType


class BinaryEncodingType(Enum):
    """Types of binary pattern encoding."""
    DIRECT = "direct"          # Direct byte representation
    COMPRESSED = "compressed"  # Compressed representation
    RESONANT = "resonant"     # Resonance-preserving encoding
    SYMBOLIC = "symbolic"      # Symbol-based encoding


@dataclass
class BinaryMapping:
    """Represents a mapping between pattern and binary form."""
    
    pattern: NaturalPattern
    encoding_type: BinaryEncodingType
    binary_form: bytes
    resonance_preserved: float = 0.0
    structure_preserved: float = 0.0
    mapping_confidence: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


class PatternMapper:
    """Maps patterns to and from binary representations."""
    
    def __init__(self):
        """Initialize the pattern mapper."""
        self.hierarchy = NaturalPatternHierarchy()
        self.resonance = PatternResonance()
        self.mappings: Dict[str, BinaryMapping] = {}
        self.encoding_cache: Dict[str, bytes] = {}
        
        # Mapping thresholds
        self.preservation_threshold = 0.8
        self.confidence_threshold = 0.7
        
        # Special markers
        self.PATTERN_START = 0x7F
        self.PATTERN_END = 0x7E
        self.RESONANCE_MARKER = 0x7D
    
    def map_to_binary(
        self,
        pattern: NaturalPattern,
        context_data: Optional[np.ndarray] = None
    ) -> BinaryMapping:
        """Map a pattern to its binary representation."""
        try:
            # Calculate resonance if context provided
            resonance_profile = None
            if context_data is not None:
                resonance_profile = (
                    self.resonance.calculate_resonance(
                        pattern,
                        context_data
                    )
                )
            
            # Choose encoding type based on pattern properties
            encoding_type = self._select_encoding_type(
                pattern, resonance_profile
            )
            
            # Encode pattern
            binary_form = self._encode_pattern(
                pattern, encoding_type, resonance_profile
            )
            
            # Calculate preservation metrics
            resonance_preserved = self._calculate_resonance_preservation(
                pattern, binary_form, resonance_profile
            )
            
            structure_preserved = self._calculate_structure_preservation(
                pattern, binary_form
            )
            
            # Create mapping
            mapping = BinaryMapping(
                pattern=pattern,
                encoding_type=encoding_type,
                binary_form=binary_form,
                resonance_preserved=resonance_preserved,
                structure_preserved=structure_preserved,
                mapping_confidence=min(resonance_preserved, structure_preserved)
            )
            
            # Cache successful mapping
            if mapping.mapping_confidence > self.confidence_threshold:
                pattern_id = f"pattern_{len(self.mappings)}"
                self.mappings[pattern_id] = mapping
            
            return mapping
            
        except Exception as e:
            print(f"Error mapping pattern to binary: {str(e)}")
            return self._create_fallback_mapping(pattern)
    
    def map_from_binary(
        self,
        binary_data: bytes,
        encoding_type: Optional[BinaryEncodingType] = None
    ) -> Optional[NaturalPattern]:
        """Map binary data back to a pattern."""
        try:
            # Detect encoding type if not provided
            if encoding_type is None:
                encoding_type = self._detect_encoding_type(binary_data)
            
            # Decode based on encoding type
            if encoding_type == BinaryEncodingType.RESONANT:
                return self._decode_resonant_pattern(binary_data)
            elif encoding_type == BinaryEncodingType.COMPRESSED:
                return self._decode_compressed_pattern(binary_data)
            elif encoding_type == BinaryEncodingType.SYMBOLIC:
                return self._decode_symbolic_pattern(binary_data)
            else:
                return self._decode_direct_pattern(binary_data)
            
        except Exception as e:
            print(f"Error mapping binary to pattern: {str(e)}")
            return None
    
    def _select_encoding_type(
        self,
        pattern: NaturalPattern,
        resonance_profile: Optional[ResonanceProfile]
    ) -> BinaryEncodingType:
        """Select the best encoding type for a pattern."""
        try:
            if resonance_profile:
                if resonance_profile.resonance_type == ResonanceType.HARMONIC:
                    return BinaryEncodingType.RESONANT
                elif resonance_profile.stability > 0.8:
                    return BinaryEncodingType.COMPRESSED
            
            # Check pattern properties
            if pattern.confidence > 0.9:
                return BinaryEncodingType.SYMBOLIC
            
            return BinaryEncodingType.DIRECT
            
        except Exception:
            return BinaryEncodingType.DIRECT
    
    def _encode_pattern(
        self,
        pattern: NaturalPattern,
        encoding_type: BinaryEncodingType,
        resonance_profile: Optional[ResonanceProfile]
    ) -> bytes:
        """Encode a pattern using specified encoding type."""
        try:
            if encoding_type == BinaryEncodingType.RESONANT:
                return self._encode_resonant_pattern(
                    pattern, resonance_profile
                )
            elif encoding_type == BinaryEncodingType.COMPRESSED:
                return self._encode_compressed_pattern(pattern)
            elif encoding_type == BinaryEncodingType.SYMBOLIC:
                return self._encode_symbolic_pattern(pattern)
            else:
                return self._encode_direct_pattern(pattern)
                
        except Exception:
            return self._encode_direct_pattern(pattern)
    
    def _encode_resonant_pattern(
        self,
        pattern: NaturalPattern,
        resonance_profile: Optional[ResonanceProfile]
    ) -> bytes:
        """Encode pattern while preserving resonance properties."""
        try:
            encoded = bytearray([self.PATTERN_START])
            
            # Add pattern type
            encoded.extend(pattern.principle_type.value.encode())
            
            # Add resonance information if available
            if resonance_profile:
                encoded.append(self.RESONANCE_MARKER)
                encoded.append(int(resonance_profile.strength * 255))
                encoded.append(int(resonance_profile.harmony * 255))
                encoded.append(int(resonance_profile.stability * 255))
            
            # Add pattern data with resonance preservation
            if pattern.sequence:
                # Preserve sequence patterns
                for value in pattern.sequence:
                    encoded.append(int(value % 256))
            else:
                # Preserve principle-based patterns
                principle_value = self.hierarchy.constants.get(
                    pattern.principle_type, 0
                )
                encoded.extend(
                    int(principle_value * 1000).to_bytes(4, byteorder='big')
                )
            
            encoded.append(self.PATTERN_END)
            return bytes(encoded)
            
        except Exception:
            return self._encode_direct_pattern(pattern)
    
    def _encode_compressed_pattern(
        self,
        pattern: NaturalPattern
    ) -> bytes:
        """Encode pattern using compression."""
        try:
            encoded = bytearray([self.PATTERN_START])
            
            # Add pattern type
            encoded.extend(pattern.principle_type.value.encode())
            
            # Compress pattern properties
            properties = bytearray()
            for key, value in pattern.properties.items():
                properties.extend(key.encode()[:2])  # First 2 chars of key
                properties.append(int(value * 255))  # Normalized value
            
            # Add compressed properties
            encoded.append(len(properties))
            encoded.extend(properties)
            
            encoded.append(self.PATTERN_END)
            return bytes(encoded)
            
        except Exception:
            return self._encode_direct_pattern(pattern)
    
    def _encode_symbolic_pattern(
        self,
        pattern: NaturalPattern
    ) -> bytes:
        """Encode pattern using symbolic representation."""
        try:
            encoded = bytearray([self.PATTERN_START])
            
            # Add pattern type symbol
            type_symbol = {
                'golden_ratio': b'\xF1',
                'fibonacci': b'\xF2',
                'pi': b'\xF3',
                'e': b'\xF4'
            }.get(pattern.principle_type.value, b'\xF0')
            
            encoded.extend(type_symbol)
            
            # Add confidence as symbol
            conf_symbol = int(pattern.confidence * 15) + 0xE0
            encoded.append(conf_symbol)
            
            encoded.append(self.PATTERN_END)
            return bytes(encoded)
            
        except Exception:
            return self._encode_direct_pattern(pattern)
    
    def _encode_direct_pattern(
        self,
        pattern: NaturalPattern
    ) -> bytes:
        """Encode pattern directly without special handling."""
        try:
            return pattern.principle_type.value.encode()
        except Exception:
            return b''
    
    def _decode_resonant_pattern(
        self,
        binary_data: bytes
    ) -> Optional[NaturalPattern]:
        """Decode a resonant pattern encoding."""
        try:
            if len(binary_data) < 7:  # Minimum length for resonant pattern
                return None
            
            # Extract resonance information
            strength = binary_data[3] / 255.0
            harmony = binary_data[4] / 255.0
            stability = binary_data[5] / 255.0
            
            # Create pattern
            pattern = self.hierarchy.detect_natural_pattern(
                np.frombuffer(binary_data[6:-1], dtype=np.uint8)
            )
            
            if pattern:
                pattern.resonance = harmony
                pattern.confidence = strength
                pattern.properties['stability'] = stability
            
            return pattern
            
        except Exception:
            return None
    
    def _decode_compressed_pattern(
        self,
        binary_data: bytes
    ) -> Optional[NaturalPattern]:
        """Decode a compressed pattern encoding."""
        try:
            if len(binary_data) < 4:
                return None
            
            # Extract properties length
            prop_length = binary_data[2]
            if len(binary_data) < prop_length + 4:
                return None
            
            # Extract properties
            properties = {}
            pos = 3
            while pos < prop_length + 3:
                key = binary_data[pos:pos + 2].decode()
                value = binary_data[pos + 2] / 255.0
                properties[key] = value
                pos += 3
            
            # Create pattern
            pattern = self.hierarchy.detect_natural_pattern(
                np.array([0], dtype=np.uint8)  # Placeholder
            )
            if pattern:
                pattern.properties = properties
            
            return pattern
            
        except Exception:
            return None
    
    def _decode_symbolic_pattern(
        self,
        binary_data: bytes
    ) -> Optional[NaturalPattern]:
        """Decode a symbolic pattern encoding."""
        try:
            if len(binary_data) < 4:
                return None
            
            # Decode type symbol
            type_map = {
                0xF1: 'golden_ratio',
                0xF2: 'fibonacci',
                0xF3: 'pi',
                0xF4: 'e'
            }
            
            principle_type = type_map.get(binary_data[1], 'unknown')
            confidence = (binary_data[2] - 0xE0) / 15.0
            
            # Create pattern with decoded type
            pattern = self.hierarchy.detect_natural_pattern(
                np.array([0], dtype=np.uint8)  # Placeholder
            )
            if pattern:
                pattern.confidence = confidence
                pattern.principle_type = principle_type
            
            return pattern
            
        except Exception:
            return None
    
    def _decode_direct_pattern(
        self,
        binary_data: bytes
    ) -> Optional[NaturalPattern]:
        """Decode a direct pattern encoding."""
        try:
            return self.hierarchy.detect_natural_pattern(
                np.frombuffer(binary_data, dtype=np.uint8)
            )
        except Exception:
            return None
    
    def _calculate_resonance_preservation(
        self,
        pattern: NaturalPattern,
        binary_form: bytes,
        resonance_profile: Optional[ResonanceProfile]
    ) -> float:
        """Calculate how well resonance is preserved in binary form."""
        try:
            if not resonance_profile:
                return 1.0
            
            # Decode binary form
            decoded = self.map_from_binary(binary_form)
            if not decoded:
                return 0.0
            
            # Calculate new resonance
            new_profile = self.resonance.calculate_resonance(
                decoded,
                np.frombuffer(binary_form, dtype=np.uint8)
            )
            
            # Compare resonance properties
            harmony_preserved = abs(
                resonance_profile.harmony - new_profile.harmony
            )
            strength_preserved = abs(
                resonance_profile.strength - new_profile.strength
            )
            stability_preserved = abs(
                resonance_profile.stability - new_profile.stability
            )
            
            # Calculate preservation score
            preservation = 1.0 - (
                harmony_preserved * 0.4 +
                strength_preserved * 0.3 +
                stability_preserved * 0.3
            )
            
            return float(np.clip(preservation, 0.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _calculate_structure_preservation(
        self,
        pattern: NaturalPattern,
        binary_form: bytes
    ) -> float:
        """Calculate how well pattern structure is preserved."""
        try:
            # Decode binary form
            decoded = self.map_from_binary(binary_form)
            if not decoded:
                return 0.0
            
            # Compare principle types
            if decoded.principle_type != pattern.principle_type:
                return 0.0
            
            # Compare properties
            prop_similarity = self._calculate_property_similarity(
                pattern.properties,
                decoded.properties
            )
            
            # Compare confidence
            conf_similarity = abs(pattern.confidence - decoded.confidence)
            
            # Calculate overall preservation
            preservation = (
                float(decoded.principle_type == pattern.principle_type) * 0.4 +
                prop_similarity * 0.3 +
                (1.0 - conf_similarity) * 0.3
            )
            
            return float(np.clip(preservation, 0.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _calculate_property_similarity(
        self,
        props1: Dict[str, float],
        props2: Dict[str, float]
    ) -> float:
        """Calculate similarity between two property dictionaries."""
        try:
            if not props1 or not props2:
                return 0.0
            
            # Get common keys
            common_keys = set(props1.keys()) & set(props2.keys())
            if not common_keys:
                return 0.0
            
            # Calculate average difference
            differences = [
                abs(props1[key] - props2[key])
                for key in common_keys
            ]
            
            return 1.0 - (sum(differences) / len(differences))
            
        except Exception:
            return 0.0
    
    def _detect_encoding_type(self, binary_data: bytes) -> BinaryEncodingType:
        """Detect the encoding type of binary data."""
        try:
            if len(binary_data) < 2:
                return BinaryEncodingType.DIRECT
            
            if binary_data[0] != self.PATTERN_START:
                return BinaryEncodingType.DIRECT
            
            # Check for resonance marker
            if self.RESONANCE_MARKER in binary_data[:4]:
                return BinaryEncodingType.RESONANT
            
            # Check for symbolic encoding
            if binary_data[1] in {0xF1, 0xF2, 0xF3, 0xF4}:
                return BinaryEncodingType.SYMBOLIC
            
            # Check for compressed encoding
            if len(binary_data) > 3 and binary_data[2] < 64:  # Property length
                return BinaryEncodingType.COMPRESSED
            
            return BinaryEncodingType.DIRECT
            
        except Exception:
            return BinaryEncodingType.DIRECT
    
    def _create_fallback_mapping(
        self,
        pattern: NaturalPattern
    ) -> BinaryMapping:
        """Create a fallback mapping when normal mapping fails."""
        return BinaryMapping(
            pattern=pattern,
            encoding_type=BinaryEncodingType.DIRECT,
            binary_form=self._encode_direct_pattern(pattern),
            resonance_preserved=0.0,
            structure_preserved=0.0,
            mapping_confidence=0.0
        ) 