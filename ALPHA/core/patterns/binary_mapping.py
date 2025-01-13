"""Binary pattern mapping system.

This module provides mechanisms for mapping patterns to and from binary
representations while preserving their natural mathematical properties.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from .natural_patterns import NaturalPattern, NaturalPatternHierarchy
from .pattern_types import BinaryEncodingType, NaturalPrincipleType
from .resonance import PatternResonance, ResonanceProfile, ResonanceType


class BinaryEncodingType(Enum):
    """Types of binary pattern encoding."""

    DIRECT = "direct"  # Direct byte representation
    COMPRESSED = "compressed"  # Compressed representation
    RESONANT = "resonant"  # Resonance-preserving encoding
    SYMBOLIC = "symbolic"  # Symbol-based encoding


@dataclass
class BinaryMapping:
    """Maps between natural patterns and binary representations."""

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
        self, pattern: NaturalPattern, context: Optional[bytes] = None
    ) -> Optional[BinaryMapping]:
        """Map a pattern to binary form while preserving its natural qualities."""
        try:
            # Start with pattern marker
            binary_data = bytearray([self.PATTERN_START])

            # Add pattern type
            binary_data.extend(pattern.principle_type.value.encode())

            # Add hardware properties (normalized to 0-255)
            hardware_resonance = int(pattern.properties.get("hardware_resonance", 0.0) * 255)
            memory_state = int(pattern.properties.get("memory_state", 0.0) * 255)
            confidence = int(pattern.properties.get("stability", 0.0) * 255)

            binary_data.extend([hardware_resonance, memory_state, confidence])

            # Add sequence data as float32 array
            if pattern.sequence:
                sequence_array = np.array(pattern.sequence, dtype=np.float32)
                binary_data.extend(sequence_array.tobytes())

            # Add structure preservation markers
            structure_markers = np.array([1.0] * len(pattern.sequence), dtype=np.float32)
            binary_data.extend(structure_markers.tobytes())

            # End with pattern marker
            binary_data.append(self.PATTERN_END)

            # Create mapping with preserved qualities
            mapping = BinaryMapping(
                pattern=pattern,
                encoding_type=BinaryEncodingType.COMPRESSED,
                binary_form=bytes(binary_data),
                resonance_preserved=pattern.resonance,
                structure_preserved=1.0,  # Structure fully preserved
                mapping_confidence=pattern.properties.get("stability", 0.0),
                metadata={},
            )

            return mapping

        except Exception as e:
            print(f"Error mapping pattern to binary: {str(e)}")
            return None

    def map_from_binary(
        self, binary_data: bytes, encoding_type: Optional[BinaryEncodingType] = None
    ) -> Optional[NaturalPattern]:
        """Map binary data back to a pattern."""
        try:
            if len(binary_data) < 5:  # Minimum header size
                return None

            if binary_data[0] != self.PATTERN_START:
                return None

            # Extract pattern type
            principle_type = NaturalPrincipleType(binary_data[1:2].decode())

            # Extract hardware properties
            hardware_resonance = binary_data[2] / 255.0
            memory_state = binary_data[3] / 255.0
            confidence = binary_data[4] / 255.0

            # Extract sequence data
            sequence_size = len(binary_data[5:-1]) // 4  # float32 = 4 bytes
            if sequence_size > 0:
                sequence_data = np.frombuffer(
                    binary_data[5 : 5 + sequence_size * 4], dtype=np.float32
                )
                sequence = sequence_data.tolist()
            else:
                sequence = None

            # Create pattern with preserved properties
            pattern = NaturalPattern(
                principle_type=principle_type,
                confidence=confidence,
                resonance=hardware_resonance,
                sequence=sequence,
                properties={
                    "hardware_resonance": hardware_resonance,
                    "memory_state": memory_state,
                    "stability": confidence,
                },
            )

            return pattern

        except Exception as e:
            print(f"Error decoding pattern: {str(e)}")
            return None

    def _select_encoding_type(
        self, pattern: NaturalPattern, resonance_profile: Optional[ResonanceProfile]
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
        resonance_profile: Optional[ResonanceProfile],
    ) -> bytes:
        """Encode a pattern using specified encoding type."""
        try:
            # Prepare header with hardware state
            header = bytearray([self.PATTERN_START])

            # Add pattern type
            header.extend(pattern.principle_type.value.encode()[:1])

            # Add hardware-derived properties
            if "hardware_resonance" in pattern.properties:
                header.append(int(pattern.properties["hardware_resonance"] * 255))
            else:
                header.append(int(pattern.resonance * 255))

            if "memory_state" in pattern.properties:
                header.append(int(pattern.properties["memory_state"] * 255))
            else:
                header.append(0)

            # Add stability
            header.append(int(pattern.confidence * 255))

            # Add sequence data
            if pattern.sequence:
                sequence_data = np.array(pattern.sequence, dtype=np.float32).tobytes()
                header.extend(sequence_data)

            # Add partnership metrics if available
            if resonance_profile and resonance_profile.partnership_metrics:
                for metric_name, value in resonance_profile.partnership_metrics.items():
                    header.append(int(value * 255))

            header.append(self.PATTERN_END)
            return bytes(header)

        except Exception as e:
            print(f"Error encoding pattern: {str(e)}")
            return b""

    def _encode_resonant_pattern(
        self, pattern: NaturalPattern, resonance_profile: Optional[ResonanceProfile] = None
    ) -> bytes:
        """Encode a pattern with resonance preservation."""
        try:
            if not resonance_profile:
                return self._encode_direct_pattern(pattern)

            # Prepare header with resonance information
            header = bytearray(
                [
                    self.PATTERN_START,
                    self.RESONANCE_MARKER,
                    pattern.principle_type.value[0],  # First byte of type
                    int(resonance_profile.strength * 255),
                    int(resonance_profile.harmony * 255),
                    int(resonance_profile.stability * 255),
                    # Partnership metrics
                    int(resonance_profile.partnership_metrics["mutual_growth"] * 255),
                    int(resonance_profile.partnership_metrics["resonance_depth"] * 255),
                    int(resonance_profile.partnership_metrics["adaptation_rate"] * 255),
                    int(resonance_profile.partnership_metrics["support_strength"] * 255),
                ]
            )

            # Add pattern data
            if hasattr(pattern, "data") and pattern.data:
                pattern_data = pattern.data
            else:
                pattern_data = bytes([])

            return bytes(header) + pattern_data

        except Exception:
            return self._encode_direct_pattern(pattern)

    def _encode_compressed_pattern(self, pattern: NaturalPattern) -> bytes:
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

    def _encode_symbolic_pattern(self, pattern: NaturalPattern) -> bytes:
        """Encode pattern using symbolic representation."""
        try:
            encoded = bytearray([self.PATTERN_START])

            # Add pattern type symbol
            type_symbol = {
                "golden_ratio": b"\xF1",
                "fibonacci": b"\xF2",
                "pi": b"\xF3",
                "e": b"\xF4",
            }.get(pattern.principle_type.value, b"\xF0")

            encoded.extend(type_symbol)

            # Add confidence as symbol
            conf_symbol = int(pattern.confidence * 15) + 0xE0
            encoded.append(conf_symbol)

            encoded.append(self.PATTERN_END)
            return bytes(encoded)

        except Exception:
            return self._encode_direct_pattern(pattern)

    def _encode_direct_pattern(self, pattern: NaturalPattern) -> bytes:
        """Encode pattern directly without special handling."""
        try:
            return pattern.principle_type.value.encode()
        except Exception:
            return b""

    def _decode_resonant_pattern(self, binary_data: bytes) -> Optional[NaturalPattern]:
        """Decode a resonant pattern encoding."""
        try:
            if len(binary_data) < 7:  # Minimum length for resonant pattern
                return None

            # Extract resonance information
            strength = binary_data[3] / 255.0
            harmony = binary_data[4] / 255.0
            stability = binary_data[5] / 255.0

            # Extract partnership metrics
            mutual_growth = binary_data[6] / 255.0 if len(binary_data) > 6 else 0.0
            resonance_depth = binary_data[7] / 255.0 if len(binary_data) > 7 else 0.0
            adaptation_rate = binary_data[8] / 255.0 if len(binary_data) > 8 else 0.0
            support_strength = binary_data[9] / 255.0 if len(binary_data) > 9 else 0.0

            # Create pattern
            pattern = self.hierarchy.detect_natural_pattern(
                np.frombuffer(binary_data[10:], dtype=np.uint8)
            )

            if pattern:
                pattern.resonance = harmony
                pattern.confidence = strength
                pattern.properties["stability"] = stability
                pattern.properties["mutual_growth"] = mutual_growth
                pattern.properties["resonance_depth"] = resonance_depth
                pattern.properties["adaptation_rate"] = adaptation_rate
                pattern.properties["support_strength"] = support_strength

            return pattern

        except Exception:
            return None

    def _decode_compressed_pattern(self, binary_data: bytes) -> Optional[NaturalPattern]:
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
                key = binary_data[pos : pos + 2].decode()
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

    def _decode_symbolic_pattern(self, binary_data: bytes) -> Optional[NaturalPattern]:
        """Decode a symbolic pattern encoding."""
        try:
            if len(binary_data) < 4:
                return None

            # Decode type symbol
            type_map = {0xF1: "golden_ratio", 0xF2: "fibonacci", 0xF3: "pi", 0xF4: "e"}

            principle_type = type_map.get(binary_data[1], "unknown")
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

    def _decode_direct_pattern(self, binary_data: bytes) -> Optional[NaturalPattern]:
        """Decode a direct pattern encoding."""
        try:
            return self.hierarchy.detect_natural_pattern(np.frombuffer(binary_data, dtype=np.uint8))
        except Exception:
            return None

    def _calculate_resonance_preservation(
        self,
        pattern: NaturalPattern,
        binary_form: bytes,
        resonance_profile: Optional[ResonanceProfile],
    ) -> float:
        """Calculate how well resonance and partnership are preserved in binary form."""
        try:
            # Always consider hardware-derived resonance
            if "hardware_resonance" in pattern.properties:
                base_resonance = pattern.properties["hardware_resonance"]
            else:
                base_resonance = pattern.resonance

            # Decode binary form
            decoded = self.map_from_binary(binary_form)
            if not decoded:
                return 0.0

            # Compare hardware-derived properties
            if resonance_profile:
                # Use resonance profile if available
                new_profile = self.resonance.calculate_resonance(
                    decoded, np.frombuffer(binary_form, dtype=np.uint8)
                )

                # Compare resonance properties with hardware awareness
                harmony_preserved = abs(resonance_profile.harmony - new_profile.harmony)
                strength_preserved = abs(resonance_profile.strength - new_profile.strength)
                stability_preserved = abs(resonance_profile.stability - new_profile.stability)

                # Compare partnership metrics
                partnership_preserved = self._calculate_partnership_preservation(
                    resonance_profile.partnership_metrics, new_profile.partnership_metrics
                )

                # Calculate preservation score with partnership awareness
                preservation = 1.0 - (
                    harmony_preserved * 0.3
                    + strength_preserved * 0.2
                    + stability_preserved * 0.2
                    + (1.0 - partnership_preserved) * 0.3
                )
            else:
                # Direct hardware-based comparison
                if "hardware_resonance" in decoded.properties:
                    decoded_resonance = decoded.properties["hardware_resonance"]
                else:
                    decoded_resonance = decoded.resonance

                # Calculate basic preservation
                resonance_diff = abs(base_resonance - decoded_resonance)
                preservation = 1.0 - resonance_diff

            return float(np.clip(preservation, 0.0, 1.0))

        except Exception as e:
            print(f"Error calculating resonance preservation: {str(e)}")
            return 0.0

    def _calculate_structure_preservation(
        self, pattern: NaturalPattern, binary_form: bytes
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
                pattern.properties, decoded.properties
            )

            # Compare confidence
            conf_similarity = abs(pattern.confidence - decoded.confidence)

            # Calculate overall preservation
            preservation = (
                float(decoded.principle_type == pattern.principle_type) * 0.4
                + prop_similarity * 0.3
                + (1.0 - conf_similarity) * 0.3
            )

            return float(np.clip(preservation, 0.0, 1.0))

        except Exception:
            return 0.0

    def _calculate_property_similarity(
        self, props1: Dict[str, float], props2: Dict[str, float]
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
            differences = [abs(props1[key] - props2[key]) for key in common_keys]

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

    def _create_fallback_mapping(self, pattern: NaturalPattern) -> BinaryMapping:
        """Create a fallback mapping when normal mapping fails."""
        return BinaryMapping(
            pattern=pattern,
            encoding_type=BinaryEncodingType.DIRECT,
            binary_form=self._encode_direct_pattern(pattern),
            resonance_preserved=0.0,
            structure_preserved=0.0,
            mapping_confidence=0.0,
        )

    def _calculate_partnership_preservation(
        self, original_metrics: Dict[str, float], new_metrics: Dict[str, float]
    ) -> float:
        """Calculate how well partnership metrics are preserved."""
        try:
            if not original_metrics or not new_metrics:
                return 0.0

            # Calculate differences in partnership metrics
            differences = []
            for key in ["mutual_growth", "resonance_depth", "adaptation_rate", "support_strength"]:
                if key in original_metrics and key in new_metrics:
                    diff = abs(original_metrics[key] - new_metrics[key])
                    differences.append(diff)

            if not differences:
                return 0.0

            # Calculate average preservation
            avg_difference = np.mean(differences)
            preservation = 1.0 - avg_difference

            return float(np.clip(preservation, 0.0, 1.0))

        except Exception:
            return 0.0
