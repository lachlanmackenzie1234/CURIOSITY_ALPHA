"""Binary translation layer for Unity-PRISM communication."""

import struct
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class BinaryPattern:
    """Container for binary pattern data."""

    pattern_type: str
    timestamp: float
    data: bytes
    checksum: int


class UnityBinaryTranslator:
    """Handles binary translation between Unity and PRISM."""

    PATTERN_TYPES = {"spatial": 0x01, "temporal": 0x02, "spectral": 0x03}

    @staticmethod
    def to_binary_pattern(unity_data: Dict[str, Any]) -> BinaryPattern:
        """Convert Unity data to binary pattern.

        Ensures precise binary representation for Binary_pulse system.
        """
        pattern_type = unity_data.get("pattern_type", "spatial")
        data = unity_data.get("data", {})

        # Convert data to numpy array first for precise binary control
        np_data = np.array(data, dtype=np.float32)

        # Add pattern type to the data array
        pattern_type_value = UnityBinaryTranslator.PATTERN_TYPES[pattern_type]
        np_data = np.concatenate([np_data, np.array([pattern_type_value], dtype=np.uint32)])

        # Create binary header
        header = struct.pack(
            "!BQ",
            pattern_type_value,
            int(time.time() * 1000),  # millisecond timestamp
        )

        # Convert numpy array to bytes
        data_bytes = np_data.tobytes()

        # Calculate checksum for data integrity
        checksum = zlib.crc32(header + data_bytes)

        return BinaryPattern(
            pattern_type=pattern_type,
            timestamp=time.time(),
            data=header + data_bytes,
            checksum=checksum,
        )

    @staticmethod
    def from_binary_pattern(binary_pattern: BinaryPattern) -> Dict[str, Any]:
        """Convert binary pattern back to Unity-compatible format.

        Maintains precise data fidelity through the conversion.
        """
        # Extract header
        type_and_time = struct.unpack("!BQ", binary_pattern.data[:9])
        pattern_type_byte, timestamp = type_and_time

        # Convert data bytes back to numpy array
        np_data = np.frombuffer(binary_pattern.data[9:], dtype=np.float32)

        # Verify checksum
        calculated_checksum = zlib.crc32(binary_pattern.data)
        if calculated_checksum != binary_pattern.checksum:
            raise ValueError("Binary pattern checksum mismatch")

        # Get pattern type string
        pattern_lookup = UnityBinaryTranslator.PATTERN_TYPES.items()
        pattern_type = next(
            (k for k, v in pattern_lookup if v == pattern_type_byte),
            "unknown",
        )

        return {
            "pattern_type": pattern_type,
            "timestamp": timestamp / 1000.0,  # Convert back to seconds
            "data": np_data.tolist(),
            "binary_verified": True,
        }
