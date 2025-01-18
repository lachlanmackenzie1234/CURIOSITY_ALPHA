"""Tests for binary foundation optimizations."""

import array
import unittest

import numpy as np

from ALPHA.core.binary_foundation.base import Binary


class TestBinaryFoundation(unittest.TestCase):
    """Test cases for binary foundation optimizations."""

    def setUp(self):
        """Set up test environment."""
        self.binary = Binary()
        self.test_data = b"test pattern data"
        self.test_pattern = b"PHI:"  # Golden ratio pattern

    def test_pattern_encoding(self):
        """Test optimized pattern encoding."""
        # Encode pattern
        self.binary.encode_pattern(self.test_pattern, self.test_data, 0.95)

        # Verify data structure
        self.assertGreater(len(self.binary._data), 0)
        self.assertEqual(bytes(self.binary._data[:4]), Binary.MARKER_PATTERN)

        # Verify pattern tracking
        pattern_type = self.test_pattern.decode("ascii").rstrip(":")
        self.assertIn(pattern_type, self.binary.patterns)
        pattern_list = self.binary.patterns[pattern_type]
        self.assertEqual(len(pattern_list), 1)

        # Verify pattern info
        pattern_info = pattern_list[0]
        self.assertEqual(pattern_info.type_code, pattern_type)
        self.assertEqual(pattern_info.confidence, 0.95)
        self.assertEqual(pattern_info.data, self.test_data)

    def test_pattern_caching(self):
        """Test pattern caching mechanism."""
        # Encode pattern
        self.binary.encode_pattern(self.test_pattern, self.test_data, 0.95)

        # First decode should cache
        pattern1 = self.binary.decode_pattern(0)
        self.assertIsNotNone(pattern1)

        # Second decode should use cache
        pattern2 = self.binary.decode_pattern(0)
        self.assertIsNotNone(pattern2)

        # Both should be the same object
        self.assertEqual(pattern1.type_code, pattern2.type_code)
        self.assertEqual(pattern1.confidence, pattern2.confidence)
        self.assertEqual(pattern1.data, pattern2.data)

        # Verify cache
        pattern_type = self.test_pattern.decode("ascii").rstrip(":")
        self.assertIn(pattern_type, self.binary._pattern_cache)

    def test_pattern_analysis(self):
        """Test pattern analysis with optimizations."""
        # Create data with golden ratio pattern
        phi = (1 + np.sqrt(5)) / 2
        pattern_data = array.array("B", [10, int(10 * phi) % 256, int(10 * phi * phi) % 256])

        # Create binary with pattern
        binary = Binary(pattern_data.tobytes())

        # Analyze patterns
        patterns = binary.analyze_patterns()

        # Verify golden ratio detection
        self.assertIn("golden", patterns)
        self.assertGreater(patterns["golden"][0][0], 0.6)

    def test_memory_efficiency(self):
        """Test memory efficiency of binary operations."""
        # Create large pattern
        large_data = bytes([i % 256 for i in range(1000)])

        # Initial memory state
        initial_size = self.binary.get_size()

        # Encode pattern
        self.binary.encode_pattern(self.test_pattern, large_data, 0.95)

        # Final memory state
        final_size = self.binary.get_size()

        # Verify efficient memory usage
        expected_size = initial_size + Binary.PATTERN_HEADER_SIZE + len(large_data)
        self.assertEqual(final_size, expected_size)

    def test_data_integrity(self):
        """Test data integrity through binary operations."""
        # Original data
        original_data = bytes([i % 256 for i in range(100)])
        binary = Binary(original_data)

        # Encode pattern
        binary.encode_pattern(self.test_pattern, self.test_data, 0.95)

        # Convert to bytes and back
        data_bytes = binary.to_bytes()
        new_binary = Binary()
        new_binary.from_bytes(data_bytes)

        # Verify data integrity
        self.assertEqual(binary.get_size(), new_binary.get_size())
        self.assertEqual(binary.to_bytes(), new_binary.to_bytes())

    def test_pattern_header_optimization(self):
        """Test pattern header size optimization."""
        # Encode pattern
        self.binary.encode_pattern(self.test_pattern, self.test_data, 0.95)

        # Verify header size
        header_size = (
            len(Binary.MARKER_PATTERN)
            + 4  # Pattern marker
            + 4  # Pattern type
            + 4  # Confidence (float)  # Length (uint32)
        )
        self.assertEqual(Binary.PATTERN_HEADER_SIZE, header_size)

        # Verify header content
        header = self.binary._data[: Binary.PATTERN_HEADER_SIZE]
        self.assertEqual(len(header), Binary.PATTERN_HEADER_SIZE)


if __name__ == "__main__":
    unittest.main()
