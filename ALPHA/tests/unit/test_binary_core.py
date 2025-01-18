"""Unit tests for ALPHA binary core functionality."""

import array
import unittest

from ALPHA.core.binary_foundation.base import Binary
from ALPHA.core.patterns.adaptive import Adaptive
from ALPHA.core.translation.translator import BinaryTranslator


class TestBinaryCore(unittest.TestCase):
    """Test cases for ALPHA binary core functionality."""

    def setUp(self):
        """Set up test environment."""
        self.binary = Binary(size=16)  # Larger size for more tests
        self.translator = BinaryTranslator()
        self.adaptive = Adaptive()

        # Test patterns
        self.test_patterns = {
            "simple": array.array("B", [0x01, 0xFF]),
            "fibonacci": array.array("B", [1, 1, 2, 3, 5, 8]),
            "code": "def test(): return 42",
        }

    def test_binary_operations(self):
        """Test binary pattern operations."""
        # Test bit operations
        self.binary.set_bit(0, True)
        self.binary.set_bit(7, True)
        self.binary.set_bit(15, True)

        # Verify bit states
        self.assertTrue(self.binary.get_bit(0))
        self.assertTrue(self.binary.get_bit(7))
        self.assertTrue(self.binary.get_bit(15))
        self.assertFalse(self.binary.get_bit(1))

        # Test bit patterns
        pattern = self.binary.get_pattern()
        self.assertIsInstance(pattern, bytes)
        self.assertEqual(len(pattern), 2)  # 16 bits = 2 bytes

        # Test pattern matching
        match = self.binary.match_pattern(pattern)
        self.assertGreaterEqual(match, 0.9)  # Should be exact match

    def test_binary_manipulation(self):
        """Test binary data manipulation."""
        # Test append operations
        self.binary.append(self.test_patterns["simple"])
        data = self.binary.to_bytes()

        # Verify data integrity
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)
        self.assertIn(self.test_patterns["simple"].tobytes(), data)

        # Test clear operation
        self.binary.clear()
        self.assertEqual(len(self.binary.to_bytes()), 2)  # Just header

        # Test pattern extraction
        self.binary.append(self.test_patterns["fibonacci"])
        patterns = self.binary.extract_patterns()
        self.assertGreater(len(patterns), 0)
        self.assertTrue(
            any(p.tobytes() == self.test_patterns["fibonacci"].tobytes() for p in patterns)
        )

    def test_translation_core(self):
        """Test binary translation core functionality."""
        # Test code translation
        binary_result = self.translator.to_binary(self.test_patterns["code"])

        # Verify translation results
        self.assertIsInstance(binary_result, bytes)
        self.assertGreater(len(binary_result), 0)

        # Test reverse translation
        code_result = self.translator.from_binary(binary_result)
        self.assertIsInstance(code_result, str)
        self.assertIn("def test", code_result)
        self.assertIn("return 42", code_result)

        # Test translation metrics
        metrics = self.translator.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("confidence", metrics)
        self.assertGreaterEqual(metrics["confidence"], 0.0)
        self.assertLessEqual(metrics["confidence"], 1.0)

    def test_adaptive_learning(self):
        """Test adaptive pattern learning."""
        # Test pattern learning
        pattern_id = "test_pattern"
        pattern_data = self.test_patterns["fibonacci"]

        # Learn pattern
        self.adaptive.learn(pattern_id, pattern_data)

        # Verify learning
        self.assertIn(pattern_id, self.adaptive.patterns)
        self.assertIn(pattern_id, self.adaptive.success_rates)
        self.assertGreaterEqual(self.adaptive.success_rates[pattern_id], 0.0)

        # Test pattern recognition
        confidence = self.adaptive.recognize(pattern_data)
        self.assertGreaterEqual(confidence, 0.5)

        # Test pattern evolution
        evolved = self.adaptive.evolve(pattern_id)
        self.assertIsNotNone(evolved)
        self.assertNotEqual(evolved, pattern_data)

    def test_error_handling(self):
        """Test error handling in binary operations."""
        # Test invalid binary operations
        with self.assertRaises(IndexError):
            self.binary.set_bit(16, True)  # Out of range

        with self.assertRaises(ValueError):
            Binary(size=0)  # Invalid size

        # Test invalid translations
        with self.assertRaises(Exception):
            self.translator.to_binary("invalid { python code")

        # Test invalid pattern learning
        with self.assertRaises(ValueError):
            self.adaptive.learn("", array.array("B", []))  # Empty pattern

    def test_pattern_compatibility(self):
        """Test pattern compatibility between components."""
        # Create test pattern
        self.binary.append(self.test_patterns["fibonacci"])
        binary_data = self.binary.to_bytes()

        # Test with translator
        translated = self.translator.from_binary(binary_data)
        self.assertIsNotNone(translated)

        # Test with adaptive system
        pattern_id = "compatibility_test"
        self.adaptive.learn(pattern_id, binary_data)

        # Verify cross-component compatibility
        confidence = self.adaptive.recognize(binary_data)
        self.assertGreaterEqual(confidence, 0.5)

        # Test pattern evolution compatibility
        evolved = self.adaptive.evolve(pattern_id)
        self.assertIsInstance(evolved, (bytes, array.array))

    def test_performance_constraints(self):
        """Test performance constraints and limits."""
        # Test large pattern handling
        large_pattern = array.array("B", [i % 256 for i in range(1024)])

        # Test binary system
        self.binary.append(large_pattern)
        result = self.binary.to_bytes()
        self.assertLess(
            len(result),
            2048,  # Should compress or limit size
            "Binary result too large",
        )

        # Test translator system
        large_code = "def test():\n" + "\n".join([f"    x_{i} = {i}" for i in range(100)])
        binary_result = self.translator.to_binary(large_code)
        self.assertLess(
            len(binary_result),
            4096,  # Should have reasonable size limit
            "Translation result too large",
        )

        # Test adaptive system
        self.adaptive.learn("large_pattern", large_pattern)
        self.assertLess(
            len(self.adaptive.patterns),
            1000,  # Should limit pattern storage
            "Too many patterns stored",
        )
