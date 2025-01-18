"""Unit tests for binary translation system."""

import array
import unittest
from typing import Any, Dict

import numpy as np

from ALPHA.core.binary_foundation.base import Binary
from ALPHA.core.translation.translator import BinaryTranslator


class TestBinaryTranslation(unittest.TestCase):
    """Test cases for binary translation functionality."""

    def setUp(self):
        """Set up test environment."""
        self.translator = BinaryTranslator()
        self.test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        self.complex_test_code = """
class PatternGenerator:
    def __init__(self):
        self.sequence = []

    def generate_fibonacci(self, n):
        if n <= 1:
            return n
        return self.generate_fibonacci(n-1) + self.generate_fibonacci(n-2)

    def generate_exponential(self, base, n):
        return [base ** i for i in range(n)]
"""

    def test_basic_functionality(self):
        """Test basic translation functionality."""
        # Test to binary
        binary = self.translator.translate_to_binary(self.test_code)
        self.assertIsInstance(binary, Binary)
        self.assertTrue(binary.to_bytes())

        # Test from binary
        self.translator.set_binary(binary)
        code = self.translator.translate_from_binary()
        self.assertIsNotNone(code)

    def test_pattern_preservation(self):
        """Test pattern preservation during translation."""
        # Test data with known patterns
        patterns = {
            "fibonacci": array.array("B", [1, 1, 2, 3, 5, 8, 13]),
            "exponential": array.array("B", [1, 2, 4, 8, 16, 32, 64]),
            "golden": array.array(
                "B",
                [
                    10,
                    int(10 * ((1 + np.sqrt(5)) / 2)) % 256,
                    int(10 * ((1 + np.sqrt(5)) / 2) ** 2) % 256,
                ],
            ),
        }

        for pattern_type, data in patterns.items():
            # Create binary with pattern
            binary = Binary(bytes(data))
            self.translator.set_binary(binary)

            # Translate and check preservation
            new_binary = self.translator.translate_to_binary(self.test_code)
            detected = new_binary.analyze_patterns()

            # Verify pattern was preserved
            self.assertIn(
                pattern_type,
                detected,
                f"Failed to preserve {pattern_type} pattern",
            )

    def test_error_handling(self):
        """Test error handling in translation."""
        # Test invalid syntax
        invalid_code = "def invalid_syntax("
        binary = self.translator.translate_to_binary(invalid_code)
        self.assertIn("syntax_error", binary.metadata)

        # Test empty input
        binary = self.translator.translate_to_binary("")
        self.assertIsInstance(binary, Binary)

        # Test None input
        with self.assertRaises(Exception):
            self.translator.translate_to_binary(None)

        # Test large input
        large_code = "x = 1\n" * 1000
        binary = self.translator.translate_to_binary(large_code)
        self.assertIsInstance(binary, Binary)

    def test_translation_metrics(self):
        """Test translation quality metrics."""
        binary = self.translator.translate_to_binary(self.complex_test_code)

        # Check required metrics
        required_metrics = [
            "translation_pattern_preservation_score",
            "translation_translation_confidence",
            "translation_patterns_preserved",
            "translation_total_patterns",
        ]

        for metric in required_metrics:
            self.assertIn(metric, binary.metadata)

        # Validate metric values
        score = float(binary.metadata["translation_pattern_preservation_score"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_bidirectional_translation(self):
        """Test code preservation through binary translation."""
        # First translation
        binary = self.translator.translate_to_binary(self.complex_test_code)
        self.translator.set_binary(binary)

        # Back to code
        recovered_code = self.translator.translate_from_binary()
        self.assertIsNotNone(recovered_code)

        # Verify code structure
        import ast

        try:
            ast.parse(recovered_code)
        except SyntaxError:
            self.fail("Recovered code has syntax errors")
