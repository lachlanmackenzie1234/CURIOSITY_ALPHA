"""Unit tests for basic ALPHA functionality."""

import unittest

from ALPHA.core.binary_foundation.base import Binary
from ALPHA.core.patterns.adaptive import Adaptive
from ALPHA.core.translation.translator import BinaryTranslator


class TestBasicFunctionality(unittest.TestCase):
    """Test cases for basic ALPHA functionality."""

    def test_binary(self):
        """Test binary pattern functionality."""
        binary = Binary(size=8)
        binary.set_bit(0, True)
        binary.set_bit(7, True)
        self.assertTrue(binary.get_bit(0))
        self.assertTrue(binary.get_bit(7))
        self.assertFalse(binary.get_bit(1))

    def test_translator(self):
        """Test translator functionality."""
        translator = BinaryTranslator()
        code = "def test(): return 42"
        pattern, metrics = translator.to_binary(code)
        self.assertIsInstance(pattern, Binary)
        self.assertIn("confidence", metrics)

    def test_adaptive(self):
        """Test adaptive functionality."""
        adaptive = Adaptive()
        pattern_id = "test"
        pattern_data = Binary(size=8).data
        adaptive.learn(pattern_id, pattern_data)
        self.assertIn(pattern_id, adaptive.patterns)
        self.assertIn(pattern_id, adaptive.success_rates)


if __name__ == "__main__":
    unittest.main()
