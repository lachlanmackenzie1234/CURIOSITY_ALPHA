"""Unit tests for ALPHA pattern functionality."""

import unittest

from ALPHA.core.patterns.pattern import Pattern


class TestPattern(unittest.TestCase):
    """Test cases for pattern functionality."""

    def setUp(self):
        """Set up test patterns."""
        self.pattern1 = Pattern("test1")
        for i in range(16):
            self.pattern1.data.write(i, i % 2 == 0)  # 1010101010101010

        self.pattern2 = Pattern("test2")
        for i in range(16):
            self.pattern2.data.write(i, i % 3 == 0)  # 1001001001001001

    def test_pattern_creation(self):
        """Test pattern creation and data writing."""
        self.assertIsNotNone(self.pattern1)
        self.assertEqual(self.pattern1.name, "test1")
        self.assertEqual(len(self.pattern1.data), 16)

    def test_pattern_evolution(self):
        """Test pattern evolution."""
        evolved = self.pattern1.evolve(mutation_rate=0.2)
        self.assertIsNotNone(evolved)
        self.assertNotEqual(evolved.data, self.pattern1.data)

    def test_pattern_comparison(self):
        """Test pattern comparison."""
        similarity = self.pattern1.data.compare(self.pattern2.data)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_pattern_serialization(self):
        """Test pattern serialization."""
        data = self.pattern1.to_bytes()
        restored = Pattern.from_bytes("restored", data)
        self.assertEqual(restored.data, self.pattern1.data)


if __name__ == "__main__":
    unittest.main()
