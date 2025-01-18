"""Unit tests for ALPHA evolution functionality."""

import unittest

from ALPHA.core.patterns.pattern_evolution import PatternEvolution


class TestEvolution(unittest.TestCase):
    """Test cases for evolution functionality."""

    def setUp(self):
        """Set up test environment."""
        self.evolution = PatternEvolution()

    def test_basic_functionality(self):
        """Test basic evolution functionality."""
        self.assertIsNotNone(self.evolution)
        self.assertIsInstance(self.evolution, PatternEvolution)

    def test_evolution_cycle(self):
        """Test evolution cycle execution."""
        # Create test pattern
        pattern = self.evolution.create_pattern([1, 2, 3, 5, 8])

        # Run evolution cycle
        evolved = self.evolution.evolve_pattern(pattern)

        # Verify evolution results
        self.assertIsNotNone(evolved)
        self.assertNotEqual(evolved.data, pattern.data)
        self.assertGreater(evolved.confidence, 0.0)

    def test_metrics_tracking(self):
        """Test evolution metrics tracking."""
        # Create and evolve multiple patterns
        patterns = [self.evolution.create_pattern([i, i * 2, i * 3]) for i in range(1, 4)]

        for pattern in patterns:
            self.evolution.evolve_pattern(pattern)

        # Check metrics
        metrics = self.evolution.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("success_rate", metrics)
        self.assertIn("adaptation_rate", metrics)
        self.assertIn("stability", metrics)

        # Verify metric ranges
        for value in metrics.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_pattern_detection(self):
        """Test natural pattern detection."""
        # Test Fibonacci sequence
        fib_data = [1, 1, 2, 3, 5, 8, 13, 21]
        patterns = self.evolution.detect_patterns(fib_data)

        self.assertGreater(len(patterns), 0)
        self.assertTrue(any(p.type == "fibonacci" for p in patterns))

        # Test golden ratio
        golden_data = [1, 1.618, 2.618, 4.236]
        patterns = self.evolution.detect_patterns(golden_data)

        self.assertGreater(len(patterns), 0)
        self.assertTrue(any(p.type == "golden_ratio" for p in patterns))

    def test_error_handling(self):
        """Test error handling in evolution system."""
        # Test invalid input
        with self.assertRaises(ValueError):
            self.evolution.create_pattern([])

        with self.assertRaises(TypeError):
            self.evolution.create_pattern(None)

        # Test evolution with invalid pattern
        invalid_pattern = self.evolution.create_pattern([1])
        invalid_pattern.data = None

        with self.assertRaises(ValueError):
            self.evolution.evolve_pattern(invalid_pattern)


if __name__ == "__main__":
    unittest.main()
