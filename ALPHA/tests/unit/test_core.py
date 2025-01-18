"""Unit tests for ALPHA core functionality."""

import unittest

from ALPHA.core.interface import ALPHACore, create_alpha


class TestALPHACore(unittest.TestCase):
    """Test cases for ALPHA core functionality."""

    def setUp(self):
        """Set up test environment."""
        self.alpha = create_alpha("test_instance")

    def test_initialization(self):
        """Test ALPHA initialization."""
        self.assertIsNotNone(self.alpha)
        self.assertIsInstance(self.alpha, ALPHACore)

        # Check initial state
        state = self.alpha.get_state()
        self.assertIsInstance(state, dict)
        self.assertIn("memory_usage", state)
        self.assertIn("pattern_count", state)
        self.assertIn("confidence", state)

    def test_basic_processing(self):
        """Test basic input processing."""
        test_inputs = [
            "def test(): return 42",
            bytes([0x00, 0xFF, 0xAA, 0x55]),
            [1, 2, 3, 5, 8, 13],
            {"type": "test", "data": "test data"},
        ]

        for input_data in test_inputs:
            result = self.alpha.process(input_data)

            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)
            self.assertIn("confidence", result)
            self.assertIn("patterns", result)

            # Check success
            self.assertEqual(result["status"], "success")
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    def test_error_handling(self):
        """Test error handling."""
        invalid_inputs = [
            None,
            "",
            bytes(),
            {"invalid": None},
            [1, 2, None, 3],
        ]

        for invalid_input in invalid_inputs:
            result = self.alpha.process(invalid_input)

            # Should handle gracefully
            self.assertEqual(result["status"], "error")
            self.assertIn("error_type", result)
            self.assertIn("error_message", result)

    def test_memory_management(self):
        """Test memory management."""
        # Process some data
        self.alpha.process("def test(): pass")

        # Get initial state
        initial_state = self.alpha.get_state()

        # Clear memory
        self.alpha.clear_memory()
        cleared_state = self.alpha.get_state()

        # Verify memory cleared
        self.assertLess(cleared_state["memory_usage"], initial_state["memory_usage"])
        self.assertEqual(cleared_state["pattern_count"], 0)


if __name__ == "__main__":
    unittest.main()
