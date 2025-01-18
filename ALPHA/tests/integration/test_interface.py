"""Integration tests for ALPHA core interface."""

import array
import asyncio
import unittest

from ALPHA.core.interface import ALPHACore, create_alpha


class TestALPHAInterface(unittest.TestCase):
    """Integration tests for ALPHA interface."""

    def setUp(self):
        """Set up test environment."""
        self.alpha = create_alpha("test_instance")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Test data
        self.test_inputs = {
            "text": "def test_function():\n    return 42",
            "binary": bytes([0x00, 0xFF, 0xAA, 0x55]),
            "array": array.array("B", [1, 2, 3, 5, 8, 13]),
            "mixed": {
                "code": "x = 1\ny = 2\nz = x + y",
                "data": bytes([0x01, 0x02, 0x03]),
            },
        }

    def tearDown(self):
        """Clean up test environment."""
        self.loop.run_until_complete(self.alpha.stop())
        self.loop.close()

    def test_initialization(self):
        """Test ALPHA initialization."""
        # Verify instance creation
        self.assertIsInstance(self.alpha, ALPHACore)

        # Check initial state
        state = self.alpha.get_state()
        self.assertIsInstance(state, dict)
        self.assertIn("memory_usage", state)
        self.assertIn("pattern_count", state)
        self.assertIn("confidence", state)

    def test_basic_processing(self):
        """Test basic input processing."""
        # Process each input type
        for input_type, input_data in self.test_inputs.items():
            result = self.loop.run_until_complete(self.alpha.process(input_data))

            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)
            self.assertIn("confidence", result)
            self.assertIn("patterns", result)

            # Check confidence score
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    def test_pattern_analysis(self):
        """Test pattern analysis capabilities."""
        # Test code pattern analysis
        patterns = self.alpha.analyze_patterns(self.test_inputs["text"])

        # Verify pattern detection
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)

        # Check pattern structure
        for pattern in patterns:
            self.assertIn("type", pattern)
            self.assertIn("confidence", pattern)
            self.assertIn("metrics", pattern)

    def test_memory_management(self):
        """Test memory management functionality."""
        # Process some data
        self.loop.run_until_complete(self.alpha.process(self.test_inputs["mixed"]))

        # Get initial state
        initial_state = self.alpha.get_state()

        # Clear memory
        self.alpha.clear_memory()
        cleared_state = self.alpha.get_state()

        # Verify memory cleared
        self.assertLess(cleared_state["memory_usage"], initial_state["memory_usage"])
        self.assertEqual(cleared_state["pattern_count"], 0)

    def test_optimization(self):
        """Test system optimization."""
        # Process data to generate patterns
        for input_data in self.test_inputs.values():
            self.loop.run_until_complete(self.alpha.process(input_data))

        # Run optimization
        opt_results = self.alpha.optimize()

        # Verify optimization results
        self.assertIsInstance(opt_results, dict)
        self.assertIn("improvements", opt_results)
        self.assertIn("metrics", opt_results)

        # Check optimization impact
        self.assertGreater(opt_results["metrics"].get("efficiency", 0), 0.0)

    def test_concurrent_processing(self):
        """Test concurrent input processing."""

        async def process_concurrent():
            # Create multiple processing tasks
            tasks = [
                self.alpha.process(input_data, priority=i % 3)
                for i, input_data in enumerate(self.test_inputs.values())
            ]

            # Run concurrently
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent processing
        results = self.loop.run_until_complete(process_concurrent())

        # Verify all processed successfully
        self.assertEqual(len(results), len(self.test_inputs))
        for result in results:
            self.assertEqual(result["status"], "success")

    def test_error_handling(self):
        """Test error handling capabilities."""
        # Test invalid inputs
        invalid_inputs = [
            None,
            "",
            bytes(),
            {"invalid": None},
            [1, 2, None, 3],
        ]

        for invalid_input in invalid_inputs:
            result = self.loop.run_until_complete(self.alpha.process(invalid_input))

            # Should handle gracefully
            self.assertEqual(result["status"], "error")
            self.assertIn("error_type", result)
            self.assertIn("error_message", result)

    def test_confidence_scoring(self):
        """Test confidence score calculation."""

        async def check_confidence():
            # Process some data
            await self.alpha.process(self.test_inputs["mixed"])

            # Get confidence score
            score = await self.alpha.get_confidence_score()
            return score

        # Get confidence score
        score = self.loop.run_until_complete(check_confidence())

        # Verify score
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_state_management(self):
        """Test state management and updates."""
        # Get initial state
        initial_state = self.alpha.get_state()

        # Process data
        self.loop.run_until_complete(self.alpha.process(self.test_inputs["text"]))

        # Get updated state
        updated_state = self.alpha.get_state()

        # Verify state changes
        self.assertNotEqual(initial_state, updated_state)
        self.assertGreater(updated_state["pattern_count"], initial_state["pattern_count"])

        # Test state consistency
        for _ in range(3):
            current_state = self.alpha.get_state()
            self.assertEqual(current_state, updated_state)

    def test_system_lifecycle(self):
        """Test full system lifecycle."""

        async def lifecycle_test():
            # Start system
            await self.alpha.start()

            # Process data
            result1 = await self.alpha.process(self.test_inputs["text"])
            self.assertEqual(result1["status"], "success")

            # Optimize
            self.alpha.optimize()

            # Process more data
            result2 = await self.alpha.process(self.test_inputs["binary"])
            self.assertEqual(result2["status"], "success")

            # Clear memory
            self.alpha.clear_memory()

            # Final processing
            result3 = await self.alpha.process(self.test_inputs["mixed"])
            self.assertEqual(result3["status"], "success")

            # Stop system
            await self.alpha.stop()

        # Run lifecycle test
        self.loop.run_until_complete(lifecycle_test())
