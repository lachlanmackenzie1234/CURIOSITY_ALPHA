"""Integration tests for ALPHA core functionality."""

import asyncio
import time
import unittest

from ALPHA.core.execution.engine import ExecutionEngine
from ALPHA.core.interface import create_alpha
from ALPHA.core.memory.monitor import MemoryMonitor
from ALPHA.core.patterns.neural_pattern import NeuralPattern


class TestALPHA(unittest.TestCase):
    """Integration tests for ALPHA core functionality."""

    def setUp(self):
        """Set up test environment."""
        self.alpha = create_alpha("test_alpha")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Test data
        self.test_inputs = {
            "code": """
                def fibonacci(n):
                    if n <= 1:
                        return n
                    return fibonacci(n-1) + fibonacci(n-2)
            """,
            "pattern": bytes([1, 1, 2, 3, 5, 8, 13]),
            "mixed": {
                "data": bytes([0xAA, 0xBB, 0xCC]),
                "metadata": {"type": "test", "version": 1.0},
            },
        }

    def tearDown(self):
        """Clean up test environment."""
        self.loop.run_until_complete(self.alpha.stop())
        self.loop.close()

    def test_system_initialization(self):
        """Test complete system initialization."""
        # Verify core components
        self.assertIsInstance(
            self.alpha.neural_pattern,
            NeuralPattern,
            "Neural pattern system not initialized",
        )
        self.assertIsInstance(
            self.alpha.memory_monitor,
            MemoryMonitor,
            "Memory monitor not initialized",
        )
        self.assertIsInstance(
            self.alpha.execution_engine,
            ExecutionEngine,
            "Execution engine not initialized",
        )

        # Check component integration
        self.assertEqual(
            id(self.alpha.neural_pattern.monitor),
            id(self.alpha.memory_monitor),
            "Neural pattern not connected to memory monitor",
        )
        self.assertEqual(
            id(self.alpha.execution_engine.memory),
            id(self.alpha.memory_monitor.memory),
            "Execution engine not sharing memory space",
        )

    def test_end_to_end_processing(self):
        """Test end-to-end data processing flow."""

        async def process_all():
            results = []
            for input_type, input_data in self.test_inputs.items():
                # Process input
                result = await self.alpha.process(input_data)
                results.append((input_type, result))

                # Allow system to stabilize
                await asyncio.sleep(0.1)
            return results

        # Run processing
        results = self.loop.run_until_complete(process_all())

        # Verify each result
        for input_type, result in results:
            self.assertEqual(result["status"], "success")
            self.assertGreater(result["confidence"], 0.5)
            self.assertGreater(len(result["patterns"]), 0)

            # Check type-specific processing
            if input_type == "code":
                self.assertIn("ast_patterns", result)
            elif input_type == "pattern":
                self.assertIn("natural_patterns", result)

    def test_system_stability(self):
        """Test system stability under load."""

        async def generate_load():
            # Process multiple inputs concurrently
            tasks = []
            for _ in range(10):
                for input_data in self.test_inputs.values():
                    task = self.alpha.process(input_data)
                    tasks.append(task)

            # Run all tasks
            results = await asyncio.gather(*tasks)
            return results

        # Run load test
        start_time = time.time()
        results = self.loop.run_until_complete(generate_load())
        end_time = time.time()

        # Verify system stability
        self.assertTrue(all(r["status"] == "success" for r in results))
        self.assertLess(
            end_time - start_time,
            10.0,
            "System performance degraded under load",
        )

        # Check memory stability
        memory_stats = self.alpha.memory_monitor.get_stats()
        self.assertLess(
            memory_stats["memory_growth"],
            50.0,  # MB
            "Excessive memory growth under load",
        )

    def test_component_interaction(self):
        """Test interaction between system components."""

        async def test_interactions():
            # Process pattern to trigger component interaction
            result = await self.alpha.process(self.test_inputs["pattern"])

            # Get component states
            pattern_state = self.alpha.neural_pattern.get_state()
            memory_state = self.alpha.memory_monitor.get_state()
            engine_state = self.alpha.execution_engine.get_state()

            return result, pattern_state, memory_state, engine_state

        # Run interaction test
        result, p_state, m_state, e_state = self.loop.run_until_complete(test_interactions())

        # Verify component coordination
        self.assertEqual(
            p_state["active_patterns"],
            m_state["pattern_references"],
            "Pattern reference mismatch",
        )
        self.assertEqual(
            e_state["memory_allocated"],
            m_state["total_allocated"],
            "Memory tracking mismatch",
        )
        self.assertGreater(
            p_state["pattern_confidence"],
            0.0,
            "No pattern confidence recorded",
        )

    def test_error_recovery(self):
        """Test system error recovery capabilities."""

        async def trigger_errors():
            results = []

            # Test various error conditions
            error_cases = [
                None,  # Null input
                {},  # Empty dict
                b"\x00" * 1024 * 1024,  # Large binary
                {"invalid": object()},  # Non-serializable
                self.test_inputs["code"][:-1],  # Incomplete code
            ]

            for error_input in error_cases:
                try:
                    result = await self.alpha.process(error_input)
                    results.append(("success", result))
                except Exception as e:
                    results.append(("error", e))

                # Check system health
                health = await self.alpha.check_health()
                results.append(("health", health))

            return results

        # Run error tests
        results = self.loop.run_until_complete(trigger_errors())

        # Verify error handling
        for result_type, result in results:
            if result_type == "health":
                self.assertTrue(
                    result["system_healthy"],
                    "System health compromised after error",
                )
                self.assertGreater(result["recovery_rate"], 0.8, "Poor error recovery rate")

    def test_learning_persistence(self):
        """Test learning persistence across sessions."""

        async def test_persistence():
            # Initial learning
            await self.alpha.process(self.test_inputs["pattern"])
            initial_patterns = self.alpha.neural_pattern.get_learned_patterns()

            # Restart system
            await self.alpha.stop()
            self.alpha = create_alpha("test_alpha")
            await self.alpha.start()

            # Process similar input
            await self.alpha.process(self.test_inputs["pattern"])
            final_patterns = self.alpha.neural_pattern.get_learned_patterns()

            return initial_patterns, final_patterns

        # Test persistence
        initial, final = self.loop.run_until_complete(test_persistence())

        # Verify learning persisted
        self.assertGreaterEqual(len(final), len(initial), "Learning not persisted across sessions")
        self.assertTrue(
            any(p in final for p in initial),
            "Previous patterns not recognized",
        )
