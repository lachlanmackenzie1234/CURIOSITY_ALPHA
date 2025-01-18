"""Smoke tests for ALPHA core components.

These tests verify basic functionality and imports of core components.
They serve as a quick sanity check that the system is properly installed
and core features are working.
"""

import array
import unittest

import numpy as np

from ALPHA.core.binary_foundation.base import Binary
from ALPHA.core.execution.engine import ExecutionEngine
from ALPHA.core.memory.monitor import MemoryMonitor
from ALPHA.core.patterns.neural_pattern import NeuralPattern
from ALPHA.core.patterns.pattern_evolution import PatternEvolution
from ALPHA.core.translation.translator import BinaryTranslator


class TestSmoke(unittest.TestCase):
    """Smoke tests for core ALPHA components."""

    def test_pattern_evolution(self):
        """Test basic pattern evolution functionality."""
        evolution = PatternEvolution()
        self.assertIsNotNone(evolution)

        # Basic pattern test
        test_pattern = array.array("B", [1, 1, 2, 3, 5, 8])
        metrics = evolution._calculate_pattern_metrics(
            test_pattern, {"expected_behavior": {"regularity": 0.8}}
        )

        self.assertIsInstance(metrics, dict)
        self.assertGreaterEqual(metrics.get("success_rate", 0), 0.0)

    def test_neural_pattern(self):
        """Test basic neural pattern functionality."""
        neural = NeuralPattern("test")
        self.assertIsNotNone(neural)

        # Basic analysis test
        result = neural.analyze_pattern(b"test data")
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.complexity, 0.0)
        self.assertLessEqual(result.complexity, 1.0)

    def test_memory_monitor(self):
        """Test basic memory monitoring functionality."""
        monitor = MemoryMonitor()
        self.assertIsNotNone(monitor)

        # Basic monitoring test
        monitor.start()
        stats = monitor.get_stats()
        monitor.stop()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_allocated", stats)
        self.assertGreaterEqual(stats["total_allocated"], 0)

    def test_execution_engine(self):
        """Test basic execution engine functionality."""
        engine = ExecutionEngine()
        self.assertIsNotNone(engine)

        # Basic execution test
        test_code = "x = 1 + 1"
        result = engine.validate_code(test_code)
        self.assertTrue(result)

    def test_binary_foundation(self):
        """Test basic binary foundation functionality."""
        binary = Binary()
        self.assertIsNotNone(binary)

        # Basic binary test
        test_data = b"test data"
        binary.append(test_data)
        result = binary.to_bytes()

        self.assertIsInstance(result, bytes)
        self.assertIn(test_data, result)

    def test_binary_translator(self):
        """Test basic binary translation functionality."""
        translator = BinaryTranslator()
        self.assertIsNotNone(translator)

        # Basic translation test
        source = "def test(): pass"
        binary = translator.to_binary(source)

        self.assertIsInstance(binary, bytes)
        self.assertGreater(len(binary), 0)

    def test_component_compatibility(self):
        """Test basic component compatibility."""
        # Create components
        evolution = PatternEvolution()
        neural = NeuralPattern("test")
        binary = Binary()
        translator = BinaryTranslator()

        # Test pattern sharing
        test_pattern = array.array("B", [1, 2, 3])

        # Evolution -> Neural
        metrics = evolution._calculate_pattern_metrics(
            test_pattern, {"expected_behavior": {"regularity": 0.8}}
        )
        signature = neural.analyze_pattern(test_pattern)

        self.assertIsNotNone(metrics)
        self.assertIsNotNone(signature)

        # Binary -> Translator
        binary.append(test_pattern)
        binary_data = binary.to_bytes()
        translated = translator.from_binary(binary_data)

        self.assertIsNotNone(translated)

    def test_numpy_integration(self):
        """Test NumPy integration with core components."""
        # Create test data
        test_array = np.array([1, 1, 2, 3, 5, 8], dtype=np.uint8)

        # Test with various components
        evolution = PatternEvolution()
        neural = NeuralPattern("test")
        binary = Binary()

        # Evolution metrics
        metrics = evolution._calculate_pattern_metrics(
            array.array("B", test_array.tobytes()),
            {"expected_behavior": {"regularity": 0.8}},
        )
        self.assertIsInstance(metrics, dict)

        # Neural analysis
        signature = neural.analyze_pattern(test_array.tobytes())
        self.assertIsNotNone(signature)

        # Binary conversion
        binary.append(test_array.tobytes())
        result = binary.to_bytes()
        self.assertIsInstance(result, bytes)

    def test_error_handling(self):
        """Test basic error handling in core components."""
        # Test invalid inputs
        evolution = PatternEvolution()
        neural = NeuralPattern("test")
        translator = BinaryTranslator()

        # Evolution with invalid pattern
        metrics = evolution._calculate_pattern_metrics(array.array("B", []), {})
        self.assertLessEqual(metrics.get("success_rate", 0), 0.1)

        # Neural with invalid input
        signature = neural.analyze_pattern(None)
        self.assertLessEqual(signature.complexity, 0.1)

        # Translator with invalid code
        with self.assertRaises(Exception):
            translator.to_binary("invalid python code {")
