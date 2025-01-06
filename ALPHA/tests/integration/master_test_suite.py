"""Master test suite for ALPHA system."""

import os
import sys
import unittest
from typing import List, Tuple

# Add project root to Python path before imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

try:
    # Core test imports
    from ALPHA.core.binary_foundation.base import Binary
    from ALPHA.core.binary_foundation.optimizer import OptimizationPattern
    from ALPHA.core.memory.space import MemoryOrganizer
    
    # Test suite imports
    from ALPHA.tests.test_neural_pattern import TestNeuralPattern
    from ALPHA.tests.test_pattern_evolution import TestPatternEvolution
    from ALPHA.tests.test_simple import TestSimple
    from ALPHA.tests.test_runner import (
        run_basic_tests,
        run_performance_tests,
        run_stress_tests,
        TestResult
    )
except ImportError as e:
    print(f"Import Error: {str(e)}")
    print(f"Python Path: {sys.path}")
    sys.exit(1)


def get_test_loader() -> unittest.TestLoader:
    """Get configured test loader."""
    return unittest.TestLoader()


def load_test_case(
    loader: unittest.TestLoader,
    test_case: type
) -> unittest.TestSuite:
    """Load test case with proper error handling."""
    try:
        return loader.loadTestsFromTestCase(test_case)
    except Exception as e:
        print(f"Error loading test case {test_case.__name__}: {str(e)}")
        return unittest.TestSuite()


class TestBinaryFoundation(unittest.TestCase):
    """Test binary foundation components."""
    
    def setUp(self):
        """Set up test environment."""
        self.binary = Binary()
        self.optimizer = OptimizationPattern("test_optimizer")
    
    def test_binary_operations(self):
        """Test basic binary operations."""
        test_data = b"Hello, World!"
        
        # Test data handling
        self.binary.from_bytes(test_data)
        self.assertEqual(self.binary.to_bytes(), test_data)
        
        # Test segmentation
        segment = self.binary.get_segment(0, 5)
        self.assertEqual(segment, b"Hello")
        
        # Test metadata
        self.binary.set_metadata("test_key", "test_value")
        self.assertEqual(
            self.binary.get_metadata("test_key"),
            "test_value"
        )
    
    def test_optimization_pattern(self):
        """Test optimization pattern functionality."""
        # Create test data
        original = Binary(b"test data")
        optimized = Binary(b"optimized")
        
        # Test learning
        self.optimizer.learn(original, optimized)
        self.assertGreater(self.optimizer.pattern.get_size(), 0)
        
        # Test confidence calculation
        confidence = self.optimizer.calculate_confidence(original)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test pattern application
        result = self.optimizer.apply(original)
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, Binary))


class TestMemorySystem(unittest.TestCase):
    """Test memory management system."""
    
    def setUp(self):
        """Set up test environment."""
        self.memory = MemoryOrganizer()
    
    def test_memory_allocation(self):
        """Test memory allocation and management."""
        test_data = b"test data"
        reference = "test_ref"
        
        # Test allocation
        success = self.memory.allocate(test_data, reference)
        self.assertTrue(success)
        
        # Test reading
        stored_data = self.memory.read(reference)
        self.assertEqual(len(stored_data), 1)
        self.assertEqual(stored_data[0], test_data)
        
        # Test deallocation
        self.memory.deallocate(reference)
        stored_data = self.memory.read(reference)
        self.assertEqual(len(stored_data), 0)
    
    def test_memory_defragmentation(self):
        """Test memory defragmentation."""
        # Fill memory with data
        for i in range(5):
            self.memory.allocate(
                f"data_{i}".encode(),
                f"ref_{i}"
            )
        
        # Deallocate some blocks
        self.memory.deallocate("ref_1")
        self.memory.deallocate("ref_3")
        
        # Defragment
        self.memory.defragment()
        
        # Verify remaining data
        data_2 = self.memory.read("ref_2")
        self.assertEqual(data_2[0], b"data_2")
        data_4 = self.memory.read("ref_4")
        self.assertEqual(data_4[0], b"data_4")


def run_unit_tests() -> Tuple[int, int]:
    """Run all unit tests and return (passed, failed) counts."""
    loader = get_test_loader()
    suite = unittest.TestSuite()
    
    # Test cases to run
    test_cases = [
        TestNeuralPattern,
        TestPatternEvolution,
        TestSimple,
        TestBinaryFoundation,
        TestMemorySystem
    ]
    
    # Load test cases
    for test_case in test_cases:
        suite.addTests(load_test_case(loader, test_case))
    
    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return len(result.successes), len(result.failures + result.errors)


def run_integration_tests() -> Tuple[TestResult, TestResult, TestResult]:
    """Run integration test suites."""
    print("\nRunning integration test suites...")
    return (
        run_basic_tests(),
        run_performance_tests(),
        run_stress_tests()
    )


def print_test_summary(
    unit_passed: int,
    unit_failed: int,
    integration_results: List[TestResult]
) -> None:
    """Print test execution summary."""
    total_passed = unit_passed
    total_failed = unit_failed
    
    for result in integration_results:
        total_passed += result.passed
        total_failed += result.failed
    
    print("\nTest Summary:")
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed > 0:
        print("\nErrors:")
        for result in integration_results:
            for error in result.errors:
                print(f"- {error}")


def main() -> bool:
    """Run all tests and return success status."""
    # Run unit tests
    unit_passed, unit_failed = run_unit_tests()
    
    # Run integration tests
    basic, perf, stress = run_integration_tests()
    
    # Print results
    print_test_summary(
        unit_passed,
        unit_failed,
        [basic, perf, stress]
    )
    
    # Return True if all tests passed
    return (
        unit_failed == 0 and
        basic.failed == 0 and
        perf.failed == 0 and
        stress.failed == 0
    )


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 