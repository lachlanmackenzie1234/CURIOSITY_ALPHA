"""Test runner for ALPHA system verification.

This module provides a comprehensive test runner that executes unit tests,
integration tests, and performance tests for the ALPHA system. It includes
detailed reporting and metrics collection.
"""

import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Core imports
from ALPHA.core.interface import ALPHACore
from ALPHA.core.monitor import Monitor

# Test imports - split into multiple lines for readability
from ALPHA.tests.unit import (
    test_binary_core,
    test_pattern_evolution,
    test_neural_pattern,
    test_smoke
)
from ALPHA.tests.integration import (
    test_alpha,
    test_interface,
    test_memory_monitor
)


@dataclass
class TestMetrics:
    """Metrics collected during test execution."""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    pattern_count: int = 0
    success_rate: float = 0.0
    error_rate: float = 0.0


@dataclass
class TestResult:
    """Results of a test run."""
    name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)
    metrics: TestMetrics = field(default_factory=TestMetrics)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def duration(self) -> float:
        """Get test duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def success_rate(self) -> float:
        """Calculate test success rate."""
        total = self.passed + self.failed
        return self.passed / total if total > 0 else 0.0


class TestRunner:
    """ALPHA system test runner."""

    def __init__(self):
        """Initialize test runner."""
        self.alpha = ALPHACore("test_runner")
        self.monitor = Monitor()
        self.results: Dict[str, TestResult] = {}
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("ALPHA.TestRunner")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    async def run_unit_tests(self) -> TestResult:
        """Run unit tests."""
        result = TestResult("Unit Tests")
        self.logger.info("Running unit tests...")

        try:
            # Import and run unit tests
            import unittest

            # Create test suite
            suite = unittest.TestSuite()
            suite.addTests(
                unittest.defaultTestLoader.loadTestsFromModule(
                    test_binary_core
                )
            )
            suite.addTests(
                unittest.defaultTestLoader.loadTestsFromModule(
                    test_pattern_evolution
                )
            )
            suite.addTests(
                unittest.defaultTestLoader.loadTestsFromModule(
                    test_neural_pattern
                )
            )
            suite.addTests(
                unittest.defaultTestLoader.loadTestsFromModule(
                    test_smoke
                )
            )

            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            test_result = await asyncio.to_thread(runner.run, suite)

            # Update results
            result.passed = test_result.wasSuccessful()
            result.failed = len(test_result.failures)
            result.errors.extend(str(e) for e in test_result.errors)

        except Exception as e:
            result.failed += 1
            result.errors.append(f"Error running unit tests: {str(e)}")
            self.logger.error("Unit test execution failed", exc_info=True)

        result.end_time = time.time()
        return result

    async def run_integration_tests(self) -> TestResult:
        """Run integration tests."""
        result = TestResult("Integration Tests")
        self.logger.info("Running integration tests...")

        try:
            # Import and run integration tests
            import unittest

            # Create test suite
            suite = unittest.TestSuite()
            suite.addTests(
                unittest.defaultTestLoader.loadTestsFromModule(test_alpha)
            )
            suite.addTests(
                unittest.defaultTestLoader.loadTestsFromModule(test_interface)
            )
            suite.addTests(
                unittest.defaultTestLoader.loadTestsFromModule(
                    test_memory_monitor
                )
            )

            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            test_result = await asyncio.to_thread(runner.run, suite)

            # Update results
            result.passed = test_result.wasSuccessful()
            result.failed = len(test_result.failures)
            result.errors.extend(str(e) for e in test_result.errors)

        except Exception as e:
            result.failed += 1
            msg = "Error running integration tests: {}"
            result.errors.append(msg.format(str(e)))
            self.logger.error(
                "Integration test execution failed",
                exc_info=True
            )

        result.end_time = time.time()
        return result

    async def run_performance_tests(self) -> TestResult:
        """Run performance tests."""
        result = TestResult("Performance Tests")
        self.logger.info("Running performance tests...")

        try:
            # Start monitoring
            self.monitor.start()

            # Test data
            test_inputs = [
                "def test(): return 42",
                bytes([i % 256 for i in range(1024)]),
                [1, 1, 2, 3, 5, 8, 13, 21],
                {"type": "test", "data": b"test data"}
            ]

            # Process test inputs
            start_time = time.time()
            tasks = [
                self.alpha.process(input_data)
                for input_data in test_inputs * 10  # 10x for stress testing
            ]
            results = await asyncio.gather(*tasks)

            # Collect metrics
            metrics = self.monitor.get_metrics()
            result.metrics.execution_time = time.time() - start_time
            result.metrics.memory_usage = metrics['memory_usage']
            result.metrics.pattern_count = metrics['pattern_count']
            success_count = sum(
                1 for r in results if r['status'] == 'success'
            )
            error_count = sum(
                1 for r in results if r['status'] == 'error'
            )
            result.metrics.success_rate = success_count / len(results)
            result.metrics.error_rate = error_count / len(results)

            # Verify performance constraints
            if result.metrics.execution_time > 10.0:
                result.failed += 1
                result.errors.append("Performance test exceeded time limit")
            if result.metrics.memory_usage > 1024:  # MB
                result.failed += 1
                result.errors.append("Performance test exceeded memory limit")
            if result.metrics.success_rate < 0.9:
                result.failed += 1
                result.errors.append("Performance test success rate too low")

            result.passed = (
                result.metrics.execution_time <= 10.0
                and result.metrics.memory_usage <= 1024
                and result.metrics.success_rate >= 0.9
            )

        except Exception as exc:
            result.failed += 1
            msg = "Error running performance tests: {}"
            result.errors.append(msg.format(str(exc)))
            self.logger.error(
                "Performance test execution failed",
                exc_info=True
            )

        finally:
            self.monitor.stop()

        result.end_time = time.time()
        return result

    def generate_report(self) -> str:
        """Generate test execution report."""
        report = [
            "\nALPHA System Test Report",
            "=" * 50,
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        ]

        total_passed = 0
        total_failed = 0
        total_time = 0.0

        for test_name, result in self.results.items():
            report.extend([
                f"\n{test_name}:",
                "-" * len(test_name),
                f"Passed: {result.passed}",
                f"Failed: {result.failed}",
                f"Skipped: {result.skipped}",
                f"Duration: {result.duration():.2f}s",
                f"Success Rate: {result.success_rate():.1%}\n"
            ])

            if result.errors:
                report.extend([
                    "Errors:",
                    *[f"- {error}" for error in result.errors],
                    ""
                ])

            if hasattr(result, 'metrics'):
                report.extend([
                    "Metrics:",
                    f"- Execution Time: {result.metrics.execution_time:.2f}s",
                    f"- Memory Usage: {result.metrics.memory_usage:.1f}MB",
                    f"- Pattern Count: {result.metrics.pattern_count}",
                    f"- Success Rate: {result.metrics.success_rate:.1%}",
                    f"- Error Rate: {result.metrics.error_rate:.1%}\n"
                ])

            total_passed += result.passed
            total_failed += result.failed
            total_time += result.duration()

        total_tests = total_passed + total_failed
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        report.extend([
            "Summary:",
            "=" * 20,
            f"Total Tests: {total_tests}",
            f"Total Passed: {total_passed}",
            f"Total Failed: {total_failed}",
            f"Overall Success Rate: {success_rate:.1%}",
            f"Total Duration: {total_time:.2f}s\n"
        ])

        return "\n".join(report)

    async def run_all(self):
        """Run all tests."""
        try:
            # Run tests
            self.results['unit'] = await self.run_unit_tests()
            self.results['integration'] = await self.run_integration_tests()
            self.results['performance'] = await self.run_performance_tests()

            # Generate and print report
            report = self.generate_report()
            print(report)

            # Save report to file
            report_path = Path("test_report.txt")
            report_path.write_text(report)
            self.logger.info(f"Test report saved to {report_path}")

            # Return overall success
            return all(
                result.failed == 0
                for result in self.results.values()
            )

        except Exception as exc:
            msg = f"Test execution failed: {str(exc)}"
            self.logger.error(msg, exc_info=True)
            return False


async def main():
    """Run test suite."""
    runner = TestRunner()
    success = await runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main()) 