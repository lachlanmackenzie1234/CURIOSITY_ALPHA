"""Tests for memory monitoring system."""

import unittest

from ALPHA.core.memory.monitor import MemoryMonitor


class TestMemoryMonitor(unittest.TestCase):
    """Test cases for memory monitoring system."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.monitor = MemoryMonitor()

    def tearDown(self) -> None:
        """Clean up test environment."""
        if hasattr(self, "monitor"):
            self.monitor.stop()

    def test_basic_monitoring(self) -> None:
        """Test basic memory monitoring functionality."""
        # Start monitoring
        self.monitor.start()

        # Update metrics
        self.monitor.update_metrics(
            cpu_usage=50.0, memory_usage=60.0, pattern_rate=100.0, error_rate=0.01
        )

        # Get metrics
        status = self.monitor.get_system_status()

        # Verify metrics
        self.assertGreaterEqual(status["uptime"], 0)
        self.assertEqual(status["cpu_usage"], 50.0)
        self.assertEqual(status["memory_usage"], 60.0)
        self.assertEqual(status["pattern_processing_rate"], 100.0)
        self.assertEqual(status["error_rate"], 0.01)
        self.assertGreaterEqual(status["stability_score"], 0.0)
        self.assertLessEqual(status["stability_score"], 1.0)
        self.assertGreaterEqual(status["health_score"], 0.0)
        self.assertLessEqual(status["health_score"], 1.0)

    def test_system_health(self) -> None:
        """Test system health monitoring."""
        self.monitor.start()

        # Test good health
        self.monitor.update_metrics(
            cpu_usage=20.0, memory_usage=30.0, pattern_rate=100.0, error_rate=0.01
        )
        status = self.monitor.get_system_status()
        self.assertGreater(status["health_score"], 0.7)

        # Test degraded health
        self.monitor.update_metrics(
            cpu_usage=90.0, memory_usage=95.0, pattern_rate=50.0, error_rate=0.5
        )
        status = self.monitor.get_system_status()
        self.assertLess(status["health_score"], 0.5)

    def test_stability_calculation(self) -> None:
        """Test stability score calculation."""
        self.monitor.start()

        # Update with stable metrics
        for _ in range(5):
            self.monitor.update_metrics(
                cpu_usage=50.0, memory_usage=50.0, pattern_rate=100.0, error_rate=0.01
            )

        status = self.monitor.get_system_status()
        self.assertGreater(status["stability_score"], 0.8)

        # Update with unstable metrics
        for i in range(5):
            self.monitor.update_metrics(
                cpu_usage=float(i * 20),
                memory_usage=float(i * 15),
                pattern_rate=float(100 - i * 10),
                error_rate=float(i) / 10,
            )

        status = self.monitor.get_system_status()
        self.assertLess(status["stability_score"], 0.8)

    def test_error_handling(self) -> None:
        """Test error handling in monitoring system."""
        self.monitor.start()

        # Test high error rate
        self.monitor.update_metrics(
            cpu_usage=50.0, memory_usage=50.0, pattern_rate=100.0, error_rate=0.5  # High error rate
        )

        status = self.monitor.get_system_status()
        self.assertLess(status["health_score"], 0.7)

    def test_monitor_lifecycle(self) -> None:
        """Test monitor start/stop functionality."""
        # Test start
        self.assertFalse(self.monitor.is_active())
        self.monitor.start()
        self.assertTrue(self.monitor.is_active())

        # Test stop
        self.monitor.stop()
        self.assertFalse(self.monitor.is_active())
