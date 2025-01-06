"""Tests for memory monitoring system."""

import unittest
import time
from dataclasses import dataclass
from ALPHA.core.memory.monitor import MemoryMonitor, MemoryConfig


@dataclass
class TestConfig(MemoryConfig):
    """Test configuration with lower thresholds."""
    threshold_mb: float = 100
    check_interval: float = 1
    cache_size_limit: int = 1000
    pattern_age_weight: float = 0.3
    pattern_size_weight: float = 0.3
    pattern_usage_weight: float = 0.4


class TestMemoryMonitor(unittest.TestCase):
    """Test cases for memory monitoring system."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = TestConfig()
        self.monitor = MemoryMonitor(config=self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
        
    def test_basic_monitoring(self):
        """Test basic memory monitoring functionality."""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Track some allocations
        obj_id1 = hash("pattern1")
        obj_id2 = hash("pattern2")
        self.monitor.track_allocation(obj_id1, 1000, "pattern1")
        self.monitor.track_allocation(obj_id2, 2000, "pattern2")
        
        # Get metrics
        metrics = self.monitor.get_metrics()
        
        # Verify metrics
        self.assertEqual(metrics.total_allocated, 3000)
        self.assertEqual(metrics.pattern_count, 2)
        self.assertEqual(len(self.monitor.get_pattern_references()), 2)
        
        # Track deallocation
        self.monitor.track_deallocation(obj_id1, "pattern1")
        
        # Verify updated metrics
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics.total_freed, 1000)
        self.assertEqual(metrics.pattern_count, 1)
        self.assertEqual(len(self.monitor.get_pattern_references()), 1)
        
    def test_memory_alerts(self):
        """Test memory alert generation."""
        # Set very low threshold
        low_threshold_config = TestConfig(threshold_mb=0.1, check_interval=0.1)
        self.monitor = MemoryMonitor(config=low_threshold_config)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Simulate high memory usage with multiple allocations
        for i in range(10):
            obj_id = hash(f"large_pattern_{i}")
            size = 1024 * 1024  # 1MB each
            self.monitor.track_allocation(obj_id, size, f"large_pattern_{i}")
        
        # Wait for check interval
        time.sleep(0.2)  # Slightly longer than check interval
        
        # Force metrics update
        alerts = self.monitor.check_alerts()
        
        # Verify alerts
        self.assertGreater(len(alerts), 0)
        self.assertTrue(
            any("memory usage" in alert.lower() for alert in alerts)
        )
        
    def test_pattern_caching(self):
        """Test pattern caching functionality."""
        pattern_key = "test_pattern"
        pattern_data = b"test_data"
        
        # Test caching
        self.monitor.cache_pattern(pattern_key, pattern_data)
        self.assertTrue(self.monitor.is_pattern_cached(pattern_key))
        
        # Test retrieval
        cached_data = self.monitor.get_cached_pattern(pattern_key)
        self.assertEqual(cached_data, pattern_data)
        
        # Test cache miss
        missing_data = self.monitor.get_cached_pattern("nonexistent")
        self.assertIsNone(missing_data)
        
    def test_reference_counting(self):
        """Test pattern reference counting."""
        # Track multiple references
        obj_id1 = hash("shared_pattern")
        obj_id2 = hash("shared_pattern_2")
        self.monitor.track_allocation(obj_id1, 100, "shared_pattern")
        self.monitor.track_allocation(obj_id2, 100, "shared_pattern")
        
        # Verify reference count
        refs = self.monitor.get_pattern_references()
        self.assertEqual(refs["shared_pattern"], 2)
        
        # Remove one reference
        self.monitor.track_deallocation(obj_id1, "shared_pattern")
        
        # Verify updated count
        refs = self.monitor.get_pattern_references()
        self.assertEqual(refs["shared_pattern"], 1)
        
        # Remove last reference
        self.monitor.track_deallocation(obj_id2, "shared_pattern")
        
        # Verify pattern is removed
        refs = self.monitor.get_pattern_references()
        self.assertNotIn("shared_pattern", refs)
        
    def test_cache_invalidation(self):
        """Test cache invalidation under pressure."""
        # Add multiple patterns
        for i in range(10):
            pattern_key = f"pattern_{i}"
            pattern_data = b"data" * i
            self.monitor.cache_pattern(pattern_key, pattern_data)
            
        # Force cache invalidation
        self.monitor._invalidate_cache()
        
        # Verify some patterns were removed
        self.assertLess(len(self.monitor._pattern_cache), 10)
        
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        # Add an allocation and wait
        obj_id = hash("test_pattern")
        self.monitor.track_allocation(obj_id, 1024, "test_pattern")
        time.sleep(2)  # Wait longer than check_interval * 2
        
        # Check for leaks
        alerts = self.monitor.check_alerts()
        self.assertTrue(
            any("memory leak" in alert.lower() for alert in alerts)
        )
        
    def test_pattern_score_calculation(self):
        """Test pattern scoring mechanism."""
        # Test score for non-existent pattern
        score1 = self.monitor._get_pattern_score("nonexistent")
        self.assertEqual(score1, 0.0)
        
        # Test score for existing pattern
        pattern_key = "test"
        pattern_data = b"data"
        self.monitor.cache_pattern(pattern_key, pattern_data)
        score2 = self.monitor._get_pattern_score(pattern_key)
        self.assertGreater(score2, 0.0) 