"""Tests for memory system."""

import logging
import os
import tempfile
import unittest

from ALPHA.core.memory.space import MemoryOrganizer

logging.basicConfig(level=logging.DEBUG)


class TestMemorySystem(unittest.TestCase):
    """Test cases for memory system."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence_path = os.path.join(self.temp_dir, "test_patterns.json")
        self.memory = MemoryOrganizer(
            initial_block_size=128, persistence_path=self.persistence_path
        )
        self.test_data = b"test pattern data"
        self.test_ref = "test_pattern_1"

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.persistence_path):
            os.remove(self.persistence_path)
        os.rmdir(self.temp_dir)

    def test_basic_operations(self):
        """Test basic memory operations."""
        # Test allocation
        success = self.memory.allocate(self.test_data, self.test_ref, importance=0.8)
        self.assertTrue(success)

        # Test block creation
        self.assertEqual(len(self.memory.blocks), 1)
        self.assertIn(self.test_ref, self.memory.reference_map)

        # Test metrics
        block = self.memory.blocks[0]
        metrics = block.get_metrics(self.test_ref)
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.importance_score, 0.8)
        self.assertGreater(metrics.access_count, 0)

    def test_pattern_connections(self):
        """Test pattern relationship functionality."""
        # Create two patterns
        ref1 = "pattern1"
        ref2 = "pattern2"
        data1 = b"pattern data 1"
        data2 = b"pattern data 2"

        self.memory.allocate(data1, ref1)
        self.memory.allocate(data2, ref2)

        # Connect patterns
        success = self.memory.connect_patterns(ref1, ref2)
        self.assertTrue(success)

        # Verify connection
        connected = self.memory.get_connected_patterns(ref1)
        self.assertIn(ref2, connected)

    def test_persistence(self):
        """Test pattern persistence functionality."""
        # Create and store pattern
        self.memory.allocate(self.test_data, self.test_ref, importance=0.9)

        # Connect to another pattern
        other_ref = "other_pattern"
        self.memory.allocate(b"other data", other_ref)
        self.memory.connect_patterns(self.test_ref, other_ref)

        # Force persistence
        self.memory._persist_patterns()

        # Create new memory organizer and verify pattern restoration
        new_memory = MemoryOrganizer(persistence_path=self.persistence_path)

        # Verify pattern data
        block_indices = new_memory.reference_map.get(self.test_ref, [])
        self.assertTrue(block_indices)

        block = new_memory.blocks[block_indices[0]]
        data = block.read(0, len(self.test_data), self.test_ref)
        self.assertEqual(data, self.test_data)

        # Verify connections
        connected = new_memory.get_connected_patterns(self.test_ref)
        self.assertIn(other_ref, connected)

    def test_memory_maintenance(self):
        """Test memory maintenance operations."""
        # Create some patterns with varying importance
        patterns = [
            ("high_imp", b"high", 0.9),
            ("med_imp", b"medium", 0.5),
            ("low_imp", b"low", 0.1),
        ]

        # Allocate patterns
        for ref, data, imp in patterns:
            self.memory.allocate(data, ref, importance=imp)

        # Get block references
        high_block = None
        high_indices = self.memory.reference_map.get("high_imp", [])
        if high_indices:
            high_block = self.memory.blocks[high_indices[0]]

        # Simulate more access to high importance patterns
        if high_block:
            for _ in range(5):
                high_block.read(0, len(b"high"), "high_imp")

        # Simulate significant time passage (15 minutes)
        self.memory.last_maintenance_time -= 900

        # Force immediate maintenance
        self.memory._perform_maintenance()

        # Verify low importance pattern was pruned (effective importance < 0.2)
        block_indices = self.memory.reference_map.get("low_imp", [])
        msg = "Low importance pattern should be pruned"
        self.assertFalse(block_indices, msg)

        # High importance pattern should remain (effective importance > 0.2)
        block_indices = self.memory.reference_map.get("high_imp", [])
        msg = "High importance pattern should remain"
        self.assertTrue(block_indices, msg)

        # Medium importance pattern state depends on access history
        med_indices = self.memory.reference_map.get("med_imp", [])
        if med_indices:
            block = self.memory.blocks[med_indices[0]]
            metrics = block.get_metrics("med_imp")
            self.assertIsNotNone(metrics)
            # Just verify metrics exist, as the exact state depends on timing

    def test_fragmentation_handling(self):
        """Test memory defragmentation."""
        # Create patterns of varying sizes to cause fragmentation
        patterns = [
            (f"pattern_{i}", os.urandom(32 * (i + 1))) for i in range(5)  # Increasing sizes
        ]

        # Allocate patterns
        for ref, data in patterns:
            self.memory.allocate(data, ref)

        # Delete patterns in a way that creates gaps
        self.memory._deallocate_pattern("pattern_1")
        self.memory._deallocate_pattern("pattern_3")

        # Add a pattern that won't fit in any single gap
        large_data = os.urandom(128)
        self.memory.allocate(large_data, "large_pattern")

        # Store block count before defragmentation
        initial_blocks = len(self.memory.blocks)

        # Force defragmentation
        self.memory.defragment()

        # Verify block consolidation
        self.assertLess(len(self.memory.blocks), initial_blocks)

        # Verify all patterns are still accessible
        self.assertTrue(self.memory.reference_map.get("pattern_0"))
        self.assertTrue(self.memory.reference_map.get("pattern_2"))
        self.assertTrue(self.memory.reference_map.get("pattern_4"))
        self.assertTrue(self.memory.reference_map.get("large_pattern"))


if __name__ == "__main__":
    unittest.main()
