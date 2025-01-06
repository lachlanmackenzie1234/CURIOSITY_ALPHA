"""Unit tests for the binary pattern mapping system."""

import unittest
import numpy as np

from core.patterns.binary_mapping import PatternMapper
from core.patterns.natural_patterns import NaturalPattern


class TestBinaryMapping(unittest.TestCase):
    """Test cases for binary pattern mapping."""

    def setUp(self):
        """Set up test fixtures."""
        self.mapper = PatternMapper()
        
        # Create test patterns
        self.fibonacci_pattern = NaturalPattern(
            principle_type="fibonacci",
            sequence=[1, 1, 2, 3, 5, 8],
            confidence=0.95
        )
        
        self.prime_pattern = NaturalPattern(
            principle_type="prime",
            sequence=[2, 3, 5, 7, 11],
            confidence=0.9
        )
        
        # Test context
        self.context = np.array([1, 1, 2, 3, 5, 8, 13])
    
    def test_pattern_to_binary(self):
        """Test mapping pattern to binary form."""
        # Map pattern to binary
        mapping = self.mapper.map_to_binary(
            self.fibonacci_pattern,
            self.context
        )
        
        # Verify mapping properties
        self.assertIsNotNone(mapping.binary_form)
        self.assertGreater(mapping.mapping_confidence, 0.0)
        self.assertEqual(mapping.pattern_type, "fibonacci")
    
    def test_binary_to_pattern(self):
        """Test mapping binary back to pattern."""
        # Create binary mapping
        original_mapping = self.mapper.map_to_binary(
            self.fibonacci_pattern,
            self.context
        )
        
        # Map back to pattern
        recovered_pattern = self.mapper.map_from_binary(
            original_mapping.binary_form,
            original_mapping.encoding_type
        )
        
        # Verify pattern recovery
        self.assertIsNotNone(recovered_pattern)
        self.assertEqual(
            recovered_pattern.principle_type,
            self.fibonacci_pattern.principle_type
        )
    
    def test_encoding_types(self):
        """Test different encoding types."""
        # Test each encoding type
        for encoding_type in self.mapper.ENCODING_TYPES:
            mapping = self.mapper.map_to_binary(
                self.fibonacci_pattern,
                self.context,
                encoding_type=encoding_type
            )
            
            # Verify encoding
            self.assertEqual(
                mapping.encoding_type,
                encoding_type
            )
            self.assertIsNotNone(mapping.binary_form)
    
    def test_pattern_preservation(self):
        """Test preservation of pattern properties."""
        # Map pattern to binary
        mapping = self.mapper.map_to_binary(
            self.fibonacci_pattern,
            self.context
        )
        
        # Map back to pattern
        recovered = self.mapper.map_from_binary(
            mapping.binary_form,
            mapping.encoding_type
        )
        
        # Verify properties preserved
        self.assertEqual(
            recovered.principle_type,
            self.fibonacci_pattern.principle_type
        )
        self.assertGreater(
            recovered.confidence,
            0.7
        )
    
    def test_compression_ratio(self):
        """Test binary mapping compression."""
        # Map pattern to binary
        mapping = self.mapper.map_to_binary(
            self.fibonacci_pattern,
            self.context
        )
        
        # Calculate compression ratio
        original_size = len(str(self.fibonacci_pattern.sequence))
        compressed_size = len(mapping.binary_form)
        ratio = compressed_size / original_size
        
        # Verify compression
        self.assertLess(ratio, 1.5)
    
    def test_pattern_markers(self):
        """Test pattern start/end markers."""
        # Map pattern to binary
        mapping = self.mapper.map_to_binary(
            self.fibonacci_pattern,
            self.context
        )
        
        # Verify markers
        self.assertTrue(
            mapping.binary_form.startswith(self.mapper.PATTERN_START)
        )
        self.assertTrue(
            mapping.binary_form.endswith(self.mapper.PATTERN_END)
        )
    
    def test_error_handling(self):
        """Test error handling in mapping process."""
        # Test with invalid pattern
        invalid_pattern = NaturalPattern(
            principle_type="invalid",
            sequence=[],
            confidence=0.0
        )
        
        # Attempt mapping
        mapping = self.mapper.map_to_binary(
            invalid_pattern,
            self.context
        )
        
        # Verify safe handling
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.mapping_confidence, 0.0)
    
    def test_context_influence(self):
        """Test influence of context on mapping."""
        # Map with different contexts
        context1 = np.array([1, 1, 2, 3, 5, 8])
        context2 = np.array([8, 5, 3, 2, 1, 1])
        
        mapping1 = self.mapper.map_to_binary(
            self.fibonacci_pattern,
            context1
        )
        mapping2 = self.mapper.map_to_binary(
            self.fibonacci_pattern,
            context2
        )
        
        # Verify context influence
        self.assertNotEqual(
            mapping1.binary_form,
            mapping2.binary_form
        )


if __name__ == '__main__':
    unittest.main() 