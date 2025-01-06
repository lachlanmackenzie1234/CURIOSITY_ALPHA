"""Unit tests for the pattern-first translation system."""

import unittest
import numpy as np
from typing import Dict

from core.translation.pattern_first import (
    PatternFirstTranslator,
    TranslationUnit
)
from core.patterns.natural_patterns import NaturalPattern


class TestPatternFirstTranslator(unittest.TestCase):
    """Test cases for PatternFirstTranslator."""

    def setUp(self):
        """Set up test fixtures."""
        self.translator = PatternFirstTranslator()
        
        # Test data with known patterns
        self.fibonacci_sequence = "1,1,2,3,5,8,13,21,34,55"
        self.prime_sequence = "2,3,5,7,11,13,17,19,23,29"
        self.random_data = "Hello, World! 123"
    
    def test_translation_unit_creation(self):
        """Test creation of TranslationUnit objects."""
        content = b"test content"
        patterns: Dict[str, NaturalPattern] = {}
        mappings = {}
        
        unit = TranslationUnit(
            content=content,
            patterns=patterns,
            mappings=mappings
        )
        
        self.assertEqual(unit.content, content)
        self.assertEqual(unit.patterns, patterns)
        self.assertEqual(unit.mappings, mappings)
        self.assertEqual(unit.resonance_score, 0.0)
        self.assertEqual(unit.preservation_score, 0.0)
    
    def test_identify_patterns_fibonacci(self):
        """Test pattern identification in Fibonacci sequence."""
        # Convert to numpy array
        data = np.array([ord(c) for c in self.fibonacci_sequence])
        
        # Identify patterns
        units = self.translator._identify_patterns(data)
        
        # Verify patterns were found
        self.assertTrue(len(units) > 0)
        has_patterns = any(len(unit.patterns) > 0 for unit in units)
        self.assertTrue(has_patterns)
    
    def test_identify_patterns_primes(self):
        """Test pattern identification in prime numbers."""
        # Convert to numpy array
        data = np.array([ord(c) for c in self.prime_sequence])
        
        # Identify patterns
        units = self.translator._identify_patterns(data)
        
        # Verify patterns were found
        self.assertTrue(len(units) > 0)
        has_patterns = any(len(unit.patterns) > 0 for unit in units)
        self.assertTrue(has_patterns)
    
    def test_binary_translation_roundtrip(self):
        """Test full translation roundtrip with pattern preservation."""
        # Translate to binary
        binary = self.translator.translate_to_binary(
            self.fibonacci_sequence,
            preserve_patterns=True
        )
        
        # Translate back
        result = self.translator.translate_from_binary(
            binary,
            preserve_patterns=True
        )
        
        # Verify roundtrip
        self.assertIsNotNone(result)
        self.assertEqual(result, self.fibonacci_sequence)
    
    def test_pattern_preservation(self):
        """Test that patterns are preserved during translation."""
        # Translate with pattern preservation
        binary_preserved = self.translator.translate_to_binary(
            self.fibonacci_sequence,
            preserve_patterns=True
        )
        
        # Translate without pattern preservation
        binary_direct = self.translator.translate_to_binary(
            self.fibonacci_sequence,
            preserve_patterns=False
        )
        
        # Verify different encodings
        self.assertNotEqual(binary_preserved, binary_direct)
    
    def test_unit_merging(self):
        """Test merging of related translation units."""
        # Create test units
        unit1 = TranslationUnit(
            content=b"test1",
            patterns={'p1': NaturalPattern(
                principle_type="fibonacci",
                sequence=[1, 1, 2, 3],
                confidence=0.9
            )},
            mappings={},
            resonance_score=0.8
        )
        
        unit2 = TranslationUnit(
            content=b"test2",
            patterns={'p2': NaturalPattern(
                principle_type="fibonacci",
                sequence=[5, 8, 13],
                confidence=0.85
            )},
            mappings={},
            resonance_score=0.7
        )
        
        # Merge units
        merged = self.translator._merge_units([unit1, unit2])
        
        # Verify merge results
        self.assertEqual(merged.content, b"test1test2")
        self.assertEqual(len(merged.patterns), 2)
        self.assertEqual(merged.resonance_score, 0.8)
    
    def test_error_handling(self):
        """Test error handling in translation process."""
        # Test with invalid input
        binary = self.translator.translate_to_binary(None)
        self.assertEqual(binary, b'None')
        
        # Test with corrupted binary
        result = self.translator.translate_from_binary(b'corrupted')
        self.assertIsNone(result)
    
    def test_pattern_recovery(self):
        """Test pattern recovery from binary data."""
        # Create binary with known pattern
        binary = self.translator.translate_to_binary(
            self.fibonacci_sequence,
            preserve_patterns=True
        )
        
        # Extract units
        units = self.translator._extract_units(binary)
        
        # Recover patterns
        recovered = self.translator._recover_patterns(units)
        
        # Verify pattern recovery
        has_patterns = any(len(unit.patterns) > 0 for unit in recovered)
        self.assertTrue(has_patterns)


if __name__ == '__main__':
    unittest.main() 