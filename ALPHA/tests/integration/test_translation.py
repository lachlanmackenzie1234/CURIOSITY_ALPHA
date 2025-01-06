"""Tests for binary translation system."""

import unittest
import array
import numpy as np
from ALPHA.core.translation.translator import BinaryTranslator
from ALPHA.core.binary_foundation.base import Binary
import ast


class TestTranslation(unittest.TestCase):
    """Test cases for binary translation system."""

    def setUp(self):
        """Set up test environment."""
        self.translator = BinaryTranslator()
        self.test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        self.complex_test_code = """
class PatternGenerator:
    def __init__(self):
        self.sequence = []
    
    def generate_fibonacci(self, n):
        if n <= 1:
            return n
        return self.generate_fibonacci(n-1) + self.generate_fibonacci(n-2)
    
    def generate_exponential(self, base, n):
        return [base ** i for i in range(n)]
"""

    def test_basic_translation(self):
        """Test basic code translation."""
        binary = self.translator.translate_to_binary(self.test_code)
        self.assertIsInstance(binary, Binary)
        self.assertTrue(binary.to_bytes())
        
        # Check translation metrics
        self.assertIn('translation_translation_confidence', binary.metadata)
        confidence = float(binary.metadata['translation_translation_confidence'])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_pattern_preservation(self):
        """Test pattern preservation during translation."""
        # Create initial binary with patterns
        initial = Binary()
        test_data = array.array('B', [1, 1, 2, 3, 5, 8, 13])  # Fibonacci
        initial.append(bytes(test_data))
        
        # Set as current and translate
        self.translator.set_binary(initial)
        binary = self.translator.translate_to_binary(self.test_code)
        
        # Check pattern preservation metrics
        self.assertIn('translation_patterns_preserved', binary.metadata)
        preserved = int(binary.metadata['translation_patterns_preserved'])
        self.assertGreater(preserved, 0)

    def test_error_recovery(self):
        """Test error recovery in translation."""
        # Test with invalid Python code
        invalid_code = "def invalid_syntax("
        binary = self.translator.translate_to_binary(invalid_code)
        
        # Should have error metadata but still produce binary
        self.assertIn('syntax_error', binary.metadata)
        self.assertTrue(binary.to_bytes())

    def test_pattern_detection(self):
        """Test pattern detection capabilities."""
        # Test various pattern types
        patterns = {
            'fibonacci': array.array('B', [1, 1, 2, 3, 5, 8, 13]),
            'exponential': array.array('B', [1, 2, 4, 8, 16, 32, 64]),
            'golden': array.array('B', [
                10,
                int(10 * ((1 + np.sqrt(5)) / 2)) % 256,
                int(10 * ((1 + np.sqrt(5)) / 2) ** 2) % 256
            ]),
            'periodic': array.array('B', [0, 1, 0, 1, 0, 1, 0, 1]),
            'polynomial': array.array('B', [0, 1, 4, 9, 16, 25, 36]),
            'logarithmic': array.array('B', [0, 69, 110, 137, 158, 175, 189]),
            'power_law': array.array('B', [1, 2, 4, 8, 16, 32, 64]),
            'symmetry': array.array('B', [1, 2, 3, 3, 2, 1])
        }
        
        for pattern_type, data in patterns.items():
            binary = Binary(bytes(data))
            detected = binary.analyze_patterns()
            self.assertIn(
                pattern_type,
                [p[0] for p in detected.items() if p[1]],
                f"Failed to detect {pattern_type} pattern"
            )

    def test_bidirectional_translation(self):
        """Test translation in both directions."""
        # First translate to binary
        binary = self.translator.translate_to_binary(self.complex_test_code)
        self.translator.set_binary(binary)
        
        # Then translate back to code
        recovered_code = self.translator.translate_from_binary()
        
        # Should be able to parse recovered code
        self.assertIsNotNone(recovered_code)
        try:
            ast.parse(recovered_code)
        except SyntaxError:
            self.fail("Recovered code has syntax errors")

    def test_pattern_confidence(self):
        """Test pattern confidence calculations."""
        # Create binary with strong pattern
        perfect_fibonacci = array.array('B', [1, 1, 2, 3, 5, 8, 13, 21])
        binary = Binary(bytes(perfect_fibonacci))
        
        patterns = binary.analyze_patterns()
        if 'fibonacci' in patterns:
            confidence = patterns['fibonacci'][0][0]
            self.assertGreater(
                confidence,
                0.8,
                "Strong Fibonacci pattern should have high confidence"
            )

    def test_translation_metrics(self):
        """Test translation metrics tracking."""
        binary = self.translator.translate_to_binary(self.complex_test_code)
        
        # Check all required metrics
        required_metrics = [
            'translation_pattern_preservation_score',
            'translation_translation_confidence',
            'translation_patterns_preserved',
            'translation_total_patterns'
        ]
        
        for metric in required_metrics:
            self.assertIn(
                metric,
                binary.metadata,
                f"Missing required metric: {metric}"
            )
            
        # Validate metric values
        preservation_score = float(
            binary.metadata['translation_pattern_preservation_score']
        )
        self.assertGreaterEqual(preservation_score, 0.0)
        self.assertLessEqual(preservation_score, 1.0)

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with None input
        with self.assertRaises(Exception):
            self.translator.translate_to_binary(None)
        
        # Test with empty string
        binary = self.translator.translate_to_binary("")
        self.assertIsInstance(binary, Binary)
        
        # Test with very large input
        large_code = "x = 1\n" * 1000
        binary = self.translator.translate_to_binary(large_code)
        self.assertIsInstance(binary, Binary)
        
        # Test with special characters
        special_code = "# 特殊字符 🌟\nx = 1"
        binary = self.translator.translate_to_binary(special_code)
        self.assertIsInstance(binary, Binary)


if __name__ == '__main__':
    unittest.main() 