"""Unit tests for the pattern resonance system."""

import unittest
import numpy as np

from core.patterns.resonance import PatternResonance
from core.patterns.natural_patterns import NaturalPattern


class TestPatternResonance(unittest.TestCase):
    """Test cases for PatternResonance."""

    def setUp(self):
        """Set up test fixtures."""
        self.resonance = PatternResonance()
        
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
        
        self.geometric_pattern = NaturalPattern(
            principle_type="geometric",
            sequence=[2, 4, 8, 16, 32],
            confidence=0.85
        )
    
    def test_resonance_calculation(self):
        """Test calculation of pattern resonance."""
        # Calculate resonance for single pattern
        resonance = self.resonance.calculate_resonance(
            self.fibonacci_pattern
        )
        
        self.assertGreater(resonance.strength, 0.0)
        self.assertGreater(resonance.stability, 0.0)
        self.assertGreater(resonance.harmony, 0.0)
    
    def test_pattern_interaction(self):
        """Test analysis of pattern interactions."""
        # Create pattern dictionary
        patterns = {
            'fib': self.fibonacci_pattern,
            'prime': self.prime_pattern
        }
        
        # Analyze interactions
        context = np.array([1, 1, 2, 3, 5, 7, 11])
        profiles = self.resonance.analyze_pattern_interactions(
            patterns,
            context
        )
        
        # Verify interaction profiles
        self.assertEqual(len(profiles), 2)
        self.assertIn('fib', profiles)
        self.assertIn('prime', profiles)
    
    def test_resonance_strength(self):
        """Test calculation of resonance strength."""
        # Calculate for different patterns
        fib_strength = self.resonance.calculate_resonance_strength(
            self.fibonacci_pattern
        )
        prime_strength = self.resonance.calculate_resonance_strength(
            self.prime_pattern
        )
        geo_strength = self.resonance.calculate_resonance_strength(
            self.geometric_pattern
        )
        
        # Verify relative strengths
        self.assertGreater(fib_strength, 0.0)
        self.assertGreater(prime_strength, 0.0)
        self.assertGreater(geo_strength, 0.0)
    
    def test_resonance_stability(self):
        """Test calculation of resonance stability."""
        # Calculate stability with context
        context = np.array([1, 1, 2, 3, 5, 8])
        stability = self.resonance.calculate_stability(
            self.fibonacci_pattern,
            context
        )
        
        # Verify stability score
        self.assertGreater(stability, 0.7)
    
    def test_resonance_harmony(self):
        """Test calculation of resonance harmony."""
        # Calculate harmony between patterns
        harmony = self.resonance.calculate_harmony(
            self.fibonacci_pattern,
            self.prime_pattern
        )
        
        # Verify harmony score
        self.assertGreaterEqual(harmony, 0.0)
        self.assertLessEqual(harmony, 1.0)
    
    def test_pattern_amplification(self):
        """Test pattern resonance amplification."""
        # Create weak pattern
        weak_pattern = NaturalPattern(
            principle_type="fibonacci",
            sequence=[1, 1, 2],
            confidence=0.6
        )
        
        # Amplify with strong pattern
        amplified = self.resonance.amplify_pattern(
            weak_pattern,
            self.fibonacci_pattern
        )
        
        # Verify amplification
        self.assertGreater(
            amplified.confidence,
            weak_pattern.confidence
        )
    
    def test_resonance_decay(self):
        """Test resonance decay over distance."""
        # Calculate decay
        distance = 3
        decay = self.resonance.calculate_decay(
            self.fibonacci_pattern,
            distance
        )
        
        # Verify decay properties
        self.assertGreater(decay, 0.0)
        self.assertLess(decay, 1.0)
    
    def test_error_handling(self):
        """Test error handling in resonance calculations."""
        # Test with invalid pattern
        invalid_pattern = NaturalPattern(
            principle_type="invalid",
            sequence=[],
            confidence=0.0
        )
        
        resonance = self.resonance.calculate_resonance(
            invalid_pattern
        )
        
        # Verify safe handling
        self.assertEqual(resonance.strength, 0.0)
        self.assertEqual(resonance.stability, 0.0)
        self.assertEqual(resonance.harmony, 0.0)


if __name__ == '__main__':
    unittest.main() 