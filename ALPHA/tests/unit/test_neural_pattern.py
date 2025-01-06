"""Tests for neural pattern analysis."""

import unittest
import array
from ALPHA.core.patterns.neural_pattern import (
    NeuralPattern,
    ComponentSignature
)


class TestNeuralPattern(unittest.TestCase):
    """Test cases for neural pattern analysis."""
    
    def setUp(self):
        """Set up test cases."""
        self.neural_pattern = NeuralPattern("test_pattern")
        
        # Sample code pattern for testing
        self.test_code = """
        def example_function():
            x = 1
            y = 2
            return x + y
        """.encode('utf-8')
    
    def test_pattern_evolution_integration(self):
        """Test pattern evolution integration."""
        # Initial analysis
        signature = self.neural_pattern.analyze_component(self.test_code)
        
        # Check evolution metrics exist
        self.assertIsNotNone(signature.evolution_metrics)
        self.assertIn('success_rate', signature.evolution_metrics)
        self.assertIn('adaptation_rate', signature.evolution_metrics)
        
        # Check pattern history
        self.assertIn('test_pattern', self.neural_pattern.pattern_history)
        self.assertEqual(len(self.neural_pattern.pattern_history['test_pattern']), 1)
        
        # Test learning and adaptation
        self.neural_pattern.learn_component_behavior(self.test_code)
        self.assertGreater(len(self.neural_pattern.learned_patterns), 0)
        
        # Check confidence updates
        self.assertIn('test_pattern', self.neural_pattern.pattern_confidence)
        confidence = self.neural_pattern.pattern_confidence['test_pattern']
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_pattern_enhancement(self):
        """Test pattern enhancement capabilities."""
        # Create test pattern
        test_pattern = array.array('B', [1, 2, 3, 4, 5])
        
        # Initial confidence
        initial_confidence = self.neural_pattern.pattern_confidence.get(
            'test_pattern', 0.0
        )
        
        # Enhance pattern
        self.neural_pattern._enhance_pattern(test_pattern)
        
        # Check confidence changes
        new_confidence = self.neural_pattern.pattern_confidence.get(
            'test_pattern', 0.0
        )
        self.assertGreaterEqual(new_confidence, initial_confidence)
    
    def test_evolution_metrics_calculation(self):
        """Test evolution metrics calculation."""
        # Analyze component
        signature = self.neural_pattern.analyze_component(self.test_code)
        
        # Check metric ranges
        metrics = signature.evolution_metrics
        for metric_name, value in metrics.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            
        # Check specific metrics exist
        expected_metrics = {'success_rate', 'adaptation_rate', 'improvement_rate'}
        self.assertTrue(
            all(metric in metrics for metric in expected_metrics)
        )


if __name__ == '__main__':
    unittest.main() 