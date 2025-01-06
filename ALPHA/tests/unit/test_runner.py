"""Test runner for neural pattern analysis."""

import unittest
from ALPHA.core.patterns.neural_pattern import NeuralPattern


class TestNeuralRunner(unittest.TestCase):
    """Test cases for neural pattern runner."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = NeuralPattern("test_analyzer")
        self.test_content = (
            b"def test_function(x, y):\n"
            b"    result = x + y\n"
            b"    return result\n"
        )
        self.similar_content = (
            b"def another_function(a, b):\n"
            b"    value = a + b\n"
            b"    return value\n"
        )
    
    def test_component_analysis(self):
        """Test component analysis functionality."""
        signature = self.analyzer.analyze_component(self.test_content)
        
        # Verify signature structure
        self.assertIsNotNone(signature)
        self.assertTrue(hasattr(signature, 'input_patterns'))
        self.assertTrue(hasattr(signature, 'output_patterns'))
        self.assertTrue(hasattr(signature, 'interaction_patterns'))
        self.assertTrue(hasattr(signature, 'role_confidence'))
        self.assertTrue(hasattr(signature, 'performance_metrics'))
    
    def test_learning_capability(self):
        """Test learning capabilities."""
        # Test learning
        self.analyzer.learn_component_behavior(self.test_content)
        self.assertGreater(len(self.analyzer.learned_patterns), 0)
    
    def test_improvement_suggestions(self):
        """Test improvement suggestion functionality."""
        # Learn from original content
        self.analyzer.learn_component_behavior(self.test_content)
        
        # Get suggestions for similar content
        suggestions = self.analyzer.suggest_improvements(self.similar_content)
        
        # Verify suggestions structure
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        for suggestion in suggestions:
            self.assertIn('confidence', suggestion)
            self.assertIn('description', suggestion)
            self.assertIn('suggested_changes', suggestion)
            self.assertGreaterEqual(suggestion['confidence'], 0.0)
            self.assertLessEqual(suggestion['confidence'], 1.0)


if __name__ == '__main__':
    unittest.main() 