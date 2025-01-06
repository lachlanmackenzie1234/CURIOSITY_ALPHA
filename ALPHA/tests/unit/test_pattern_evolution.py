import unittest
import numpy as np
import array
from ALPHA.core.patterns.pattern_evolution import PatternEvolution

class TestPatternEvolution(unittest.TestCase):
    """Test cases for pattern evolution system."""
    
    def setUp(self):
        """Set up test environment."""
        self.evolution = PatternEvolution()
        
        # Create test pattern
        self.test_data = array.array('B', [
            0, 10, 20, 30, 40, 50, 60, 70,  # Gradual increase
            70, 60, 50, 40, 30, 20, 10, 0,  # Gradual decrease (symmetry)
            0, 0, 255, 255, 0, 0, 255, 255  # Repeating pattern
        ])
        
        # Create test context
        self.test_context = {
            'expected_role': 'processor',
            'expected_behavior': {
                'regularity': 0.8,
                'symmetry': 0.7,
                'complexity': 0.5
            },
            'performance_targets': {
                'efficiency': 0.7,
                'reliability': 0.8,
                'accuracy': 0.9
            }
        }
    
    def test_pattern_metrics(self):
        """Test pattern metrics calculation."""
        metrics = self.evolution._calculate_pattern_metrics(
            self.test_data,
            self.test_context
        )
        
        # Verify all expected metrics are present
        expected_metrics = {
            'success_rate', 'adaptation_rate', 'improvement_rate',
            'stability', 'effectiveness', 'complexity'
        }
        self.assertEqual(set(metrics.keys()), expected_metrics)
        
        # Verify metric values are in valid range [0, 1]
        for metric, value in metrics.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_pattern_evolution(self):
        """Test pattern evolution over multiple iterations."""
        pattern_id = "test_pattern"
        
        # Simulate pattern evolution over multiple iterations
        for i in range(5):
            # Evolve pattern with slight modifications
            evolved_data = array.array('B', np.clip(
                np.frombuffer(self.test_data, dtype=np.uint8) + 
                np.random.normal(0, 10, len(self.test_data)),
                0, 255
            ).astype(np.uint8))
            
            # Record evolution
            self.evolution.pattern_history[pattern_id] = []
            self.evolution.pattern_history[pattern_id].append({
                'pattern_data': evolved_data,
                'metrics': self.evolution._calculate_pattern_metrics(
                    evolved_data,
                    self.test_context
                )
            })
        
        # Calculate adaptation rate
        adaptation_rate = self.evolution._calculate_adaptation_rate(
            pattern_id,
            self.test_data
        )
        
        # Verify adaptation rate is in valid range
        self.assertGreaterEqual(adaptation_rate, 0.0)
        self.assertLessEqual(adaptation_rate, 1.0)
    
    def test_pattern_analysis(self):
        """Test pattern analysis capabilities."""
        # Test entropy calculation
        entropy = self.evolution._calculate_entropy(
            np.frombuffer(self.test_data, dtype=np.uint8)
        )
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)
        
        # Test symmetry calculation
        symmetry = self.evolution._calculate_symmetry(
            np.frombuffer(self.test_data, dtype=np.uint8)
        )
        self.assertGreaterEqual(symmetry, 0.0)
        self.assertLessEqual(symmetry, 1.0)
        
        # Test consistency calculation
        consistency = self.evolution._calculate_consistency(
            np.frombuffer(self.test_data, dtype=np.uint8)
        )
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)
    
    def test_error_resistance(self):
        """Test error resistance capabilities."""
        # Test error resistance calculation
        resistance = self.evolution._calculate_error_resistance(
            np.frombuffer(self.test_data, dtype=np.uint8)
        )
        self.assertGreaterEqual(resistance, 0.0)
        self.assertLessEqual(resistance, 1.0)
        
        # Test correction potential
        potential = self.evolution._calculate_correction_potential(
            np.frombuffer(self.test_data, dtype=np.uint8)
        )
        self.assertGreaterEqual(potential, 0.0)
        self.assertLessEqual(potential, 1.0)

if __name__ == '__main__':
    unittest.main() 