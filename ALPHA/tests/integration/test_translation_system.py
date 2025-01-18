"""Integration tests for the ALPHA translation system.

Tests the interaction between:
1. Pattern Recognition
2. Pattern Resonance
3. Binary Mapping
4. Pattern-First Translation
"""

import unittest

import numpy as np
from core.patterns.binary_mapping import PatternMapper
from core.patterns.natural_patterns import NaturalPatternHierarchy
from core.patterns.resonance import PatternResonance
from core.translation.pattern_first import PatternFirstTranslator


class TestTranslationSystem(unittest.TestCase):
    """Integration tests for the complete translation system."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize all components
        self.translator = PatternFirstTranslator()
        self.resonance = PatternResonance()
        self.mapper = PatternMapper()
        self.hierarchy = NaturalPatternHierarchy()

        # Test sequences with known patterns
        self.test_sequences = {
            "fibonacci": "1,1,2,3,5,8,13,21,34,55",
            "prime": "2,3,5,7,11,13,17,19,23,29",
            "geometric": "2,4,8,16,32,64,128,256",
            "mixed": "1,2,3,5,8,11,13,17,21,34",
            "noise": "Hello, World! 123",
        }

    def test_end_to_end_translation(self):
        """Test complete translation pipeline."""
        # Original sequence
        sequence = self.test_sequences["fibonacci"]

        # Step 1: Pattern Recognition
        data = np.array([ord(c) for c in sequence])
        pattern = self.hierarchy.detect_natural_pattern(data)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.principle_type, "fibonacci")
        self.assertGreater(pattern.confidence, 0.8)

        # Step 2: Calculate Resonance
        resonance = self.resonance.calculate_resonance(pattern)

        self.assertGreater(resonance.strength, 0.0)
        self.assertGreater(resonance.stability, 0.0)
        self.assertGreater(resonance.harmony, 0.0)

        # Step 3: Binary Mapping
        mapping = self.mapper.map_to_binary(pattern, data)

        self.assertIsNotNone(mapping.binary_form)
        self.assertGreater(mapping.mapping_confidence, 0.7)

        # Step 4: Full Translation
        binary = self.translator.translate_to_binary(sequence, preserve_patterns=True)

        self.assertIsNotNone(binary)
        self.assertGreater(len(binary), 0)

        # Step 5: Recovery
        recovered = self.translator.translate_from_binary(binary, preserve_patterns=True)

        self.assertEqual(recovered, sequence)

    def test_pattern_interaction_chain(self):
        """Test pattern interaction through the system."""
        # Create sequence with multiple patterns
        sequence = self.test_sequences["mixed"]

        # Step 1: Identify Multiple Patterns
        data = np.array([ord(c) for c in sequence])
        units = self.translator._identify_patterns(data)

        self.assertTrue(any(len(unit.patterns) > 0 for unit in units))

        # Step 2: Analyze Pattern Interactions
        patterns = {}
        for unit in units:
            patterns.update(unit.patterns)

        profiles = self.resonance.analyze_pattern_interactions(patterns, data)

        self.assertGreater(len(profiles), 0)

        # Step 3: Create Optimized Mappings
        mappings = {}
        for pattern_id, pattern in patterns.items():
            mapping = self.mapper.map_to_binary(pattern, data)
            mappings[pattern_id] = mapping

        self.assertEqual(len(mappings), len(patterns))

        # Step 4: Verify Pattern Preservation
        binary = self.translator.translate_to_binary(sequence, preserve_patterns=True)
        recovered = self.translator.translate_from_binary(binary, preserve_patterns=True)

        self.assertEqual(recovered, sequence)

    def test_noise_resilience(self):
        """Test system's resilience to noise and pattern mixing."""
        # Mix patterns with noise
        clean_sequence = self.test_sequences["fibonacci"]
        noisy_sequence = clean_sequence + self.test_sequences["noise"]

        # Step 1: Pattern Detection in Noise
        data = np.array([ord(c) for c in noisy_sequence])
        pattern = self.hierarchy.detect_natural_pattern(data)

        self.assertIsNotNone(pattern)
        self.assertGreater(pattern.confidence, 0.5)

        # Step 2: Resonance in Noise
        resonance = self.resonance.calculate_resonance(pattern)
        self.assertGreater(resonance.stability, 0.3)

        # Step 3: Translation with Noise
        binary = self.translator.translate_to_binary(noisy_sequence, preserve_patterns=True)
        recovered = self.translator.translate_from_binary(binary, preserve_patterns=True)

        self.assertEqual(recovered, noisy_sequence)

    def test_pattern_amplification_chain(self):
        """Test pattern amplification through the system."""
        # Create weak pattern sequence
        weak_sequence = "1,1,2,3,5"  # Short Fibonacci

        # Step 1: Detect Weak Pattern
        data = np.array([ord(c) for c in weak_sequence])
        weak_pattern = self.hierarchy.detect_natural_pattern(data)

        self.assertIsNotNone(weak_pattern)
        self.assertLess(weak_pattern.confidence, 0.8)

        # Step 2: Amplify with Strong Pattern
        strong_pattern = self.hierarchy.detect_natural_pattern(
            np.array([ord(c) for c in self.test_sequences["fibonacci"]])
        )

        amplified = self.resonance.amplify_pattern(weak_pattern, strong_pattern)

        self.assertGreater(amplified.confidence, weak_pattern.confidence)

        # Step 3: Verify Improved Translation
        binary_weak = self.translator.translate_to_binary(weak_sequence, preserve_patterns=False)
        binary_amplified = self.translator.translate_to_binary(
            weak_sequence, preserve_patterns=True
        )

        self.assertNotEqual(binary_weak, binary_amplified)

    def test_cross_component_optimization(self):
        """Test optimization across all components."""
        sequence = self.test_sequences["geometric"]

        # Step 1: Initial Translation
        binary1 = self.translator.translate_to_binary(sequence, preserve_patterns=True)

        # Step 2: Enhance Pattern Recognition
        data = np.array([ord(c) for c in sequence])
        pattern = self.hierarchy.detect_natural_pattern(data)
        resonance = self.resonance.calculate_resonance(pattern)

        # Use resonance to adjust confidence threshold
        confidence_threshold = 0.7 * resonance.stability

        # Step 3: Optimize Mapping with adjusted confidence
        optimized_mapping = self.mapper.map_to_binary(
            pattern,
            data,
            encoding_type=self.mapper.ENCODING_TYPES[0],
            min_confidence=confidence_threshold,
        )

        # Apply optimized mapping settings
        self.mapper.update_encoding_settings(optimized_mapping)

        # Step 4: Final Translation
        binary2 = self.translator.translate_to_binary(sequence, preserve_patterns=True)

        # Verify Optimization
        self.assertNotEqual(binary1, binary2)
        self.assertLess(len(binary2), len(binary1))


if __name__ == "__main__":
    unittest.main()
