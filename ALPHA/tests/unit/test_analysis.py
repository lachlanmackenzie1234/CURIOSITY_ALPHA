#!/usr/bin/env python3
"""Test script for ALPHA self-analysis."""

import unittest

from ALPHA.core.alpha_self_analysis import ALPHASelfAnalysis


class TestALPHASelfAnalysis(unittest.TestCase):
    """Test cases for ALPHA self-analysis functionality."""

    def setUp(self):
        """Set up test environment."""
        self.analysis = ALPHASelfAnalysis()

    def test_initialization(self):
        """Test initialization of self-analysis system."""
        self.assertIsNotNone(self.analysis.alpha)
        self.assertIsNotNone(self.analysis.logger)
        self.assertIsInstance(self.analysis.python_files, list)
        self.assertIsInstance(self.analysis.analyzed_files, set)

    def test_analyze_codebase(self):
        """Test codebase analysis functionality."""
        results = self.analysis.analyze_codebase("ALPHA/core")
        self.assertIsNotNone(results)
        self.assertIn("files_analyzed", results)
        self.assertIn("patterns_identified", results)
        self.assertIn("translation_effectiveness", results)
        self.assertIn("learning_metrics", results)


if __name__ == "__main__":
    unittest.main()
