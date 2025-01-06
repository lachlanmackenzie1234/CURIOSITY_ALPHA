"""Unit tests for ALPHA self-analysis functionality."""

import asyncio
import unittest
from collections import defaultdict
from pathlib import Path
from ALPHA.core.alpha_self_analysis import ALPHASelfAnalysis


class TestALPHASelfAnalysis(unittest.TestCase):
    """Test cases for ALPHA self-analysis functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = ALPHASelfAnalysis()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Test data
        self.test_dir = str(Path(__file__).parent / 'test_data')
        self.test_files = {
            'simple': 'def test():\n    return True',
            'pattern': 'for i in range(5):\n    print(i ** 2)',
            'complex': '''
                class TestClass:
                    def __init__(self):
                        self.data = []
                    
                    def process(self, item):
                        self.data.append(item)
                        return len(self.data)
            '''
        }

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_basic_functionality(self):
        """Test basic self-analysis functionality."""
        # Verify initialization
        self.assertIsNotNone(self.analyzer)
        self.assertIsInstance(self.analyzer, ALPHASelfAnalysis)
        
        # Check default state
        self.assertEqual(self.analyzer.files_processed, 0)
        self.assertIsInstance(self.analyzer.pattern_stats, defaultdict)

    def test_pattern_recognition(self):
        """Test pattern recognition capabilities."""
        async def analyze_patterns():
            # Analyze test files
            results = await self.analyzer.analyze_codebase(self.test_dir)
            return results
        
        # Run analysis
        results = self.loop.run_until_complete(analyze_patterns())
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('patterns_identified', results)
        self.assertIn('pattern_stats', results)
        
        # Check pattern identification
        self.assertGreater(results['patterns_identified'], 0)
        self.assertGreater(len(results['pattern_stats']), 0)
        
        # Verify pattern types
        pattern_types = results['pattern_stats'].keys()
        self.assertTrue(any('loop' in pt for pt in pattern_types))
        self.assertTrue(any('class' in pt for pt in pattern_types))

    def test_translation_effectiveness(self):
        """Test translation effectiveness analysis."""
        async def check_translation():
            results = await self.analyzer.analyze_codebase(self.test_dir)
            return results['translation_effectiveness']
        
        # Get translation metrics
        metrics = self.loop.run_until_complete(check_translation())
        
        # Verify metrics structure
        self.assertIsInstance(metrics, dict)
        self.assertIn('success_rate', metrics)
        self.assertIn('error_rate', metrics)
        self.assertIn('total_translations', metrics)
        
        # Check metric values
        self.assertGreaterEqual(metrics['success_rate'], 0.0)
        self.assertLessEqual(metrics['success_rate'], 1.0)
        self.assertGreaterEqual(metrics['error_rate'], 0.0)
        self.assertLessEqual(metrics['error_rate'], 1.0)
        self.assertGreater(metrics['total_translations'], 0)

    def test_component_analysis(self):
        """Test component analysis capabilities."""
        async def analyze_components():
            results = await self.analyzer.analyze_codebase(self.test_dir)
            return results['component_analysis']
        
        # Get component analysis
        analysis = self.loop.run_until_complete(analyze_components())
        
        # Verify analysis structure
        self.assertIsInstance(analysis, dict)
        self.assertIn('complexity', analysis)
        self.assertIn('cohesion', analysis)
        self.assertIn('coupling', analysis)
        
        # Check metric ranges
        self.assertGreaterEqual(analysis['complexity'], 0.0)
        self.assertLessEqual(analysis['complexity'], 1.0)
        self.assertGreaterEqual(analysis['cohesion'], 0.0)
        self.assertLessEqual(analysis['cohesion'], 1.0)
        self.assertGreaterEqual(analysis['coupling'], 0.0)
        self.assertLessEqual(analysis['coupling'], 1.0)

    def test_incremental_analysis(self):
        """Test incremental analysis capabilities."""
        async def analyze_incremental():
            # First analysis
            results1 = await self.analyzer.analyze_codebase(self.test_dir)
            stats1 = results1['pattern_stats'].copy()
            
            # Add more test data and analyze again
            results2 = await self.analyzer.analyze_codebase(self.test_dir)
            stats2 = results2['pattern_stats']
            
            return stats1, stats2
        
        # Run incremental analysis
        stats1, stats2 = self.loop.run_until_complete(analyze_incremental())
        
        # Verify pattern accumulation
        self.assertGreaterEqual(
            sum(stats2.values()),
            sum(stats1.values())
        )

    def test_error_handling(self):
        """Test error handling capabilities."""
        async def test_invalid_inputs():
            # Test invalid directory
            try:
                await self.analyzer.analyze_codebase('/nonexistent/dir')
                self.fail('Should raise FileNotFoundError')
            except FileNotFoundError:
                pass
            
            # Test empty directory
            results = await self.analyzer.analyze_codebase(
                str(Path(self.test_dir) / 'empty')
            )
            self.assertEqual(results['files_analyzed'], 0)
            
            # Test invalid file content
            results = await self.analyzer.analyze_codebase(
                str(Path(self.test_dir) / 'invalid')
            )
            self.assertLess(
                results['translation_effectiveness']['success_rate'],
                1.0
            )
        
        # Run error tests
        self.loop.run_until_complete(test_invalid_inputs())


if __name__ == '__main__':
    unittest.main() 