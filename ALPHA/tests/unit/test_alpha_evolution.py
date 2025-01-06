"""Unit tests for ALPHA evolution system."""

import unittest
import asyncio
import signal
from ALPHA.core.analysis.run_alpha_evolution import (
    PatternEvolution,
    EvolutionMetrics
)


class TestALPHAEvolution(unittest.TestCase):
    """Test cases for ALPHA evolution functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.evolution = PatternEvolution()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Test data
        self.test_recommendations = [
            {
                'type': 'pattern_optimization',
                'target': 'memory_usage',
                'description': 'Optimize memory pattern usage',
                'confidence': 0.8
            },
            {
                'type': 'code_improvement',
                'target': 'translation',
                'description': 'Enhance pattern translation',
                'confidence': 0.9
            }
        ]

    def tearDown(self):
        """Clean up test environment."""
        self.loop.close()

    def test_basic_functionality(self):
        """Test basic evolution functionality."""
        # Verify initialization
        self.assertIsNotNone(self.evolution)
        self.assertIsInstance(self.evolution, PatternEvolution)
        
        # Check initial metrics
        metrics = self.evolution.metrics
        self.assertIsInstance(metrics, EvolutionMetrics)
        
        # Verify metric tracking setup
        summary = metrics.get_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('improvement_rate', summary)
        self.assertIn('success_rate', summary)

    def test_evolution_cycle(self):
        """Test evolution cycle execution."""
        async def run_cycle():
            # Run evolution cycle
            await self.evolution.run_evolution_cycle()
            
            # Get updated metrics
            metrics = self.evolution.metrics.get_summary()
            return metrics
        
        # Execute cycle
        metrics = self.loop.run_until_complete(run_cycle())
        
        # Verify cycle execution
        self.assertGreater(metrics['cycles_completed'], 0)
        self.assertGreaterEqual(metrics['success_rate'], 0.0)
        self.assertLessEqual(metrics['success_rate'], 1.0)

    def test_improvement_application(self):
        """Test improvement application."""
        async def apply_improvements():
            results = []
            # Apply each recommendation
            for rec in self.test_recommendations:
                success = await self.evolution.apply_improvement(rec)
                results.append((rec, success))
            return results
        
        # Apply improvements
        results = self.loop.run_until_complete(apply_improvements())
        
        # Verify results
        for rec, success in results:
            self.assertIsInstance(success, bool)
            if success:
                metrics = self.evolution.metrics.get_summary()
                self.assertGreater(
                    metrics['improvements_applied'],
                    0
                )

    def test_metrics_tracking(self):
        """Test evolution metrics tracking."""
        # Initial state
        initial_metrics = self.evolution.metrics.get_summary()
        
        async def run_evolution():
            # Run multiple cycles
            for _ in range(3):
                await self.evolution.run_evolution_cycle()
            
            # Apply improvements
            for rec in self.test_recommendations:
                await self.evolution.apply_improvement(rec)
            
            return self.evolution.metrics.get_summary()
        
        # Run evolution
        final_metrics = self.loop.run_until_complete(run_evolution())
        
        # Verify metric changes
        self.assertGreater(
            final_metrics['cycles_completed'],
            initial_metrics['cycles_completed']
        )
        self.assertGreater(
            final_metrics['improvements_applied'],
            initial_metrics['improvements_applied']
        )

    def test_shutdown_handling(self):
        """Test shutdown signal handling."""
        # Test shutdown handler
        self.evolution._handle_shutdown(signal.SIGTERM, None)
        
        # Verify shutdown state
        self.assertTrue(self.evolution.should_stop)
        
        # Verify graceful shutdown
        async def check_shutdown():
            try:
                await self.evolution.run()
                return True
            except Exception:
                return False
        
        success = self.loop.run_until_complete(check_shutdown())
        self.assertTrue(success)

    def test_error_handling(self):
        """Test error handling capabilities."""
        async def test_invalid_inputs():
            # Test invalid recommendation
            invalid_recs = [
                {},  # Empty
                {'type': 'invalid'},  # Missing fields
                None,  # None
                {  # Invalid confidence
                    'type': 'pattern_optimization',
                    'confidence': 2.0
                }
            ]
            
            results = []
            for rec in invalid_recs:
                try:
                    success = await self.evolution.apply_improvement(rec)
                    results.append(success)
                except Exception as e:
                    results.append(e)
            
            return results
        
        # Run error tests
        results = self.loop.run_until_complete(test_invalid_inputs())
        
        # Verify error handling
        for result in results:
            if isinstance(result, bool):
                self.assertFalse(result)
            else:
                self.assertIsInstance(result, Exception)

    def test_concurrent_evolution(self):
        """Test concurrent evolution capabilities."""
        async def run_concurrent():
            # Create multiple evolution tasks
            tasks = [
                self.evolution.run_evolution_cycle()
                for _ in range(3)
            ]
            
            # Run concurrently
            results = await asyncio.gather(*tasks)
            return results
        
        # Run concurrent evolution
        results = self.loop.run_until_complete(run_concurrent())
        
        # Verify results
        self.assertEqual(len(results), 3)
        metrics = self.evolution.metrics.get_summary()
        self.assertGreaterEqual(metrics['cycles_completed'], 3)

    def test_evolution_stability(self):
        """Test evolution system stability."""
        async def check_stability():
            initial_metrics = self.evolution.metrics.get_summary()
            
            # Run multiple cycles
            for _ in range(5):
                await self.evolution.run_evolution_cycle()
                
                # Apply random improvements
                for rec in self.test_recommendations:
                    await self.evolution.apply_improvement(rec)
            
            final_metrics = self.evolution.metrics.get_summary()
            return initial_metrics, final_metrics
        
        # Check stability
        initial, final = self.loop.run_until_complete(check_stability())
        
        # Verify system remains stable
        self.assertGreaterEqual(
            final['success_rate'],
            initial['success_rate'] * 0.8  # Allow some degradation
        )
        self.assertGreater(
            final['improvements_applied'],
            initial['improvements_applied']
        ) 