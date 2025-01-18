#!/usr/bin/env python3
"""ALPHA Self-Evolution Runner.

Coordinates ALPHA's self-analysis and evolution process.
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from ..alpha_self_analysis import ALPHASelfAnalysis
from ..execution.engine import ExecutionEngine
from ..memory.space import MemoryOrganizer
from ..monitor import create_monitor
from ..patterns.pattern import Pattern, PatternType


class EvolutionMetrics:
    """Tracks metrics for evolution process."""

    def __init__(self):
        self.start_time = time.time()
        self.patterns_processed = 0
        self.improvements_attempted = 0
        self.improvements_successful = 0
        self.errors_encountered = 0
        self.current_confidence = 0.0
        self.current_complexity = 0.0

    def get_summary(self) -> Dict[str, float]:
        """Generate performance summary."""
        runtime_hours = (time.time() - self.start_time) / 3600
        return {
            "runtime_hours": runtime_hours,
            "patterns_per_hour": (
                self.patterns_processed / runtime_hours if runtime_hours > 0 else 0
            ),
            "improvement_success_rate": (
                self.improvements_successful / self.improvements_attempted
                if self.improvements_attempted > 0
                else 0
            ),
            "error_rate": (
                self.errors_encountered / self.patterns_processed
                if self.patterns_processed > 0
                else 0
            ),
            "current_confidence": self.current_confidence,
            "current_complexity": self.current_complexity,
        }


class PatternEvolution:
    """Manages ALPHA's evolution process."""

    def __init__(self):
        """Initialize the evolution system."""
        self.analyzer = ALPHASelfAnalysis()
        self.monitor = create_monitor()
        self.engine = ExecutionEngine()
        self.memory = MemoryOrganizer()
        self.metrics = EvolutionMetrics()
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Configure logging
        log_format = "%(asctime)s | %(levelname)s | %(message)s"
        log_file = f"alpha_evolution_{int(time.time())}.log"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        )
        self.logger = logging.getLogger(__name__)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        msg = "Received shutdown signal. Initiating graceful shutdown..."
        self.logger.info(msg)
        self.running = False

    async def apply_improvement(self, recommendation: Dict) -> bool:
        """Apply a recommended improvement."""
        try:
            pattern_type = recommendation.get("pattern_type")
            target_file = recommendation.get("file")
            changes = recommendation.get("changes", {})

            if not all([pattern_type, target_file, changes]):
                self.logger.warning("Incomplete recommendation data")
                return False

            # Create improvement pattern
            pattern_id = f"improvement_{int(time.time())}"
            pattern = Pattern(id=pattern_id, pattern_type=PatternType(pattern_type))

            # Apply changes through execution engine
            result = await self.engine.execute_pattern(
                pattern_id=pattern.id,
                binary_data=pattern.data,
                target_file=target_file,
                changes=changes,
            )

            if result.success:
                # Update memory organization
                self.memory.organize_pattern(pattern)
                self.metrics.improvements_successful += 1
                msg = f"Successfully applied improvement to {target_file}"
                self.logger.info(msg)
                return True

            msg = f"Failed to apply improvement to {target_file}: {result.errors}"
            self.logger.warning(msg)
            return False

        except Exception as e:
            msg = f"Error applying improvement: {str(e)}"
            self.logger.error(msg)
            self.metrics.errors_encountered += 1
            return False

    async def run_evolution_cycle(self) -> None:
        """Run a single evolution cycle."""
        try:
            # Analyze codebase
            analysis_report = await self.analyzer.analyze_codebase()
            patterns = analysis_report.get("patterns", [])
            self.metrics.patterns_processed += len(patterns)

            # Update metrics
            self.metrics.current_confidence = analysis_report.get("average_confidence", 0)
            self.metrics.current_complexity = analysis_report.get("average_complexity", 0)
            self.monitor.update_metrics(
                {
                    "confidence": self.metrics.current_confidence,
                    "complexity": self.metrics.current_complexity,
                }
            )

            # Process and apply recommendations
            recommendations = analysis_report.get("recommendations", [])
            for rec in recommendations:
                self.monitor.record_evolution_event("improvement_suggested")
                self.metrics.improvements_attempted += 1

                msg = f"Evolution recommendation for {rec['file']}: " f"{rec['message']}"
                self.logger.info(msg)

                # Actually apply the improvement
                success = await self.apply_improvement(rec)
                if success:
                    self.monitor.record_evolution_event("improvement_applied")

            # Generate and log performance summary
            summary = self.metrics.get_summary()
            self.logger.info("\n=== Performance Summary ===")
            self.logger.info(f"Runtime: {summary['runtime_hours']:.2f} hours")
            self.logger.info(f"Patterns/hour: {summary['patterns_per_hour']:.2f}")
            self.logger.info(f"Success rate: {summary['improvement_success_rate']:.2%}")
            self.logger.info(f"Error rate: {summary['error_rate']:.2%}")
            self.logger.info(f"Current confidence: {summary['current_confidence']:.2f}")
            self.logger.info(f"Current complexity: {summary['current_complexity']:.2f}")

        except Exception as e:
            msg = f"Error in evolution cycle: {str(e)}"
            self.logger.error(msg)
            self.metrics.errors_encountered += 1

    async def run(self) -> None:
        """Run the evolution process continuously."""
        self.logger.info("Starting ALPHA evolution process...")

        while self.running:
            await self.run_evolution_cycle()
            await asyncio.sleep(60)  # Wait between cycles

        self.logger.info("ALPHA evolution process completed.")


async def main():
    """Run the ALPHA evolution process."""
    evolution = PatternEvolution()
    await evolution.run()


if __name__ == "__main__":
    asyncio.run(main())
