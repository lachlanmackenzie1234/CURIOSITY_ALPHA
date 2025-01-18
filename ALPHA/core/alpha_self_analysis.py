#!/usr/bin/env python3
"""ALPHA Self-Analysis System.

Enables ALPHA to analyze and evolve using its own codebase as a learning
environment.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Set, TypedDict, Union, cast

from ALPHA.core.interface import create_alpha


class TranslationEffectiveness(TypedDict):
    """Type definition for translation effectiveness metrics."""

    success_rate: float
    error_rate: float
    total_translations: int


class LearningMetrics(TypedDict):
    """Type definition for learning metrics."""

    pattern_recognition: float
    adaptation_rate: float
    confidence_score: float


class AnalysisResults(TypedDict):
    """Type definition for analysis results."""

    files_analyzed: int
    patterns_identified: int
    pattern_stats: DefaultDict[str, int]
    translation_effectiveness: TranslationEffectiveness
    learning_metrics: LearningMetrics
    recommendations: List[Dict[str, str]]


class ALPHASelfAnalysis:
    """Manages ALPHA's self-analysis and evolution process."""

    def __init__(self):
        """Initialize ALPHA self-analysis system."""
        # Initialize logger
        self.logger = logging.getLogger("alpha_self_analysis")
        self.logger.setLevel(logging.INFO)

        # Configure logging
        log_file = f"alpha_evolution_{int(time.time())}.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        try:
            # Initialize ALPHA instance
            self.alpha = create_alpha()

            # Initialize analysis tracking
            self.python_files: List[Path] = []
            self.analyzed_files: Set[Path] = set()
            self.component_analysis: Dict[str, dict] = {}
            self.component_signatures: Dict[str, str] = {}
            self.learning_challenges: Dict[str, List[str]] = defaultdict(list)

            # Initialize metrics
            self.calibration_metrics: Dict[str, Union[int, float, set, list]] = {
                "patterns_identified": 0,
                "pattern_types": set(),
                "confidence_scores": [],
                "learning_rates": [],
                "success_rates": [],
                "confidence_baseline": 0.0,
            }

            self.logger.info("Initialization complete!")

        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    async def analyze_codebase(self, target_dir: str) -> AnalysisResults:
        """Analyze the ALPHA codebase.

        Args:
            target_dir: The directory to analyze, relative to project root.

        Returns:
            A dictionary containing analysis results and metrics.
        """
        self.logger.info(f"Starting analysis of {target_dir}...")

        # Get all Python files in target directory
        target_path = Path(target_dir)
        self.python_files = list(target_path.rglob("*.py"))

        # Initialize analysis results
        pattern_stats: DefaultDict[str, int] = defaultdict(int)
        results: AnalysisResults = {
            "files_analyzed": 0,
            "patterns_identified": 0,
            "pattern_stats": pattern_stats,
            "translation_effectiveness": {
                "success_rate": 0.0,
                "error_rate": 0.0,
                "total_translations": 0,
            },
            "learning_metrics": {
                "pattern_recognition": 0.0,
                "adaptation_rate": 0.0,
                "confidence_score": 0.0,
            },
            "recommendations": [],
        }

        # Track translation metrics
        successful_translations = 0
        failed_translations = 0

        # Analyze each file
        for file_path in self.python_files:
            if file_path in self.analyzed_files:
                continue

            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if not content:
                    continue

                # Process file content
                try:
                    # Translate to binary format
                    binary = self.alpha.translator.translate_to_binary(content)
                    self.alpha.translator.set_binary(binary)

                    # Analyze code structure
                    structure = self.alpha.translator.analyze_structure()

                    # Update pattern statistics
                    pattern_stats["input"] += len(structure.get("imports", []))
                    pattern_stats["interaction"] += len(structure.get("functions", [])) + len(
                        structure.get("assignments", [])
                    )
                    pattern_stats["role"] += len(structure.get("classes", []))
                    pattern_stats["output"] += structure.get("expression_count", 0)

                    # Attempt translation back to Python
                    translated = self.alpha.translator.translate_from_binary()
                    if translated is not None:
                        successful_translations += 1
                    else:
                        failed_translations += 1

                    # Process through ALPHA
                    self.alpha.process(content)

                except Exception as e:
                    err_msg = f"Error processing {file_path}: {str(e)}"
                    self.logger.error(err_msg)
                    failed_translations += 1
                    continue

                # Update results
                results["files_analyzed"] += 1
                results["patterns_identified"] = sum(pattern_stats.values())

                # Update translation effectiveness
                total = successful_translations + failed_translations
                if total > 0:
                    success_rate = successful_translations / total
                    results["translation_effectiveness"] = {
                        "success_rate": success_rate * 100,
                        "error_rate": (1 - success_rate) * 100,
                        "total_translations": total,
                    }

                # Track learning metrics
                metrics = cast(LearningMetrics, results["learning_metrics"])

                try:
                    # Calculate pattern recognition rate
                    files = max(results["files_analyzed"], 1)
                    total_patterns = results["patterns_identified"] / files
                    metrics["pattern_recognition"] = min(
                        total_patterns * 10.0,  # Scale for readability
                        100.0,  # Cap at 100%
                    )

                    # Calculate adaptation rate
                    learning_rates = cast(List[float], self.calibration_metrics["learning_rates"])
                    metrics["adaptation_rate"] = (
                        float(sum(learning_rates)) / max(len(learning_rates), 1)
                        if learning_rates
                        else 0.3  # Default adaptation rate
                    )

                    # Calculate confidence score
                    metrics["confidence_score"] = float(
                        min(
                            metrics["pattern_recognition"] / 100.0 + metrics["adaptation_rate"],
                            1.0,  # Cap at 1.0
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error calculating metrics: {str(e)}")

                # Clear memory after processing each file
                self.alpha.clear_memory()

            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {str(e)}")
                continue

        self.logger.info("Analysis complete!")
        return results
