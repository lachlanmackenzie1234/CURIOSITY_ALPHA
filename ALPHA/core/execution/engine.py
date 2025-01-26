"""Execution engine module."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...NEXUS.core.adaptive_field import AdaptiveField


@dataclass
class ExecutionMetrics:
    """Metrics for execution performance."""

    execution_time: float = 0.0
    memory_usage: int = 0
    error_rate: float = 0.0
    success_rate: float = 0.0
    optimization_gain: float = 0.0
    natural_harmony: float = 0.0


@dataclass
class ExecutionResult:
    """Result of pattern execution."""

    success: bool = False
    output: Any = None
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class ExecutionEngine(AdaptiveField):
    """Engine for executing optimized binary patterns with adaptive thresholds."""

    def __init__(self):
        """Initialize execution engine."""
        super().__init__()  # Initialize adaptive field
        self.logger = logging.getLogger("execution_engine")
        self.active_patterns: Dict[str, bytes] = {}
        self.execution_history: List[ExecutionResult] = []
        self.performance_cache: Dict[str, ExecutionMetrics] = {}

        # Register adaptive thresholds
        self.register_threshold("execution_time_limit", 1.0)  # 1 second initial limit
        self.register_threshold("optimization_threshold", 0.2)  # 20% improvement needed
        self.register_threshold("harmony_threshold", 0.5)  # Minimum harmony required
        self.register_threshold("error_tolerance", 0.1)  # 10% error tolerance

    def execute(self, binary_data: bytes) -> ExecutionResult:
        """Execute binary pattern with adaptive thresholds."""
        result = ExecutionResult()

        try:
            start_time = time.time()

            # Process binary data
            processed_data = self._process_pattern(binary_data)

            # Calculate metrics
            execution_time = time.time() - start_time
            result.metrics.execution_time = execution_time
            result.metrics.memory_usage = len(binary_data)

            # Let field observe execution time
            self.sense_pressure("execution_time_limit", execution_time)

            # Update success metrics
            if processed_data:
                result.success = True
                result.output = processed_data
                result.metrics.success_rate = 1.0

                # Calculate and observe optimization gain
                if len(processed_data) < len(binary_data):
                    optimization = 1 - len(processed_data) / len(binary_data)
                    result.metrics.optimization_gain = optimization
                    self.sense_pressure("optimization_threshold", optimization)

                # Calculate and observe natural harmony
                harmony = self._calculate_harmony(processed_data)
                result.metrics.natural_harmony = harmony
                self.sense_pressure("harmony_threshold", harmony)

                # Check against adaptive thresholds
                if execution_time > self.get_threshold("execution_time_limit"):
                    result.suggestions.append("Execution time exceeded adaptive threshold")

                if optimization < self.get_threshold("optimization_threshold"):
                    result.suggestions.append("Optimization below adaptive threshold")

                if harmony < self.get_threshold("harmony_threshold"):
                    result.suggestions.append("Harmony below adaptive threshold")

            else:
                # Track error rate
                error_rate = 1.0
                self.sense_pressure("error_tolerance", error_rate)
                if error_rate > self.get_threshold("error_tolerance"):
                    result.suggestions.append("Error rate exceeded tolerance threshold")

            # Cache performance metrics
            pattern_hash = str(hash(binary_data))
            self.performance_cache[pattern_hash] = result.metrics

            # Update history
            self.execution_history.append(result)

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            result.metrics.error_rate = 1.0
            # Observe error occurrence
            self.sense_pressure("error_tolerance", 1.0)

        return result

    def _process_pattern(self, binary_data: bytes) -> Optional[bytes]:
        """Process binary pattern."""
        try:
            # Simple processing: reverse bits in each byte
            processed = bytearray()
            for byte in binary_data:
                # Reverse bits in byte
                reversed_byte = int(format(byte, "08b")[::-1], 2)
                processed.append(reversed_byte)

            return bytes(processed)

        except Exception as e:
            print(f"Error processing pattern: {str(e)}")
            return None

    def _calculate_harmony(self, data: bytes) -> float:
        """Calculate natural harmony of processed data."""
        if not data:
            return 0.0

        try:
            # Simple harmony metric: byte value distribution
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1

            # Calculate entropy-based harmony
            total_bytes = len(data)
            harmony = 0.0

            for count in byte_counts:
                if count > 0:
                    probability = count / total_bytes
                    harmony += probability * (1 - probability)

            return min(1.0, harmony * 2)  # Scale to 0-1

        except Exception as e:
            print(f"Error calculating harmony: {str(e)}")
            return 0.0
