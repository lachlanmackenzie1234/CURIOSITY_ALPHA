"""Pattern evolution and adaptation system."""

import array
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import time
import math


@dataclass
class NaturalPattern:
    """Represents a fundamental natural pattern."""
    name: str
    confidence: float
    ratio: float
    sequence: Optional[List[float]] = None
    properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvolutionState:
    """Track evolution state and metrics."""
    pattern_id: str
    success_count: int = 0
    failure_count: int = 0
    adaptation_history: List[float] = field(default_factory=list)
    improvement_history: List[float] = field(default_factory=list)
    last_success_time: float = 0.0
    recovery_attempts: int = 0
    stability_score: float = 1.0
    natural_patterns: List[NaturalPattern] = field(default_factory=list)


class PatternEvolution:
    """Manages pattern evolution and adaptation."""
    
    def __init__(self):
        """Initialize pattern evolution system."""
        self.logger = logging.getLogger('pattern_evolution')
        self.evolution_metrics: Dict[str, float] = {
            'success_rate': 0.0,
            'adaptation_rate': 0.0,
            'improvement_rate': 0.0,
            'stability': 1.0,
            'natural_alignment': 0.0
        }
        self.states: Dict[str, EvolutionState] = {}
        self.error_threshold = 0.3
        self.adaptation_threshold = 0.7
        self.recovery_limit = 3
        
        # Natural constants
        self.GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
        self.FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        self.E = math.e
        self.PI = math.pi
    
    def _detect_natural_patterns(
        self,
        data: np.ndarray
    ) -> List[NaturalPattern]:
        """Detect natural patterns in data."""
        patterns = []
        
        # Check for golden ratio patterns
        golden_confidence = self._check_golden_ratio(data)
        if golden_confidence > 0.6:
            patterns.append(NaturalPattern(
                name="golden_ratio",
                confidence=golden_confidence,
                ratio=self.GOLDEN_RATIO,
                properties={'harmony': golden_confidence}
            ))
        
        # Check for Fibonacci patterns
        fib_confidence, fib_seq = self._check_fibonacci(data)
        if fib_confidence > 0.6:
            patterns.append(NaturalPattern(
                name="fibonacci",
                confidence=fib_confidence,
                ratio=self.GOLDEN_RATIO,
                sequence=fib_seq,
                properties={'growth': fib_confidence}
            ))
        
        # Check for exponential patterns (e-based)
        exp_confidence = self._check_exponential(data)
        if exp_confidence > 0.6:
            patterns.append(NaturalPattern(
                name="exponential",
                confidence=exp_confidence,
                ratio=self.E,
                properties={'growth_rate': exp_confidence}
            ))
        
        # Check for circular/periodic patterns
        periodic_confidence = self._check_periodic(data)
        if periodic_confidence > 0.6:
            patterns.append(NaturalPattern(
                name="periodic",
                confidence=periodic_confidence,
                ratio=self.PI,
                properties={'cyclical': periodic_confidence}
            ))
        
        return patterns
    
    def _check_golden_ratio(self, data: np.ndarray) -> float:
        """Check for golden ratio patterns."""
        try:
            if len(data) < 2:
                return 0.0
            
            # Convert to float64 for calculations
            arr = data.astype(np.float64) + 1e-10
            
            # Calculate ratios between consecutive elements
            # Handle modulo arithmetic by considering all possible ratios
            ratios = []
            for i in range(len(arr) - 1):
                a = arr[i]
                b = arr[i + 1]
                
                # Consider original ratio
                ratio1 = b / a
                ratios.append(abs(ratio1 - ((1 + np.sqrt(5)) / 2)))
                
                # Consider ratio with modulo unwrapping
                ratio2 = (b + 256) / a
                ratios.append(abs(ratio2 - ((1 + np.sqrt(5)) / 2)))
                
                ratio3 = b / (a + 256)
                ratios.append(abs(ratio3 - ((1 + np.sqrt(5)) / 2)))
            
            # Find best matching ratio
            min_deviation = min(ratios)
            confidence = 1.0 / (1.0 + min_deviation)
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error checking golden ratio: {str(e)}")
            return 0.0
    
    def _check_fibonacci(
        self,
        data: np.ndarray
    ) -> Tuple[float, Optional[List[float]]]:
        """Check for Fibonacci sequence patterns."""
        try:
            if len(data) < 3:
                return 0.0, None
            
            # Convert to float64 for calculations
            arr = data.astype(np.float64) + 1e-10
            
            # Check if consecutive elements follow Fibonacci rule
            a = arr[:-2]  # First number
            b = arr[1:-1]  # Second number
            c = arr[2:]    # Sum
            
            # Calculate deviations from Fibonacci rule
            expected = a + b
            deviations = np.abs(c - expected)
            
            # Calculate confidence based on deviations
            matches = deviations < 3
            if not np.any(matches):
                return 0.0, None
            
            confidence = float(np.mean(matches))
            sequence = data.tolist()
            
            return confidence, sequence
            
        except Exception as e:
            self.logger.error(f"Error checking Fibonacci: {str(e)}")
            return 0.0, None
    
    def _check_exponential(self, data: np.ndarray) -> float:
        """Check for exponential growth/decay patterns."""
        try:
            if len(data) < 3:
                return 0.0
            
            # Calculate rate of change
            rates = []
            for i in range(len(data) - 1):
                if data[i] != 0:
                    rate = data[i + 1] / data[i]
                    rates.append(rate)
            
            if not rates:
                return 0.0
            
            # Check consistency of growth/decay rate
            rate_std = np.std(rates)
            confidence = 1.0 / (1.0 + rate_std)
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error checking exponential pattern: {str(e)}")
            return 0.0
    
    def _check_periodic(self, data: np.ndarray) -> float:
        """Check for periodic/circular patterns."""
        try:
            if len(data) < 4:
                return 0.0
            
            # Use FFT to detect periodic components
            fft = np.fft.fft(data)
            
            # Look at dominant frequencies
            main_freq_idx = np.argmax(np.abs(fft))
            
            # Calculate periodicity confidence
            power_spectrum = np.abs(fft) ** 2
            total_power = np.sum(power_spectrum)
            main_power = power_spectrum[main_freq_idx]
            
            if total_power == 0:
                return 0.0
            
            confidence = main_power / total_power
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error checking periodic pattern: {str(e)}")
            return 0.0
    
    def _calculate_pattern_metrics(
        self,
        pattern_data: array.array,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive pattern metrics with error handling."""
        metrics = self.evolution_metrics.copy()
        pattern_id = context.get('pattern_id', '')
        
        try:
            # Initialize state if needed
            if pattern_id not in self.states:
                self.states[pattern_id] = EvolutionState(pattern_id)
            
            state = self.states[pattern_id]
            data = np.frombuffer(pattern_data, dtype=np.uint8)
            
            # Calculate base metrics with validation
            if not self._validate_pattern_data(data):
                raise ValueError("Invalid pattern data")
            
            # Detect natural patterns
            natural_patterns = self._detect_natural_patterns(data)
            state.natural_patterns = natural_patterns
            
            # Calculate natural alignment score
            if natural_patterns:
                metrics['natural_alignment'] = max(
                    p.confidence for p in natural_patterns
                )
            
            # Calculate success rate with natural pattern influence
            if 'expected_behavior' in context:
                behavior = context['expected_behavior']
                success_rate = self._calculate_success_rate(
                    data,
                    behavior,
                    natural_patterns
                )
                metrics['success_rate'] = success_rate
                
                if success_rate >= self.adaptation_threshold:
                    state.success_count += 1
                    state.last_success_time = time.time()
                else:
                    state.failure_count += 1
            
            # Calculate adaptation rate
            adaptation_rate = self._calculate_adaptation_rate(
                pattern_id,
                pattern_data
            )
            metrics['adaptation_rate'] = adaptation_rate
            state.adaptation_history.append(adaptation_rate)
            
            # Calculate improvement rate
            improvement_rate = self._calculate_improvement_rate(
                data,
                natural_patterns
            )
            metrics['improvement_rate'] = improvement_rate
            state.improvement_history.append(improvement_rate)
            
            # Update stability score
            state.stability_score = self._calculate_stability(
                data,
                natural_patterns
            )
            metrics['stability'] = state.stability_score
            
            return metrics
            
        except Exception as e:
            self.logger.error(
                f"Error calculating metrics for pattern {pattern_id}: {str(e)}"
            )
            self._handle_evolution_error(pattern_id)
            return self._get_safe_metrics()
    
    def _validate_pattern_data(self, data: np.ndarray) -> bool:
        """Validate pattern data integrity."""
        try:
            return (
                len(data) > 0 and
                not np.any(np.isnan(data)) and
                not np.any(np.isinf(data))
            )
        except Exception:
            return False
    
    def _calculate_success_rate(
        self,
        data: np.ndarray,
        behavior: Dict[str, float],
        natural_patterns: List[NaturalPattern]
    ) -> float:
        """Calculate success rate based on expected behavior."""
        try:
            regularity = self._calculate_regularity(data)
            symmetry = self._calculate_symmetry(data)
            complexity = self._calculate_complexity(data)
            
            # Calculate natural pattern influence
            natural_influence = 0.0
            for pattern in natural_patterns:
                natural_influence += pattern.confidence
            
            return (
                behavior.get('regularity', 0.5) * regularity +
                behavior.get('symmetry', 0.5) * symmetry +
                behavior.get('complexity', 0.5) * complexity +
                natural_influence * 0.2
            ) / 3
            
        except Exception as e:
            self.logger.error(f"Error calculating success rate: {str(e)}")
            return 0.0
    
    def _calculate_adaptation_rate(
        self,
        pattern_id: str,
        pattern_data: array.array
    ) -> float:
        """Calculate adaptation rate with error handling."""
        try:
            state = self.states.get(pattern_id)
            if not state or not state.adaptation_history:
                return 0.3  # Default rate
            
            # Calculate rate based on recent history
            recent_rates = state.adaptation_history[-10:]
            current_rate = np.mean(recent_rates)
            
            # Adjust based on success/failure ratio
            total_attempts = state.success_count + state.failure_count
            if total_attempts > 0:
                success_ratio = state.success_count / total_attempts
                current_rate *= (0.5 + 0.5 * success_ratio)
            
            return float(np.clip(current_rate, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptation rate: {str(e)}")
            return 0.3
    
    def _calculate_improvement_rate(
        self,
        data: np.ndarray,
        natural_patterns: List[NaturalPattern]
    ) -> float:
        """Calculate improvement rate with validation."""
        try:
            if not self._validate_pattern_data(data):
                return 0.0
            
            # Calculate improvement factors
            complexity = self._calculate_complexity(data)
            consistency = self._calculate_consistency(data)
            error_resistance = self._calculate_error_resistance(data)
            
            # Combine factors
            return float(
                complexity * 0.3 +
                consistency * 0.4 +
                error_resistance * 0.3
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating improvement rate: {str(e)}")
            return 0.0
    
    def _should_trigger_recovery(self, state: EvolutionState) -> bool:
        """Determine if pattern needs recovery."""
        try:
            # Check failure threshold
            total_attempts = state.success_count + state.failure_count
            if total_attempts > 0:
                failure_rate = state.failure_count / total_attempts
                if failure_rate > self.error_threshold:
                    return True
            
            # Check stability
            if state.stability_score < self.error_threshold:
                return True
            
            # Check adaptation trend
            if len(state.adaptation_history) >= 3:
                recent = state.adaptation_history[-3:]
                if all(rate < self.error_threshold for rate in recent):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking recovery trigger: {str(e)}")
            return False
    
    def _attempt_pattern_recovery(
        self,
        state: EvolutionState,
        pattern_data: array.array
    ) -> None:
        """Attempt to recover pattern stability."""
        try:
            state.recovery_attempts += 1
            msg = (
                f"Attempting recovery for pattern {state.pattern_id} "
                f"(Attempt {state.recovery_attempts})"
            )
            self.logger.warning(msg)
            
            if state.recovery_attempts > self.recovery_limit:
                msg = f"Recovery limit reached for pattern {state.pattern_id}"
                self.logger.error(msg)
                self._reset_pattern_state(state.pattern_id)
                return
            
            # Try to stabilize pattern
            data = np.frombuffer(pattern_data, dtype=np.uint8)
            if self._validate_pattern_data(data):
                # Apply correction strategies
                self._apply_stability_corrections(state, data)
            
        except Exception as e:
            self.logger.error(f"Error in pattern recovery: {str(e)}")
    
    def _apply_stability_corrections(
        self,
        state: EvolutionState,
        data: np.ndarray
    ) -> None:
        """Apply corrections to improve pattern stability."""
        try:
            # Reset metrics if severely unstable
            if state.stability_score < 0.2:
                state.adaptation_history = []
                state.improvement_history = []
                state.success_count = 0
                state.failure_count = 0
            
            # Adjust adaptation threshold
            if state.stability_score < 0.5:
                self.adaptation_threshold = max(
                    0.5,
                    self.adaptation_threshold * 0.9
                )
            
        except Exception as e:
            self.logger.error(f"Error applying corrections: {str(e)}")
    
    def _handle_evolution_error(self, pattern_id: str) -> None:
        """Handle evolution system errors."""
        try:
            self.logger.error(f"Evolution error for pattern {pattern_id}")
            state = self.states.get(pattern_id)
            
            if state and state.recovery_attempts > self.recovery_limit:
                self._reset_pattern_state(pattern_id)
            
        except Exception as e:
            self.logger.critical(f"Error in error handler: {str(e)}")
    
    def _reset_pattern_state(self, pattern_id: str) -> None:
        """Reset pattern state after multiple failures."""
        try:
            self.states[pattern_id] = EvolutionState(pattern_id)
            self.logger.info(f"Reset state for pattern {pattern_id}")
            
        except Exception as e:
            self.logger.critical(f"Error resetting pattern state: {str(e)}")
    
    def _get_safe_metrics(self) -> Dict[str, float]:
        """Return safe default metrics for error cases."""
        return {
            'success_rate': 0.0,
            'adaptation_rate': 0.3,
            'improvement_rate': 0.0,
            'stability': 0.5
        }
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate normalized entropy of the data."""
        try:
            # Calculate frequency distribution
            unique, counts = np.unique(data, return_counts=True)
            probabilities = counts / len(data)
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))
            # Max entropy for byte values
            max_entropy = np.log2(256)
            
            return entropy / max_entropy
        except Exception:
            return 0.0
    
    def _calculate_symmetry(self, data: np.ndarray) -> float:
        """Calculate pattern symmetry."""
        try:
            length = len(data)
            if length < 2:
                return 1.0
            
            # Check for mirror symmetry
            mid = length // 2
            left = data[:mid]
            right = data[length - mid:][::-1]  # Reverse right half
            
            matches = np.sum(left == right)
            return float(matches / len(left))
            
        except Exception:
            return 0.0
    
    def _calculate_consistency(self, data: np.ndarray) -> float:
        """Calculate pattern consistency."""
        try:
            if len(data) < 2:
                return 1.0
            
            # Calculate standard deviation of differences
            differences = np.diff(data)
            std_dev = np.std(differences)
            # Max std dev for byte values
            max_std = 255
            
            return 1.0 - (std_dev / max_std)
            
        except Exception:
            return 0.0
    
    def _calculate_error_resistance(self, data: np.ndarray) -> float:
        """Calculate error resistance capability."""
        try:
            # Calculate based on pattern redundancy and structure
            entropy = self._calculate_entropy(data)
            consistency = self._calculate_consistency(data)
            
            # Higher entropy -> lower error resistance
            # Higher consistency -> higher error resistance
            return (1 - entropy) * 0.3 + consistency * 0.7
            
        except Exception:
            return 0.0
    
    def _calculate_regularity(self, data: np.ndarray) -> float:
        """Calculate pattern regularity."""
        try:
            if len(data) < 2:
                return 1.0
            
            # Look for repeating subsequences
            regularity_scores = []
            for length in range(1, min(len(data) // 2 + 1, 8)):
                chunks = [
                    tuple(data[i:i + length])
                    for i in range(0, len(data) - length + 1)
                ]
                unique_chunks = len(set(chunks))
                regularity_scores.append(1 - (unique_chunks / len(chunks)))
            
            return max(regularity_scores) if regularity_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_complexity(self, data: np.ndarray) -> float:
        """Calculate pattern complexity."""
        try:
            # Combine multiple complexity metrics
            entropy = self._calculate_entropy(data)
            unique_ratio = len(set(data)) / len(data)
            
            return (entropy + unique_ratio) / 2
            
        except Exception:
            return 0.0
    
    def _calculate_stability(
        self,
        data: np.ndarray,
        natural_patterns: Optional[List[NaturalPattern]] = None
    ) -> float:
        """Calculate pattern stability."""
        try:
            # Base stability metrics
            consistency = self._calculate_consistency(data)
            error_resistance = self._calculate_error_resistance(data)
            base_stability = (consistency + error_resistance) / 2
            
            # If natural patterns exist, they contribute to stability
            if natural_patterns:
                natural_stability = sum(
                    p.confidence for p in natural_patterns
                ) / len(natural_patterns)
                return (base_stability * 0.7 + natural_stability * 0.3)
            
            return base_stability
            
        except Exception:
            return 0.0
    
    def _analyze_failure_pattern(
        self,
        state: EvolutionState,
        data: np.ndarray
    ) -> str:
        """Analyze pattern failure type."""
        # Check stability failure
        if state.stability_score < 0.3:
            return "stability_failure"
            
        # Check adaptation failure
        if len(state.adaptation_history) >= 3:
            recent = state.adaptation_history[-3:]
            if all(rate < 0.3 for rate in recent):
                return "adaptation_failure"
                
        # Check pattern degradation
        if state.natural_patterns:
            current_patterns = self._detect_natural_patterns(data)
            if len(current_patterns) < len(state.natural_patterns):
                return "pattern_degradation"
                
        return "unknown_failure"

    def _apply_general_recovery(self, state: EvolutionState) -> bool:
        """Apply general recovery strategy."""
        try:
            # Reset thresholds to default values
            self.error_threshold = 0.3
            self.adaptation_threshold = 0.7
            
            # Trim history to prevent old data influence
            if len(state.adaptation_history) > 3:
                state.adaptation_history = state.adaptation_history[-3:]
            if len(state.improvement_history) > 3:
                state.improvement_history = state.improvement_history[-3:]
                
            return True
            
        except Exception as e:
            self.logger.error(f"General recovery failed: {str(e)}")
            return False

    def _restore_natural_patterns(
        self,
        state: EvolutionState,
        data: np.ndarray
    ) -> bool:
        """Attempt to restore natural patterns."""
        try:
            # Get current patterns
            current_patterns = self._detect_natural_patterns(data)
            
            # Compare with stored patterns
            if not state.natural_patterns:
                return False
                
            # Try to restore each missing pattern
            for stored in state.natural_patterns:
                if not any(p.name == stored.name for p in current_patterns):
                    self.logger.info(
                        f"Attempting to restore {stored.name} pattern"
                    )
                    # Pattern restoration logic would go here
                    # For now, just mark the attempt
                    state.recovery_attempts += 1
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Pattern restoration failed: {str(e)}")
            return False

    def _apply_recovery_strategy(
        self,
        state: EvolutionState,
        data: np.ndarray,
        failure_type: str
    ) -> bool:
        """Apply specific recovery strategy based on failure type."""
        try:
            if failure_type == "stability_failure":
                # Lower thresholds temporarily
                self.error_threshold *= 0.8
                self.adaptation_threshold *= 0.8
                return True
                
            elif failure_type == "adaptation_failure":
                # Reset adaptation history
                state.adaptation_history.clear()
                # Lower adaptation threshold
                self.adaptation_threshold *= 0.9
                return True
                
            elif failure_type == "pattern_degradation":
                return self._restore_natural_patterns(state, data)
                
            return self._apply_general_recovery(state)
            
        except Exception as e:
            self.logger.error(
                f"Recovery strategy failed for {failure_type}: {str(e)}"
            )
            return False 