#!/usr/bin/env python3
"""ALPHA monitoring system."""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    pattern_processing_rate: float = 0.0
    error_rate: float = 0.0
    stability_score: float = 1.0
    health_score: float = 1.0
    metrics_history: List[Dict[str, float]] = field(default_factory=list)

class MemoryMonitor:
    """Monitor ALPHA system state and performance."""

    def __init__(self):
        """Initialize monitor."""
        self.logger = logging.getLogger('alpha_monitor')
        self.logger.setLevel(logging.INFO)
        self.active = False
        self.metrics = SystemMetrics()
        self.start_time = time.time()
        self.error_threshold = 0.1
        self.warning_threshold = 0.3
        self.check_interval = 60  # seconds
        self.last_check = time.time()
        self.recovery_attempts = 0
        
    def start(self) -> None:
        """Start monitoring."""
        self.active = True
        self.start_time = time.time()
        self.logger.info("Monitoring started")
        self._initialize_baseline()
    
    def stop(self) -> None:
        """Stop monitoring."""
        self.active = False
        self.logger.info("Monitoring stopped")
        self._save_metrics()
    
    def is_active(self) -> bool:
        """Check if monitoring is active."""
        return self.active
    
    def update_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        pattern_rate: float,
        error_rate: float
    ) -> None:
        """Update system metrics."""
        try:
            # Update current metrics
            self.metrics.cpu_usage = cpu_usage
            self.metrics.memory_usage = memory_usage
            self.metrics.pattern_processing_rate = pattern_rate
            self.metrics.error_rate = error_rate
            
            # Calculate stability score
            self.metrics.stability_score = self._calculate_stability()
            
            # Calculate overall health
            self.metrics.health_score = self._calculate_health()
            
            # Store metrics history
            self.metrics.metrics_history.append({
                'timestamp': time.time(),
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'pattern_rate': pattern_rate,
                'error_rate': error_rate,
                'stability': self.metrics.stability_score,
                'health': self.metrics.health_score
            })
            
            # Check system health
            self._check_system_health()
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            self._handle_monitoring_error()
    
    def get_system_status(self) -> Dict[str, float]:
        """Get current system status."""
        return {
            'uptime': time.time() - self.start_time,
            'cpu_usage': self.metrics.cpu_usage,
            'memory_usage': self.metrics.memory_usage,
            'pattern_processing_rate': self.metrics.pattern_processing_rate,
            'error_rate': self.metrics.error_rate,
            'stability_score': self.metrics.stability_score,
            'health_score': self.metrics.health_score
        }
    
    def _initialize_baseline(self) -> None:
        """Initialize baseline metrics."""
        try:
            self.baseline_metrics = {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'pattern_rate': 0.0,
                'error_rate': 0.0
            }
            self.logger.info("Baseline metrics initialized")
        except Exception as e:
            self.logger.error(f"Error initializing baseline: {str(e)}")
    
    def _calculate_stability(self) -> float:
        """Calculate system stability score."""
        try:
            if not self.metrics.metrics_history:
                return 1.0
            
            # Get recent metrics
            recent = self.metrics.metrics_history[-10:]
            
            # Calculate stability factors
            cpu_stability = 1.0 - np.std([m['cpu_usage'] for m in recent]) / 100
            memory_stability = 1.0 - np.std([m['memory_usage'] for m in recent]) / 100
            error_stability = 1.0 - np.mean([m['error_rate'] for m in recent])
            
            # Combine factors
            stability = (
                cpu_stability * 0.3 +
                memory_stability * 0.3 +
                error_stability * 0.4
            )
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            self.logger.error(f"Error calculating stability: {str(e)}")
            return 0.0
    
    def _calculate_health(self) -> float:
        """Calculate overall system health score."""
        try:
            # Weight factors
            cpu_factor = max(0.0, 1.0 - self.metrics.cpu_usage / 100)
            memory_factor = max(0.0, 1.0 - self.metrics.memory_usage / 100)
            error_factor = max(0.0, 1.0 - self.metrics.error_rate)
            stability_factor = self.metrics.stability_score
            
            # Calculate weighted health score
            health = (
                cpu_factor * 0.25 +
                memory_factor * 0.25 +
                error_factor * 0.25 +
                stability_factor * 0.25
            )
            
            return max(0.0, min(1.0, health))
            
        except Exception as e:
            self.logger.error(f"Error calculating health: {str(e)}")
            return 0.0
    
    def _check_system_health(self) -> None:
        """Check system health and trigger alerts if needed."""
        try:
            current_time = time.time()
            if current_time - self.last_check < self.check_interval:
                return
                
            self.last_check = current_time
            
            # Check critical metrics
            if self.metrics.error_rate > self.error_threshold:
                self.logger.error(
                    f"High error rate detected: {self.metrics.error_rate:.2%}"
                )
                self._trigger_recovery()
            
            if self.metrics.stability_score < self.warning_threshold:
                self.logger.warning(
                    f"Low stability detected: {self.metrics.stability_score:.2f}"
                )
                
            if self.metrics.health_score < self.warning_threshold:
                self.logger.warning(
                    f"Poor system health: {self.metrics.health_score:.2f}"
                )
                
        except Exception as e:
            self.logger.error(f"Error in health check: {str(e)}")
    
    def _handle_monitoring_error(self) -> None:
        """Handle errors in the monitoring system."""
        try:
            self.logger.error("Monitoring system error detected")
            self.recovery_attempts += 1
            
            if self.recovery_attempts > 3:
                self.logger.critical(
                    "Multiple monitoring errors. Initiating system reset."
                )
                self._reset_monitoring()
            
        except Exception as e:
            self.logger.critical(f"Critical error in error handler: {str(e)}")
    
    def _reset_monitoring(self) -> None:
        """Reset monitoring system to recover from errors."""
        try:
            self.metrics = SystemMetrics()
            self.recovery_attempts = 0
            self.last_check = time.time()
            self._initialize_baseline()
            self.logger.info("Monitoring system reset complete")
            
        except Exception as e:
            self.logger.critical(f"Failed to reset monitoring: {str(e)}")
    
    def _save_metrics(self) -> None:
        """Save metrics history."""
        try:
            if self.metrics.metrics_history:
                # Save metrics to persistent storage
                self.logger.info(
                    f"Saving {len(self.metrics.metrics_history)} metric records"
                )
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")


def create_monitor() -> MemoryMonitor:
    """Create and return a new monitor instance."""
    return MemoryMonitor() 