"""Field Observer - Automatic threshold detection and adaptation propagation."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from ALPHA.NEXUS.core.adaptive_field import AdaptiveField


@dataclass
class PressurePoint:
    """A detected point of pressure in the system."""

    component_id: str
    value_name: str
    typical_range: tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    pressure_count: int = 0
    last_value: float = 0.0
    is_threshold: bool = False


class FieldObserver(AdaptiveField):
    """Observes and manages pressure points in the system."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("field_observer")
        self._pressure_points: Dict[str, PressurePoint] = {}
        self._component_connections: Dict[str, Set[str]] = {}
        self._observation_window: int = 100

    def observe_value(self, component_id: str, value_name: str, value: float) -> None:
        """Observe a value in the system for pressure patterns."""
        point_id = f"{component_id}:{value_name}"

        # Create or update pressure point
        if point_id not in self._pressure_points:
            self._pressure_points[point_id] = PressurePoint(
                component_id=component_id, value_name=value_name
            )

        point = self._pressure_points[point_id]
        point.last_value = value

        # Detect if this value acts like a threshold
        if self._detect_threshold_behavior(point, value):
            if not point.is_threshold:
                self._register_adaptive_threshold(point)
                point.is_threshold = True

        # Update pressure tracking
        if self._detect_pressure(point, value):
            point.pressure_count += 1
            if point.pressure_count >= self._observation_window:
                self._propagate_adaptation(point)

    def connect_components(self, source_id: str, target_id: str) -> None:
        """Register a connection between components for adaptation propagation."""
        if source_id not in self._component_connections:
            self._component_connections[source_id] = set()
        self._component_connections[source_id].add(target_id)

    def _detect_threshold_behavior(self, point: PressurePoint, value: float) -> bool:
        """Detect if a value is being used as a threshold."""
        # Value is likely a threshold if:
        # 1. It's frequently compared against
        # 2. Values cluster around it
        # 3. It creates binary outcomes
        min_val, max_val = point.typical_range
        normalized_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5

        # Update typical range
        point.typical_range = (
            min(point.typical_range[0], value),
            max(point.typical_range[1], value),
        )

        # Check for threshold-like patterns
        return (
            abs(normalized_value - 0.5) < 0.3  # Value near middle of range
            or abs(value - point.last_value) < 0.1  # Value relatively stable
        )

    def _detect_pressure(self, point: PressurePoint, value: float) -> bool:
        """Detect if there's pressure on this point."""
        if not point.is_threshold:
            return False

        # Get threshold value if this is registered
        threshold_id = f"{point.component_id}:{point.value_name}"
        threshold = self.get_threshold(threshold_id)

        # Detect pressure near threshold
        return abs(value - threshold) < 0.2 * threshold

    def _register_adaptive_threshold(self, point: PressurePoint) -> None:
        """Register a new adaptive threshold."""
        threshold_id = f"{point.component_id}:{point.value_name}"
        initial_value = (point.typical_range[0] + point.typical_range[1]) / 2
        self.register_threshold(threshold_id, initial_value)
        self.logger.info(f"Registered new adaptive threshold: {threshold_id}")

    def _propagate_adaptation(self, point: PressurePoint) -> None:
        """Propagate adaptation effects to connected components."""
        source_id = point.component_id
        if source_id in self._component_connections:
            # Get adaptation factor
            threshold_id = f"{point.component_id}:{point.value_name}"
            adaptation = self.get_threshold(threshold_id)

            # Propagate to connected components
            for target_id in self._component_connections[source_id]:
                target_threshold = f"{target_id}:{point.value_name}"
                if target_threshold in self._thresholds:
                    # Let connected threshold feel the pressure
                    self.sense_pressure(target_threshold, adaptation)

        # Reset pressure count
        point.pressure_count = 0
