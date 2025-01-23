from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class Position:
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass
class Wave:
    origin: Position
    amplitude: float
    frequency: float
    phase: float
    id: str = ""


@dataclass
class PatternField:
    energy: float
    resonance: float
    position: Position = Position(0.0, 0.0)
    relationships: Dict[str, float] = None

    def __post_init__(self):
        if self.relationships is None:
            self.relationships = {}

    @property
    def field_strength(self) -> float:
        return (self.energy * self.resonance * (1 + len(self.relationships))) ** 0.5


class NexusField:
    def __init__(self) -> None:
        # Field properties
        self.pattern_fields: Dict[str, PatternField] = {}
        self.resonance_waves: Dict[str, Wave] = {}
        self.field_memory: Dict[str, float] = {}  # Using string repr of Position
        self.coherence: float = 0.0

        # Natural constants
        self.phi = (1 + 5**0.5) / 2  # Golden ratio for harmonics
        self.damping = 0.1  # Natural damping of waves

    def _propagate_resonance_wave(self, source: Position, strength: float) -> None:
        """Let resonance spread through field like a wave."""
        wave_id = f"wave_{len(self.resonance_waves)}"
        self.resonance_waves[wave_id] = Wave(
            origin=source,
            amplitude=strength,
            frequency=strength * self.phi,  # Natural frequency based on strength
            phase=0.0,
            id=wave_id,
        )

    def _calculate_interference(self, position: Position) -> float:
        """Calculate wave interference at a point."""
        total_amplitude = 0.0
        for wave in self.resonance_waves.values():
            # Distance affects phase
            distance = position.distance_to(wave.origin)
            phase = wave.phase + (distance * wave.frequency)

            # Wave equation with natural decay
            amplitude = wave.amplitude * np.exp(-self.damping * distance) * np.cos(phase)
            total_amplitude += amplitude

        return total_amplitude

    def _update_field_memory(self, position: Position, resonance: float) -> None:
        """Field remembers strong resonance points."""
        pos_key = f"{position.x:.2f},{position.y:.2f}"
        if pos_key in self.field_memory:
            # Memory strengthens with repeated resonance
            self.field_memory[pos_key] = self.field_memory[pos_key] * 0.9 + resonance * 0.1
        else:
            self.field_memory[pos_key] = resonance

    def _calculate_probability_cloud(self, pattern: PatternField) -> List[Position]:
        """Calculate quantum-like probability cloud for pattern position."""
        positions = []
        base_position = pattern.position

        # Generate probable positions based on energy and resonance
        radius = pattern.energy * pattern.resonance
        angles = np.linspace(0, 2 * np.pi, 8)  # 8 possible positions

        for angle in angles:
            # Position probability affected by:
            # - Field memory
            # - Wave interference
            # - Distance from base
            x = base_position.x + radius * np.cos(angle)
            y = base_position.y + radius * np.sin(angle)
            pos = Position(x, y)

            # Calculate probability factors
            pos_key = f"{pos.x:.2f},{pos.y:.2f}"
            memory_factor = self.field_memory.get(pos_key, 0.0)
            interference = self._calculate_interference(pos)
            distance_factor = 1.0 / (1.0 + pos.distance_to(base_position))

            probability = (memory_factor + interference + distance_factor) / 3.0

            if probability > 0.5:  # Only keep likely positions
                positions.append(pos)

        return positions

    def allow_pattern_flow(self, pattern: Dict[str, Any]) -> None:
        """Let pattern flow naturally in field."""
        pattern_field = PatternField(
            energy=float(pattern.get("energy", 0.0)), resonance=float(pattern.get("resonance", 0.0))
        )

        # Find probable positions
        probable_positions = self._calculate_probability_cloud(pattern_field)

        # Let pattern choose position with highest resonance
        best_position = max(
            probable_positions,
            key=lambda pos: (
                self._calculate_interference(pos)
                + self.field_memory.get(f"{pos.x:.2f},{pos.y:.2f}", 0.0)
            ),
        )

        pattern_field.position = best_position

        # Propagate resonance wave from new pattern
        self._propagate_resonance_wave(
            best_position, pattern_field.energy * pattern_field.resonance
        )

        # Update field memory
        self._update_field_memory(best_position, pattern_field.resonance)

        # Store pattern field
        self.pattern_fields[pattern.get("id")] = pattern_field

    def update_field_dynamics(self) -> None:
        """Update overall field dynamics."""
        # Update waves
        for wave in list(self.resonance_waves.values()):
            wave.phase += wave.frequency
            wave.amplitude *= 1.0 - self.damping

            # Remove dissipated waves
            if wave.amplitude < 0.1:
                self.resonance_waves.pop(wave.id)

        # Calculate field coherence from interference patterns
        total_interference = 0.0
        sample_points = [
            Position(x / 10.0, y / 10.0) for x in range(-10, 11) for y in range(-10, 11)
        ]

        for point in sample_points:
            interference = self._calculate_interference(point)
            total_interference += abs(interference)

        self.coherence = total_interference / len(sample_points)
