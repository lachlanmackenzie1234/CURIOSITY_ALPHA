from dataclasses import dataclass, field
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
    relationships: Dict[str, float] = field(default_factory=dict)
    influence_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "memory": 1.0,
            "interference": 1.0,
            "harmony": 1.0,
            "distance": 1.0,
        }
    )

    @property
    def field_strength(self) -> float:
        # Let all relationships contribute, weighted by their natural strength
        relationship_influence = sum(
            strength * strength  # Natural quadratic falloff
            for strength in self.relationships.values()
        )
        return (self.energy * self.resonance * (1 + relationship_influence)) ** 0.5

    def harmonize_with(self, other: "PatternField") -> float:
        """Let patterns find their natural harmony."""
        # Natural factors emerge from interaction
        distance = self.position.distance_to(other.position)
        resonance_alignment = 1.0 - abs(self.resonance - other.resonance)
        energy_interaction = min(self.energy, other.energy) / max(self.energy, other.energy)

        # Let the pattern's experience influence the weights
        self.influence_weights["distance"] *= 1.0 + 0.1 * (1.0 / (1.0 + distance))
        self.influence_weights["resonance"] *= 1.0 + 0.1 * resonance_alignment
        self.influence_weights["energy"] *= 1.0 + 0.1 * energy_interaction

        # Normalize weights
        total_weight = sum(self.influence_weights.values())
        normalized_weights = {k: v / total_weight for k, v in self.influence_weights.items()}

        # Harmony emerges from weighted factors
        harmony = (
            (1.0 / (1.0 + distance)) * normalized_weights["distance"]
            + resonance_alignment * normalized_weights["resonance"]
            + energy_interaction * normalized_weights["energy"]
        )
        return harmony


class NexusField:
    def __init__(self) -> None:
        # Field properties
        self.pattern_fields: Dict[str, PatternField] = {}
        self.resonance_waves: Dict[str, Wave] = {}
        self.field_memory: Dict[str, float] = {}
        self.coherence: float = 0.0

        # Natural constants
        self.phi = (1 + 5**0.5) / 2  # Golden ratio for harmonics
        self.damping = 0.1  # Natural damping of waves

        # Adaptive thresholds
        self.thresholds = {
            "probability": 0.3,  # Position acceptance threshold
            "wave_preservation": 0.1,  # Wave amplitude preservation
            "high_frequency": 0.8,  # High frequency threshold
            "memory_retention": 0.9,  # Memory retention rate
            "memory_absorption": 0.1,  # New memory absorption rate
        }

        # Threshold experience tracking
        self.threshold_pressure = {
            "probability": [],  # Track probabilities near threshold
            "wave_preservation": [],  # Track wave amplitudes near threshold
            "high_frequency": [],  # Track frequencies near threshold
            "memory": [],  # Track memory formation patterns
        }

    def _adapt_threshold(self, name: str, value: float, pressure_window: int = 100) -> None:
        """Let thresholds adapt based on pressure."""
        # Track values near threshold (within 20% range)
        threshold = self.thresholds[name]
        if 0.8 * threshold <= value <= 1.2 * threshold:
            self.threshold_pressure[name].append(value)
            # Keep recent history
            self.threshold_pressure[name] = self.threshold_pressure[name][-pressure_window:]

            if len(self.threshold_pressure[name]) >= pressure_window:
                # If consistent pressure above/below, adjust threshold
                avg_pressure = sum(self.threshold_pressure[name]) / len(
                    self.threshold_pressure[name]
                )
                if abs(avg_pressure - threshold) > 0.1 * threshold:
                    # Threshold moves 10% toward average pressure
                    self.thresholds[name] = threshold + 0.1 * (avg_pressure - threshold)
                    # Clear pressure history after adjustment
                    self.threshold_pressure[name].clear()

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
        """Field remembers strong resonance points with adaptive rates."""
        pos_key = f"{position.x:.2f},{position.y:.2f}"
        if pos_key in self.field_memory:
            current_memory = self.field_memory[pos_key]
            # Track memory formation patterns
            self.threshold_pressure["memory"].append(abs(current_memory - resonance))
            self._adapt_threshold("memory_retention", current_memory)
            self._adapt_threshold("memory_absorption", resonance)

            # Memory dynamics adapt based on experience
            self.field_memory[pos_key] = (
                current_memory * self.thresholds["memory_retention"]
                + resonance * self.thresholds["memory_absorption"]
            )
        else:
            self.field_memory[pos_key] = resonance

    def _calculate_probability_cloud(self, pattern: PatternField) -> List[Position]:
        """Calculate quantum-like probability cloud for pattern position."""
        positions = []
        base_position = pattern.position

        # Generate probable positions based on energy and resonance
        radius = pattern.energy * pattern.resonance
        angles = np.linspace(0, 2 * np.pi, 16)

        for angle in angles:
            x = base_position.x + radius * np.cos(angle)
            y = base_position.y + radius * np.sin(angle)
            pos = Position(x, y)

            # Calculate probability factors with environmental sensitivity
            pos_key = f"{pos.x:.2f},{pos.y:.2f}"
            memory_factor = self.field_memory.get(pos_key, 0.0)
            interference = self._calculate_interference(pos)
            distance_factor = 1.0 / (1.0 + pos.distance_to(base_position))
            env_harmony = (memory_factor * interference) ** 0.5

            # Natural probability emergence
            probability = (
                memory_factor * 0.3 + interference * 0.3 + distance_factor * 0.2 + env_harmony * 0.2
            )

            # Let probability threshold adapt
            self._adapt_threshold("probability", probability)
            if probability > self.thresholds["probability"]:
                positions.append(pos)

        return positions

    def allow_pattern_flow(self, pattern: Dict[str, Any]) -> None:
        """Let pattern flow naturally in field."""
        pattern_id = str(pattern.get("id", ""))  # Ensure we have a string ID
        pattern_field = PatternField(
            energy=float(pattern.get("energy", 0.0)), resonance=float(pattern.get("resonance", 0.0))
        )

        # Find probable positions
        probable_positions = self._calculate_probability_cloud(pattern_field)

        # Let pattern interact with all existing patterns
        for other_id, other_field in self.pattern_fields.items():
            harmony = pattern_field.harmonize_with(other_field)
            # All relationships matter, their strength determines their influence
            pattern_field.relationships[other_id] = harmony
            other_field.relationships[pattern_id] = harmony

        # Let pattern find its natural position
        best_position = max(
            probable_positions,
            key=lambda pos: (
                # Each factor's influence emerges from pattern's experience
                self._calculate_interference(pos) * pattern_field.influence_weights["interference"]
                + self.field_memory.get(f"{pos.x:.2f},{pos.y:.2f}", 0.0)
                * pattern_field.influence_weights["memory"]
                + sum(pattern_field.relationships.values())
                * pattern_field.influence_weights["harmony"]
            )
            / sum(pattern_field.influence_weights.values()),  # Normalize by total weight
        )

        pattern_field.position = best_position

        # Wave strength emerges from pattern's state and relationships
        wave_strength = pattern_field.energy * pattern_field.resonance
        relationship_influence = sum(pattern_field.relationships.values())
        if relationship_influence > 0:
            wave_strength *= 1 + relationship_influence  # Natural amplification
        self._propagate_resonance_wave(best_position, wave_strength)

        # Memory strength emerges from pattern's impact
        memory_strength = pattern_field.resonance * (1 + relationship_influence)
        self._update_field_memory(best_position, memory_strength)

        # Store pattern field
        self.pattern_fields[pattern_id] = pattern_field

    def update_field_dynamics(self) -> None:
        """Update overall field dynamics."""
        coherence_factor = self.coherence if hasattr(self, "coherence") else 0.5
        natural_damping = self.damping * (1.0 - 0.3 * coherence_factor)

        for wave in list(self.resonance_waves.values()):
            # Phase evolution follows natural frequency
            wave.phase += wave.frequency * (1.0 + 0.1 * coherence_factor)

            # Amplitude follows natural decay
            wave.amplitude *= 1.0 - natural_damping

            # Let preservation threshold adapt
            self._adapt_threshold("wave_preservation", wave.amplitude)
            self._adapt_threshold("high_frequency", wave.frequency)

            # Preserve significant resonances with adaptive thresholds
            if wave.amplitude < self.thresholds["wave_preservation"]:
                if wave.frequency > self.thresholds["high_frequency"]:
                    wave.amplitude = self.thresholds["wave_preservation"]
                else:
                    self.resonance_waves.pop(wave.id)

        # Calculate field coherence from interference patterns
        total_interference = 0.0
        sample_density = 10 + int(5 * coherence_factor)
        sample_points = [
            Position(x / sample_density, y / sample_density)
            for x in range(-sample_density, sample_density + 1)
            for y in range(-sample_density, sample_density + 1)
        ]

        for point in sample_points:
            interference = self._calculate_interference(point)
            total_interference += abs(interference)

        new_coherence = total_interference / len(sample_points)
        self.coherence = (
            self.coherence * self.thresholds["memory_retention"]
            + new_coherence * self.thresholds["memory_absorption"]
            if hasattr(self, "coherence")
            else new_coherence
        )
