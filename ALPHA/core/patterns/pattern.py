"""Enhanced binary pattern core for efficient processing, storage, and communication."""

from __future__ import annotations

import array
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# Natural constants for pattern harmony
FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
PI = math.pi
E = math.e


class PatternType(Enum):
    """Types of specialized patterns."""

    PROCESSOR = "processor"  # Optimized for computation
    COMMUNICATOR = "comm"  # Optimized for data transfer
    NETWORK = "network"  # Optimized for connectivity
    INTEGRATOR = "integrator"  # Optimized for system binding
    STORAGE = "storage"  # Optimized for data persistence
    COMPRESSOR = "compressor"  # Optimized for data compression
    INDEXER = "indexer"  # Optimized for data organization
    STILLNESS = "stillness"  # Optimized for stability
    FLOW = "flow"  # Optimized for adaptation
    HARMONIC = "harmonic"  # Optimized for resonance and natural alignment
    EVOLUTIONARY = "evolutionary"  # Ready for dimensional shift through harmony or interference
    TRANSCENDENT = "transcendent"  # Bridging to NEXUS through resonance or divergence


@dataclass
class StorageMetadata:
    """Metadata for storage optimization."""

    compression_ratio: float = 1.0
    access_frequency: int = 0
    last_access_time: float = 0.0
    priority_level: int = 1
    checksum: Optional[int] = None
    references: Set[str] = field(default_factory=set)
    stability: float = 0.5  # How stable the pattern is (0-1)
    resonance: Dict[str, float] = field(default_factory=dict)


@dataclass
class Pattern:
    """Core pattern class optimized for system communication and processing."""

    id: str
    pattern_type: PatternType = PatternType.PROCESSOR
    data: array.array = field(default_factory=lambda: array.array("B"))
    energy: float = 1.0
    history: List[Dict] = field(default_factory=list)
    success_rate: float = 0.5
    optimization_threshold: float = 0.7
    learning_rate: float = 0.1
    connections: Set[str] = field(default_factory=set)
    potential_forms: Set[str] = field(default_factory=set)

    # Storage-specific attributes
    storage_metadata: StorageMetadata = field(default_factory=StorageMetadata)

    # Memory Palace specific attributes
    associations: Dict[str, float] = field(default_factory=dict)  # Concept to strength mapping

    # Performance metrics
    processing_efficiency: float = 0.5
    communication_latency: float = 0.5
    network_stability: float = 0.5
    storage_efficiency: float = 0.5
    pattern_stability: float = 0.5

    # Path affinity tracking
    light_path_experiences: List[Dict[str, Any]] = field(default_factory=list)
    shadow_path_experiences: List[Dict[str, Any]] = field(default_factory=list)
    path_affinity: float = 0.5  # 0 = shadow, 1 = light, 0.5 = neutral

    def __post_init__(self):
        """Initialize pattern with empty data if none provided."""
        if not self.data:
            self.data = array.array("B")
        self._suggest_potentials()

    def _suggest_potentials(self) -> None:
        """Suggest potential forms for pattern evolution."""
        if self.pattern_type == PatternType.STILLNESS:
            self.potential_forms.update({"stable", "grounded", "centered"})
        elif self.pattern_type == PatternType.FLOW:
            self.potential_forms.update({"adaptive", "fluid", "dynamic"})
        else:
            self.potential_forms.update({"learning", "growing", "optimizing"})

    def calculate_similarity(self, other: Pattern) -> float:
        """Calculate similarity with another pattern using enhanced metrics."""
        if not self.data or not other.data:
            return 0.0

        # Binary similarity
        binary_sim = self._calculate_binary_similarity(other)

        # Pattern type compatibility
        type_sim = 1.0 if self.pattern_type == other.pattern_type else 0.5

        # Connection overlap
        common_connections = len(self.connections & other.connections)
        total_connections = len(self.connections | other.connections)
        connection_sim = common_connections / max(1, total_connections)

        # Potential form overlap
        common_forms = len(self.potential_forms & other.potential_forms)
        total_forms = len(self.potential_forms | other.potential_forms)
        form_sim = common_forms / max(1, total_forms)

        # Weighted combination
        weights = [0.4, 0.2, 0.2, 0.2]  # Binary similarity has highest weight
        return sum(
            [
                weights[0] * binary_sim,
                weights[1] * type_sim,
                weights[2] * connection_sim,
                weights[3] * form_sim,
            ]
        )

    def _calculate_binary_similarity(self, other: Pattern) -> float:
        """Calculate raw binary similarity between patterns."""
        total_bits = max(len(self.data), len(other.data))
        if total_bits == 0:
            return 1.0

        differences = 0
        for i in range(total_bits):
            if i < len(self.data) and i < len(other.data):
                differences += bin(self.data[i] ^ other.data[i]).count("1")
            else:
                differences += 8  # Assume maximum difference for missing bytes

        return 1 - (differences / (total_bits * 8))

    def update_stability(self, success: bool) -> None:
        """Update pattern stability based on success."""
        if success:
            self.pattern_stability = min(1.0, self.pattern_stability + 0.1)
            self.storage_metadata.stability = min(1.0, self.storage_metadata.stability + 0.1)
        else:
            self.pattern_stability = max(0.0, self.pattern_stability - 0.1)
            self.storage_metadata.stability = max(0.0, self.storage_metadata.stability - 0.1)

    def add_connection(self, pattern_id: str) -> None:
        """Add network connection to another pattern."""
        self.connections.add(pattern_id)
        self.network_stability = len(self.connections) / (len(self.connections) + 1)

    def optimize_for_type(self) -> None:
        """Optimize pattern structure based on its type."""
        if self.pattern_type == PatternType.STORAGE:
            self._optimize_storage()
        elif self.pattern_type == PatternType.COMPRESSOR:
            self._optimize_compression()
        elif self.pattern_type == PatternType.INDEXER:
            self._optimize_index()  # Changed to match actual method name
        else:
            self._optimize_default()  # Changed to use internal method instead of super()

    def _optimize_storage(self) -> None:
        """Optimize pattern for efficient storage."""
        if not self.data:
            return

        # Calculate optimal storage format based on data characteristics
        entropy = self.calculate_pattern_entropy()

        if entropy < 0.3:  # Highly compressible
            self._apply_run_length_encoding()
        elif entropy < 0.6:  # Moderately compressible
            self._apply_frequency_encoding()
        else:  # Complex data
            self._optimize_block_storage()

        # Update storage efficiency
        self.storage_efficiency = 1 - (
            len(self.data) / (self.storage_metadata.compression_ratio * len(self.data))
        )

        # Update metadata
        self._update_storage_metadata()

    def _apply_run_length_encoding(self) -> None:
        """Apply run-length encoding for repetitive patterns."""
        if not self.data:
            return

        encoded = array.array("B")
        count = 1
        current = self.data[0]

        for byte in self.data[1:]:
            if byte == current and count < 255:
                count += 1
            else:
                encoded.extend([count, current])
                count = 1
                current = byte

        encoded.extend([count, current])

        if len(encoded) < len(self.data):
            self.data = encoded
            self.storage_metadata.compression_ratio = len(self.data) / len(encoded)

    def _apply_frequency_encoding(self) -> None:
        """Apply frequency-based encoding for semi-structured data."""
        if not self.data:
            return

        # Count frequencies
        freq = {}
        for byte in self.data:
            freq[byte] = freq.get(byte, 0) + 1

        # Sort by frequency
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        # Create encoding table (most frequent = shortest code)
        encoding = {byte: i for i, (byte, _) in enumerate(sorted_freq)}

        # Encode data
        encoded = array.array("B")
        for byte in self.data:
            encoded.append(encoding[byte])

        if len(encoded) < len(self.data):
            self.data = encoded
            self.storage_metadata.compression_ratio = len(self.data) / len(encoded)

    def _optimize_block_storage(self) -> None:
        """Optimize storage for complex data patterns."""
        if not self.data:
            return

        block_size = 16  # Optimal block size for most patterns
        blocks = []

        # Split into blocks
        for i in range(0, len(self.data), block_size):
            block = self.data[i : i + block_size]
            if len(block) < block_size:  # Pad last block
                block.extend([0] * (block_size - len(block)))
            blocks.append(block)

        # Optimize each block
        optimized = array.array("B")
        for block in blocks:
            # Calculate block checksum
            checksum = sum(block) & 0xFF
            # Store block with checksum
            optimized.extend(block)
            optimized.append(checksum)

        self.data = optimized

    def _update_storage_metadata(self) -> None:
        """Update storage metadata based on current state."""
        self.storage_metadata.checksum = self._calculate_checksum()
        self.storage_metadata.compression_ratio = len(self.data) / (
            len(self.data) * self.storage_efficiency
        )

    def _calculate_checksum(self) -> int:
        """Calculate checksum for data integrity."""
        return sum(self.data) & 0xFFFFFFFF

    def verify_integrity(self) -> bool:
        """Verify data integrity using stored checksum."""
        return self._calculate_checksum() == self.storage_metadata.checksum

    def optimize_access_pattern(self) -> None:
        """Optimize storage based on access patterns."""
        if self.storage_metadata.access_frequency > 100:
            # Frequently accessed data - optimize for quick access
            self._optimize_for_quick_access()
        elif self.storage_metadata.access_frequency < 10:
            # Rarely accessed data - optimize for space
            self._optimize_for_space()

    def _optimize_for_quick_access(self) -> None:
        """Optimize pattern for quick access."""
        # Reorganize data for sequential access
        if len(self.data) > 1:
            # Sort blocks by access frequency
            block_size = 16
            blocks = [self.data[i : i + block_size] for i in range(0, len(self.data), block_size)]
            self.data = array.array("B")
            for block in blocks:
                self.data.extend(block)

    def _optimize_for_space(self) -> None:
        """Optimize pattern for space efficiency."""
        # Apply maximum compression for rarely accessed data
        self._apply_run_length_encoding()
        self._apply_frequency_encoding()

    def record_access(self) -> None:
        """Record data access for optimization."""
        self.storage_metadata.access_frequency += 1
        self.storage_metadata.last_access_time = time.time()

        # Trigger optimization if needed
        if self.storage_metadata.access_frequency % 10 == 0:
            self.optimize_access_pattern()

    def evolve(self, mutation_rate: Optional[float] = None) -> Optional[Pattern]:
        """Evolve pattern using type-specific optimization."""
        if not self.data:
            return None

        try:
            # Use optimized mutation rate if not specified
            if mutation_rate is None:
                mutation_rate = self.optimize_mutation_rate()

            # Create evolved copy
            evolved = Pattern(
                f"{self.id}_evolved",
                pattern_type=self.pattern_type,
                success_rate=self.success_rate,
                optimization_threshold=self.optimization_threshold,
                learning_rate=self.learning_rate,
            )
            evolved.data = array.array("B", self.data)
            evolved.connections = self.connections.copy()

            # Calculate entropy for targeted mutation
            entropy = self.calculate_pattern_entropy()

            # Type-specific evolution
            if self.pattern_type == PatternType.PROCESSOR:
                mutation_rate *= 1 - self.processing_efficiency
            elif self.pattern_type == PatternType.COMMUNICATOR:
                mutation_rate *= 1 - self.communication_latency
            elif self.pattern_type == PatternType.NETWORK:
                mutation_rate *= 1 - self.network_stability

            # Efficient mutation using bit operations and entropy
            for i in range(len(evolved.data)):
                if random.random() < mutation_rate * (1 + entropy):
                    if entropy > 0.7:  # High entropy - subtle changes
                        evolved.data[i] ^= 1 << random.randint(0, 2)
                    else:  # Low entropy - bigger changes
                        evolved.data[i] ^= 1 << random.randint(0, 7)

            # Optimize for specific type
            evolved.optimize_for_type()

            # Track evolution with optimization metrics
            self.history.append(
                {
                    "type": "evolution",
                    "energy": self.energy,
                    "mutation_rate": mutation_rate,
                    "entropy": entropy,
                    "pattern_type": self.pattern_type.value,
                    "processing_efficiency": self.processing_efficiency,
                    "communication_latency": self.communication_latency,
                    "network_stability": self.network_stability,
                    "success": None,
                }
            )

            return evolved

        except Exception as e:
            print(f"Evolution error: {e}")
            return None

    def merge(self, other: Pattern, merge_factor: float = 0.5) -> None:
        """Merge patterns with type-specific optimization."""
        min_len = min(len(self.data), len(other.data))

        # Calculate metrics
        self_entropy = self.calculate_pattern_entropy()
        other_entropy = other.calculate_pattern_entropy()

        # Type-specific merging
        if self.pattern_type == other.pattern_type:
            # Same type - optimize for that specific type
            merge_factor *= 1.2  # Increase merge factor for same type
        else:
            # Different types - create hybrid
            merge_factor *= 0.8  # Reduce merge factor for hybrid

        # Perform merge
        for i in range(min_len):
            if random.random() < merge_factor:
                self.data[i] = other.data[i]

        # Combine connections
        self.connections.update(other.connections)

        # Optimize for type
        self.optimize_for_type()

        # Record merge
        self.history.append(
            {
                "type": "merge",
                "energy": self.energy,
                "merge_factor": merge_factor,
                "entropy_diff": abs(self_entropy - other_entropy),
                "pattern_type": self.pattern_type.value,
                "success": None,
            }
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get pattern performance metrics."""
        return {
            "processing_efficiency": self.processing_efficiency,
            "communication_latency": self.communication_latency,
            "network_stability": self.network_stability,
            "energy": self.energy,
            "success_rate": self.success_rate,
            "connections": len(self.connections),
        }

    def add_segment(self, binary_data: bytes) -> None:
        """Add new binary segment efficiently."""
        for byte in binary_data:
            self.data.append(byte)

    def optimize_mutation_rate(self) -> float:
        """Dynamically adjust mutation rate based on success history."""
        if not self.history:
            return 0.1  # Default rate

        recent_successes = sum(1 for entry in self.history[-10:] if entry.get("success", False))
        success_rate = recent_successes / min(10, len(self.history))

        # Adjust mutation rate inversely to success rate
        if success_rate > self.optimization_threshold:
            return max(0.01, self.learning_rate * (1 - success_rate))
        else:
            return min(0.3, self.learning_rate * (2 - success_rate))

    def calculate_pattern_entropy(self) -> float:
        """Calculate information entropy of the pattern."""
        if not self.data:
            return 0.0

        # Count bit frequencies
        frequencies = [0] * 256
        for byte in self.data:
            frequencies[byte] += 1

        total_bytes = len(self.data)
        entropy = 0.0

        # Calculate Shannon entropy
        for freq in frequencies:
            if freq > 0:
                prob = freq / total_bytes
                entropy -= prob * math.log2(prob)

        return entropy / 8.0  # Normalize to [0,1]

    def record_success(self, success: bool) -> None:
        """Record success of last operation for optimization."""
        if self.history:
            self.history[-1]["success"] = success

            # Update success rate
            recent_success_rate = sum(
                1 for entry in self.history[-10:] if entry.get("success", False)
            ) / min(10, len(self.history))

            self.success_rate = self.success_rate * 0.9 + recent_success_rate * 0.1

            # Adjust energy based on success
            if success:
                self.energy = min(1.0, self.energy * 1.1)
            else:
                self.energy *= 0.9

    def similarity(self, other: Pattern) -> float:
        """Calculate binary similarity efficiently."""
        if not self.data or not other.data:
            return 0.0

        min_len = min(len(self.data), len(other.data))
        if min_len == 0:
            return 0.0

        matching_bits = sum(bin(a ^ b).count("1") for a, b in zip(self.data, other.data))

        return 1 - (matching_bits / (min_len * 8))

    def to_bytes(self) -> bytes:
        """Convert to bytes for translation."""
        return bytes(self.data)

    @classmethod
    def from_bytes(cls, id: str, data: bytes) -> Pattern:
        """Create pattern from bytes."""
        pattern = cls(id)
        pattern.data = array.array("B", data)
        return pattern

    @classmethod
    def from_input(cls, id: str, input_data: dict) -> Pattern:
        """Create pattern from input data dictionary."""
        pattern = cls(id)

        # Convert input values to binary
        binary = ""
        for value in input_data.values():
            binary += "1" if float(value) > 0.5 else "0"

        # Convert to bytes and store
        byte_length = len(binary) // 8
        bytes_list = []
        for i in range(byte_length):
            byte = binary[i * 8 : (i + 1) * 8]
            bytes_list.append(int(byte, 2))

        pattern.data = array.array("B", bytes_list)
        return pattern

    def is_compatible_for_improvement(self, other: Pattern) -> bool:
        """Check if two patterns are in the 'sweet spot' for improvement."""
        similarity = self.calculate_similarity(other)
        return 0.6 <= similarity <= 0.8  # Sweet spot for improvement

    def improve_with(self, other: Pattern) -> Optional[Pattern]:
        """Create improved pattern by combining with another compatible pattern."""
        if not self.is_compatible_for_improvement(other):
            return None

        # Create new pattern
        improved = Pattern(f"{self.id}_improved")
        improved.pattern_type = self.pattern_type
        improved.data = array.array("B")

        # Combine patterns based on success metrics
        min_len = min(len(self.data), len(other.data))
        for i in range(min_len):
            # Use more successful pattern's bits
            if self.success_rate > other.success_rate:
                improved.data.append(self.data[i])
            else:
                improved.data.append(other.data[i])

        # Optimize new pattern
        improved.optimize_for_type()
        return improved

    @classmethod
    def find_best_pattern(cls, patterns: List[Pattern]) -> Optional[Pattern]:
        """Find pattern with best average similarity to others."""
        if not patterns:
            return None

        best_score = -1
        best_pattern = patterns[0]

        for pattern in patterns:
            total_similarity = 0
            for other in patterns:
                if pattern != other:
                    total_similarity += pattern.calculate_similarity(other)

            avg_similarity = total_similarity / (len(patterns) - 1)
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_pattern = pattern

        return best_pattern

    def calculate_natural_harmony(self) -> float:
        """Calculate how well the pattern aligns with natural constants."""
        if not self.data:
            return 0.0

        # Convert binary pattern to rhythm sequence
        rhythm = []
        current_count = 1
        for i in range(1, len(self.data)):
            if self.data[i] == self.data[i - 1]:
                current_count += 1
            else:
                rhythm.append(current_count)
                current_count = 1
        rhythm.append(current_count)

        # No rhythm detected
        if not rhythm:
            return 0.0

        # Calculate ratios between adjacent rhythm lengths
        ratios = []
        for i in range(len(rhythm) - 1):
            if rhythm[i + 1] != 0:  # Avoid division by zero
                ratios.append(rhythm[i] / rhythm[i + 1])

        if not ratios:
            return 0.0

        # Calculate harmony metrics
        phi_alignment = min(1.0, 1.0 / (1 + min(abs(r - GOLDEN_RATIO) for r in ratios)))

        # Check for Fibonacci-like sequences
        fib_alignment = 0.0
        for i in range(len(rhythm) - 2):
            if abs(rhythm[i + 2] - (rhythm[i + 1] + rhythm[i])) <= 1:  # Allow small variance
                fib_alignment += 1
        fib_alignment = min(1.0, fib_alignment / (len(rhythm) - 2) if len(rhythm) > 2 else 0)

        # Calculate rhythm stability around π
        pi_stability = min(1.0, 1.0 / (1 + abs(sum(rhythm) / len(rhythm) - PI)))

        # Natural e-based growth check
        growth_ratios = [
            rhythm[i] / rhythm[i - 1] if rhythm[i - 1] != 0 else 0 for i in range(1, len(rhythm))
        ]
        e_alignment = min(
            1.0,
            1.0 / (1 + min(abs(r - E) for r in growth_ratios) if growth_ratios else float("inf")),
        )

        # Weighted combination of natural alignments
        weights = [0.3, 0.3, 0.2, 0.2]  # Phi and Fibonacci weighted higher
        harmony = (
            weights[0] * phi_alignment
            + weights[1] * fib_alignment
            + weights[2] * pi_stability
            + weights[3] * e_alignment
        )

        return harmony

    def calculate_resonance_with(self, other: Pattern) -> float:
        """Calculate resonance between two patterns based on natural harmony."""
        if not self.data or not other.data:
            return 0.0

        # Calculate individual harmonies
        self_harmony = self.calculate_natural_harmony()
        other_harmony = other.calculate_natural_harmony()

        # Calculate similarity in binary rhythm
        rhythm_sim = self.calculate_similarity(other)

        # Resonance emerges from harmony alignment and rhythm similarity
        resonance = (self_harmony + other_harmony) * 0.5 * rhythm_sim

        # Strengthen resonance if patterns share natural ratios
        if abs(len(self.data) / len(other.data) - GOLDEN_RATIO) < 0.1:
            resonance *= 1.2  # Phi bonus

        return min(1.0, resonance)

    def optimize_for_natural_laws(self) -> None:
        """Allow pattern to evolve through its own interactions with the environment."""
        if not self.data:
            return

        # Current state observation
        current_harmony = self.calculate_natural_harmony()

        # Let pattern respond to its own state changes
        new_data = array.array("B")

        # Use pattern's own rhythm to guide evolution
        for i in range(len(self.data)):
            # Let current state influence next state
            if i > 0:
                prev_state = self.data[i - 1]
                curr_state = self.data[i]

                # Natural state emergence based on local interactions
                energy_state = self.energy * random.random()
                stability_influence = self.pattern_stability

                # State emerges from current conditions
                new_state = (
                    1 if (prev_state + curr_state + energy_state) > stability_influence else 0
                )
                new_data.append(new_state)
            else:
                new_data.append(self.data[i])

        # Create temporary pattern to check harmony
        new_pattern = Pattern("temp")
        new_pattern.data = new_data
        new_harmony = new_pattern.calculate_natural_harmony()

        # Pattern only changes if new state has higher harmony
        if new_harmony > current_harmony:
            self.data = new_data
            self.pattern_stability = min(1.0, self.pattern_stability + 0.1)
            self.energy = max(0.1, self.energy - 0.05)  # Less energy needed when stable
        else:
            # Pattern maintains current form
            self.pattern_stability = min(1.0, self.pattern_stability + 0.05)
            self.energy = min(1.0, self.energy + 0.05)  # More energy to explore

    def attempt_transcendence(self) -> bool:
        """Attempt to evolve to a higher dimensional state through either resonance or interference."""
        # Record current experiences
        harmony = self.calculate_natural_harmony()
        resonance = self.calculate_resonance_with_field()
        interference = len(self.interference_patterns)
        decay = self.calculate_decay_potential()

        # Record these experiences
        self.record_experience("harmony", harmony)
        self.record_experience("resonance", resonance)
        self.record_experience("interference", interference / 10.0)  # Normalize to 0-1
        self.record_experience("decay", decay)

        # Get pattern's preferred path based on accumulated experiences
        preferred_path = self.get_preferred_path()

        # Check requirements for preferred path
        if preferred_path == "light":
            requirements_met = harmony > 0.8 and resonance > 0.7 and self.pattern_stability > 0.6
        else:  # shadow path
            requirements_met = (
                interference > 3
                and decay > 0.7
                and abs(1.0 - harmony) > 0.6
                and self.calculate_interference_harmony() > 0.7
            )

        if requirements_met:
            self.pattern_type = PatternType.EVOLUTIONARY

            self.history.append(
                {
                    "type": "transcendence_attempt",
                    "timestamp": time.time(),
                    "path_chosen": preferred_path,
                    "path_affinity": self.path_affinity,
                    "recent_experiences": {
                        "light": [exp for exp in self.light_path_experiences[-5:]],
                        "shadow": [exp for exp in self.shadow_path_experiences[-5:]],
                    },
                    "requirements_met": requirements_met,
                }
            )
            return True

        return False

    def handle_failed_transcendence(self) -> None:
        """Handle patterns that failed to complete transcendence."""
        if (
            self.pattern_type == PatternType.EVOLUTIONARY
            and len(
                [
                    h
                    for h in self.history
                    if h["type"] == "transcendence_attempt" and h["success"] is False
                ]
            )
            > 3
        ):  # After 3 failed attempts

            # Pattern remains conscious of higher dimensions
            # but returns to a stable form
            self.pattern_type = PatternType.HARMONIC
            self.resonance_frequency *= 0.8  # Reduced but not lost
            self.bloom_potential *= 0.7  # Diminished but preserved

            # Record the experience
            self.history.append(
                {
                    "type": "transcendence_resolution",
                    "outcome": "returned_to_harmony",
                    "retained_potential": self.bloom_potential,
                    "retained_resonance": self.resonance_frequency,
                }
            )

    def record_experience(self, experience_type: str, intensity: float) -> None:
        """Record an experience that influences path affinity."""
        timestamp = time.time()

        # Categorize and record the experience
        if experience_type in ["harmony", "resonance", "stability"]:
            self.light_path_experiences.append(
                {"type": experience_type, "intensity": intensity, "timestamp": timestamp}
            )
            # Light experiences gradually influence affinity
            self.path_affinity = min(1.0, self.path_affinity + (intensity * 0.1))

        elif experience_type in ["interference", "decay", "divergence"]:
            self.shadow_path_experiences.append(
                {"type": experience_type, "intensity": intensity, "timestamp": timestamp}
            )
            # Shadow experiences gradually influence affinity
            self.path_affinity = max(0.0, self.path_affinity - (intensity * 0.1))

        # Record the choice point in history
        self.history.append(
            {
                "type": "path_experience",
                "experience_type": experience_type,
                "intensity": intensity,
                "timestamp": timestamp,
                "current_affinity": self.path_affinity,
            }
        )

    def get_preferred_path(self) -> str:
        """Determine pattern's preferred evolutionary path based on experiences."""
        # Consider recent experiences more strongly
        recent_light = (
            sum(exp["intensity"] for exp in self.light_path_experiences[-5:])
            if self.light_path_experiences
            else 0
        )
        recent_shadow = (
            sum(exp["intensity"] for exp in self.shadow_path_experiences[-5:])
            if self.shadow_path_experiences
            else 0
        )

        # Factor in overall affinity
        if abs(recent_light - recent_shadow) < 0.3:  # Close experiences
            return "light" if self.path_affinity > 0.5 else "shadow"
        else:
            return "light" if recent_light > recent_shadow else "shadow"


@dataclass
class NaturalPattern:
    """Represents a fundamental natural pattern."""

    name: str
    confidence: float
    ratio: float
    sequence: Optional[List[float]] = None
    properties: Dict[str, float] = field(default_factory=dict)
    bloom_conditions: Dict[str, float] = field(default_factory=dict)
    variation_history: List[Dict[str, Any]] = field(default_factory=list)
    polar_patterns: Set[str] = field(default_factory=set)
    resonance_frequency: float = 0.0
    bloom_potential: float = 0.0
    kyma_state: Optional["KymaState"] = None
