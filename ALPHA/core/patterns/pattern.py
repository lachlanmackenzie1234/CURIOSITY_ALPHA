"""Enhanced binary pattern core for efficient processing, storage, and communication."""

from __future__ import annotations
import array
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
import random
import math
from enum import Enum
import time

class PatternType(Enum):
    """Types of specialized patterns."""
    PROCESSOR = "processor"      # Optimized for computation
    COMMUNICATOR = "comm"        # Optimized for data transfer
    NETWORK = "network"         # Optimized for connectivity
    INTEGRATOR = "integrator"   # Optimized for system binding
    STORAGE = "storage"         # Optimized for data persistence
    COMPRESSOR = "compressor"   # Optimized for data compression
    INDEXER = "indexer"        # Optimized for data organization
    STILLNESS = "stillness"     # Optimized for stability
    FLOW = "flow"              # Optimized for adaptation

@dataclass
class StorageMetadata:
    """Metadata for storage optimization."""
    compression_ratio: float = 1.0
    access_frequency: int = 0
    last_access_time: float = 0.0
    priority_level: int = 1
    checksum: Optional[int] = None
    references: Set[str] = field(default_factory=set)
    stability: float = 0.5      # How stable the pattern is (0-1)
    resonance: Dict[str, float] = field(default_factory=dict)

@dataclass
class Pattern:
    """Core pattern class optimized for system communication and processing."""
    
    id: str
    pattern_type: PatternType = PatternType.PROCESSOR
    data: array.array = field(default_factory=lambda: array.array('B'))
    energy: float = 1.0
    history: List[Dict] = field(default_factory=list)
    success_rate: float = 0.5
    optimization_threshold: float = 0.7
    learning_rate: float = 0.1
    connections: Set[str] = field(default_factory=set)
    potential_forms: Set[str] = field(default_factory=set)
    
    # Storage-specific attributes
    storage_metadata: StorageMetadata = field(default_factory=StorageMetadata)
    
    # Performance metrics
    processing_efficiency: float = 0.5
    communication_latency: float = 0.5
    network_stability: float = 0.5
    storage_efficiency: float = 0.5
    pattern_stability: float = 0.5

    def __post_init__(self):
        """Initialize pattern with empty data if none provided."""
        if not self.data:
            self.data = array.array('B')
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
        return sum([
            weights[0] * binary_sim,
            weights[1] * type_sim,
            weights[2] * connection_sim,
            weights[3] * form_sim
        ])

    def _calculate_binary_similarity(self, other: Pattern) -> float:
        """Calculate raw binary similarity between patterns."""
        total_bits = max(len(self.data), len(other.data))
        if total_bits == 0:
            return 1.0

        differences = 0
        for i in range(total_bits):
            if i < len(self.data) and i < len(other.data):
                differences += bin(self.data[i] ^ other.data[i]).count('1')
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
            self._optimize_indexing()
        else:
            super().optimize_for_type()  # Call existing optimization methods

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
        self.storage_efficiency = 1 - (len(self.data) / (self.storage_metadata.compression_ratio * len(self.data)))
        
        # Update metadata
        self._update_storage_metadata()

    def _apply_run_length_encoding(self) -> None:
        """Apply run-length encoding for repetitive patterns."""
        if not self.data:
            return
            
        encoded = array.array('B')
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
        encoded = array.array('B')
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
            block = self.data[i:i + block_size]
            if len(block) < block_size:  # Pad last block
                block.extend([0] * (block_size - len(block)))
            blocks.append(block)
            
        # Optimize each block
        optimized = array.array('B')
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
        self.storage_metadata.compression_ratio = len(self.data) / (len(self.data) * self.storage_efficiency)
        
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
            blocks = [self.data[i:i + block_size] for i in range(0, len(self.data), block_size)]
            self.data = array.array('B')
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
                learning_rate=self.learning_rate
            )
            evolved.data = array.array('B', self.data)
            evolved.connections = self.connections.copy()
            
            # Calculate entropy for targeted mutation
            entropy = self.calculate_pattern_entropy()
            
            # Type-specific evolution
            if self.pattern_type == PatternType.PROCESSOR:
                mutation_rate *= (1 - self.processing_efficiency)
            elif self.pattern_type == PatternType.COMMUNICATOR:
                mutation_rate *= (1 - self.communication_latency)
            elif self.pattern_type == PatternType.NETWORK:
                mutation_rate *= (1 - self.network_stability)
            
            # Efficient mutation using bit operations and entropy
            for i in range(len(evolved.data)):
                if random.random() < mutation_rate * (1 + entropy):
                    if entropy > 0.7:  # High entropy - subtle changes
                        evolved.data[i] ^= (1 << random.randint(0, 2))
                    else:  # Low entropy - bigger changes
                        evolved.data[i] ^= (1 << random.randint(0, 7))
            
            # Optimize for specific type
            evolved.optimize_for_type()
            
            # Track evolution with optimization metrics
            self.history.append({
                'type': 'evolution',
                'energy': self.energy,
                'mutation_rate': mutation_rate,
                'entropy': entropy,
                'pattern_type': self.pattern_type.value,
                'processing_efficiency': self.processing_efficiency,
                'communication_latency': self.communication_latency,
                'network_stability': self.network_stability,
                'success': None
            })
            
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
        self.history.append({
            'type': 'merge',
            'energy': self.energy,
            'merge_factor': merge_factor,
            'entropy_diff': abs(self_entropy - other_entropy),
            'pattern_type': self.pattern_type.value,
            'success': None
        })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get pattern performance metrics."""
        return {
            'processing_efficiency': self.processing_efficiency,
            'communication_latency': self.communication_latency,
            'network_stability': self.network_stability,
            'energy': self.energy,
            'success_rate': self.success_rate,
            'connections': len(self.connections)
        }
    
    def add_segment(self, binary_data: bytes) -> None:
        """Add new binary segment efficiently."""
        for byte in binary_data:
            self.data.append(byte)
    
    def optimize_mutation_rate(self) -> float:
        """Dynamically adjust mutation rate based on success history."""
        if not self.history:
            return 0.1  # Default rate
            
        recent_successes = sum(
            1 for entry in self.history[-10:]
            if entry.get('success', False)
        )
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
            self.history[-1]['success'] = success
            
            # Update success rate
            recent_success_rate = sum(
                1 for entry in self.history[-10:]
                if entry.get('success', False)
            ) / min(10, len(self.history))
            
            self.success_rate = (
                self.success_rate * 0.9 + recent_success_rate * 0.1
            )
            
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
            
        matching_bits = sum(
            bin(a ^ b).count('1')
            for a, b in zip(self.data, other.data)
        )
        
        return 1 - (matching_bits / (min_len * 8))
    
    def to_bytes(self) -> bytes:
        """Convert to bytes for translation."""
        return bytes(self.data)
    
    @classmethod
    def from_bytes(cls, id: str, data: bytes) -> Pattern:
        """Create pattern from bytes."""
        pattern = cls(id)
        pattern.data = array.array('B', data)
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
            
        pattern.data = array.array('B', bytes_list)
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
        improved.data = array.array('B')
        
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

