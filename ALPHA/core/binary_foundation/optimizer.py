"""Binary optimization module."""

from typing import List, Dict, Optional
from .base import Binary


class OptimizationPattern:
    """Pattern for binary optimization."""
    
    def __init__(self, name: str):
        """Initialize optimization pattern."""
        self.name = name
        self.pattern = Binary()
        self.confidence = 0.0
        self.metrics: Dict[str, float] = {
            'efficiency': 0.0,
            'reliability': 0.0,
            'adaptability': 0.0
        }
    
    def apply(self, data: Binary) -> Binary:
        """Apply optimization pattern to binary data."""
        result = Binary()
        
        try:
            # Apply pattern transformations
            input_bytes = data.to_bytes()
            output_bytes = self._transform(input_bytes)
            result.from_bytes(output_bytes)
            
            # Copy relevant metadata
            for key in data.metadata:
                if key.startswith('opt_'):
                    result.set_metadata(key, data.get_metadata(key))
            
            # Add optimization metadata
            result.set_metadata('opt_pattern', self.name)
            result.set_metadata('opt_confidence', str(self.confidence))
            
        except Exception as e:
            print(f"Error applying optimization pattern: {str(e)}")
            return data
        
        return result
    
    def learn(self, original: Binary, optimized: Binary) -> None:
        """Learn from successful optimization."""
        try:
            # Update pattern based on differences
            orig_bytes = original.to_bytes()
            opt_bytes = optimized.to_bytes()
            
            # Simple learning: store the transformation
            self.pattern.clear()
            self.pattern.from_bytes(opt_bytes)
            
            # Update metrics
            self._update_metrics(orig_bytes, opt_bytes)
            
        except Exception as e:
            print(f"Error learning optimization pattern: {str(e)}")
    
    def calculate_confidence(self, data: Binary) -> float:
        """Calculate confidence for applying this pattern."""
        try:
            # Compare with stored pattern
            similarity = self._calculate_similarity(
                data.to_bytes(),
                self.pattern.to_bytes()
            )
            
            # Weight similarity with metrics
            weighted_metrics = sum(self.metrics.values()) / len(self.metrics)
            self.confidence = similarity * weighted_metrics
            
            return self.confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.0
    
    def _transform(self, data: bytes) -> bytes:
        """Transform binary data using the pattern."""
        # Simple transformation: apply learned pattern
        if self.pattern.get_size() > 0:
            pattern_bytes = self.pattern.to_bytes()
            result = bytearray(data)
            
            # Apply pattern transformations
            for i in range(min(len(result), len(pattern_bytes))):
                result[i] = result[i] ^ pattern_bytes[i]
            
            return bytes(result)
        
        return data
    
    def _update_metrics(self, original: bytes, optimized: bytes) -> None:
        """Update optimization metrics."""
        if not original or not optimized:
            return
        
        # Efficiency: reduction in size
        size_ratio = len(optimized) / len(original)
        self.metrics['efficiency'] = 1.0 - min(1.0, size_ratio)
        
        # Reliability: pattern consistency
        matching_bytes = sum(1 for a, b in zip(original, optimized) if a == b)
        self.metrics['reliability'] = matching_bytes / len(original)
        
        # Adaptability: pattern flexibility
        unique_transforms = len(set(a ^ b for a, b in zip(original, optimized)))
        self.metrics['adaptability'] = unique_transforms / 256  # Max possible transforms
    
    def _calculate_similarity(self, data1: bytes, data2: bytes) -> float:
        """Calculate similarity between two binary sequences."""
        if not data1 or not data2:
            return 0.0
        
        min_len = min(len(data1), len(data2))
        if min_len == 0:
            return 0.0
        
        # Calculate normalized Hamming distance
        matches = sum(1 for i in range(min_len) if data1[i] == data2[i])
        return matches / min_len 