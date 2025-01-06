"""Base binary foundation module."""

from typing import Optional, Any, Tuple, Dict, List, Union
import array
import ast
import struct
import math
import numpy as np
from dataclasses import dataclass, field
import time


@dataclass
class PatternInfo:
    """Information about a detected pattern."""
    type_code: str
    confidence: float
    data: bytes
    metrics: Dict[str, float] = field(default_factory=dict)


class Binary:
    """Base class for binary data handling with code structure support."""
    
    # Binary format markers - using 4-bit markers for efficiency
    MARKER_MODULE = bytes([0x1])
    MARKER_CLASS = bytes([0x2])
    MARKER_FUNCTION = bytes([0x3])
    MARKER_IMPORT = bytes([0x4])
    MARKER_ASSIGN = bytes([0x5])
    MARKER_EXPR = bytes([0x6])
    MARKER_PATTERN = bytes([0x7])
    
    # Natural pattern identifiers - using 4-bit identifiers
    PATTERN_GOLDEN = bytes([0x8])
    PATTERN_FIBONACCI = bytes([0x9])
    PATTERN_EXPONENTIAL = bytes([0xA])
    PATTERN_PERIODIC = bytes([0xB])
    
    # Pattern header size (marker|type + confidence + length) = 13 bytes
    PATTERN_HEADER_SIZE = 13
    
    # Cache configuration
    MAX_CACHE_SIZE = 1000
    
    def __init__(self, data: Optional[bytes] = None):
        """Initialize binary data handler."""
        self._data = bytearray(data if data else b'')
        self.metadata: Dict[str, str] = {}
        self.structure: Dict[str, Any] = {}
        self.patterns: Dict[str, List[PatternInfo]] = {}
        
        # Initialize natural constants
        self.GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
        self.E = math.e
        self.PI = math.pi
        
        # Pattern cache with LRU tracking (pos, info, timestamp)
        self._pattern_cache: Dict[str, Tuple[int, PatternInfo, float]] = {}
        
    def to_bytes(self) -> bytes:
        """Convert to bytes."""
        return bytes(self._data)
        
    def from_bytes(self, data: bytes) -> None:
        """Load from bytes."""
        self._data = bytearray(data)
        self._pattern_cache.clear()  # Clear cache on data change
        
    def encode_python(self, source_code: str) -> None:
        """Encode Python source code into structured binary format."""
        try:
            # Parse Python code into AST
            tree = ast.parse(source_code)
            
            # Store original data
            original_data = bytes(self._data)
            
            # Clear data for encoding
            self._data = bytearray()
            
            # Encode module start
            self._data.extend(self.MARKER_MODULE)
            
            # Process each node in the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self._encode_import(node)
                elif isinstance(node, ast.ClassDef):
                    self._encode_class(node)
                elif isinstance(node, ast.FunctionDef):
                    self._encode_function(node)
                elif isinstance(node, ast.Assign):
                    self._encode_assignment(node)
                elif isinstance(node, ast.Expr):
                    self._encode_expression(node)
            
            # Store structure information
            self.structure = {
                'type': 'module',
                'ast': tree
            }
            
            # Append original data after encoded Python
            self._data.extend(original_data)
            
        except SyntaxError as e:
            self.metadata['error'] = f"Syntax error: {str(e)}"
        except Exception as e:
            self.metadata['error'] = f"Encoding error: {str(e)}"
    
    def decode_python(self) -> Optional[str]:
        """Decode binary format back to Python source code."""
        try:
            if not self._data:
                return None
                
            result = []
            pos = 0
            
            while pos < len(self._data):
                marker = self.get_segment(pos, 4)
                pos += 4
                
                if marker == self.MARKER_MODULE:
                    result.append("# Module")
                elif marker == self.MARKER_CLASS:
                    name_len = struct.unpack(
                        '!H',
                        self.get_segment(pos, 2)
                    )[0]
                    pos += 2
                    name = self.get_segment(pos, name_len).decode('utf-8')
                    pos += name_len
                    result.append(f"\nclass {name}:")
                elif marker == self.MARKER_FUNCTION:
                    name_len = struct.unpack(
                        '!H',
                        self.get_segment(pos, 2)
                    )[0]
                    pos += 2
                    name = self.get_segment(pos, name_len).decode('utf-8')
                    pos += name_len
                    result.append(f"\n    def {name}():")
                
            return '\n'.join(result)
            
        except Exception as e:
            self.metadata['error'] = f"Decoding error: {str(e)}"
            return None
    
    def _encode_import(self, node: ast.Import) -> None:
        """Encode import statement."""
        self._data += self.MARKER_IMPORT
        for name in node.names:
            name_bytes = name.name.encode('utf-8')
            self._data += struct.pack('!H', len(name_bytes))
            self._data += name_bytes
    
    def _encode_class(self, node: ast.ClassDef) -> None:
        """Encode class definition."""
        self._data += self.MARKER_CLASS
        name_bytes = node.name.encode('utf-8')
        self._data += struct.pack('!H', len(name_bytes))
        self._data += name_bytes
    
    def _encode_function(self, node: ast.FunctionDef) -> None:
        """Encode function definition."""
        self._data += self.MARKER_FUNCTION
        name_bytes = node.name.encode('utf-8')
        self._data += struct.pack('!H', len(name_bytes))
        self._data += name_bytes
    
    def _encode_assignment(self, node: ast.Assign) -> None:
        """Encode assignment statement."""
        self._data += self.MARKER_ASSIGN
        # Simplified for now - just store target names
        for target in node.targets:
            if isinstance(target, ast.Name):
                name_bytes = target.id.encode('utf-8')
                self._data += struct.pack('!H', len(name_bytes))
                self._data += name_bytes
    
    def _encode_expression(self, node: ast.Expr) -> None:
        """Encode expression."""
        self._data += self.MARKER_EXPR
        # Simplified - just mark expression boundary
    
    def get_segment(self, start: int, length: int) -> bytes:
        """Get a segment of binary data."""
        return bytes(self._data[start:start + length])
    
    def set_segment(self, start: int, data: bytes) -> None:
        """Set a segment of binary data."""
        temp = bytearray(data)
        self._data[start:start + len(temp)] = temp
    
    def append(self, data: bytes) -> None:
        """Append binary data."""
        self._data += data
    
    def clear(self) -> None:
        """Clear binary data."""
        self._data.clear()
        self.metadata.clear()
        self.structure.clear()
        self._pattern_cache.clear()
    
    def get_size(self) -> int:
        """Get size of binary data."""
        return len(self._data)
    
    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value."""
        return self.metadata.get(key)
    
    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists."""
        return key in self.metadata 
    
    def extend_data(self, value: bytes) -> None:
        """Extend binary data with proper type handling."""
        self._data.extend(value)
        
    def append_byte(self, value: int) -> None:
        """Append a single byte with proper type handling."""
        self._data.extend(bytes([value]))
        
    def encode_pattern(
        self,
        pattern_type: bytes,
        data: bytes,
        confidence: float
    ) -> None:
        """Encode a natural pattern into the binary structure."""
        try:
            # Create pattern marker byte
            marker_type = (self.MARKER_PATTERN[0] << 4) | pattern_type[0]
            
            # Create pattern header
            header = bytearray([marker_type])
            header.append(int(confidence * 255))  # Normalize to 0-255
            
            # Combine header and data
            pattern_data = bytes(header) + data
            
            # Append to binary data
            self._data.extend(pattern_data)
            
            # Update pattern tracking
            type_str = {
                self.PATTERN_EXPONENTIAL[0]: 'EXP',
                self.PATTERN_FIBONACCI[0]: 'FIB',
                self.PATTERN_GOLDEN[0]: 'PHI',
                self.PATTERN_PERIODIC[0]: 'PER'
            }.get(pattern_type[0], chr(pattern_type[0] + ord('A')))
            
            pattern_info = PatternInfo(
                type_code=type_str,
                confidence=confidence,
                data=data
            )
            
            if type_str not in self.patterns:
                self.patterns[type_str] = []
            self.patterns[type_str].append(pattern_info)
            
            # Update cache with timestamp
            self._update_cache(
                type_str,
                (len(self._data) - len(pattern_data), pattern_info, time.time())
            )
            
        except Exception as e:
            self.set_metadata('pattern_encode_error', str(e))
            
    def _update_cache(
        self,
        key: str,
        value: Tuple[int, PatternInfo, float]
    ) -> None:
        """Update pattern cache with LRU eviction."""
        if len(self._pattern_cache) >= self.MAX_CACHE_SIZE:
            # Remove oldest entry
            oldest_key = min(
                self._pattern_cache,
                key=lambda k: self._pattern_cache[k][2]
            )
            del self._pattern_cache[oldest_key]
        self._pattern_cache[key] = value
        
    def decode_pattern(
        self,
        pattern_type_or_pos: Union[str, int],
        data: Optional[bytes] = None
    ) -> Optional[Union[Tuple[str, float, bytes], PatternInfo]]:
        """Decode a pattern from binary data."""
        try:
            if isinstance(pattern_type_or_pos, int):
                # Position-based decoding
                pos = pattern_type_or_pos
                
                # Check cache first
                for type_str, (cached_pos, info, _) in (
                    self._pattern_cache.items()
                ):
                    if cached_pos == pos:
                        # Update access timestamp
                        self._pattern_cache[type_str] = (
                            pos, info, time.time()
                        )
                        return info
                
                # Ensure we have enough data
                if pos + 2 > len(self._data):  # Need at least marker and confidence
                    return None
                
                # Read pattern data
                marker_type = self._data[pos]
                marker = marker_type >> 4
                pattern_type = marker_type & 0xF
                
                if marker != self.MARKER_PATTERN[0]:
                    return None
                    
                # Read confidence
                confidence = float(self._data[pos + 1]) / 255.0
                
                # Get pattern data
                pattern_data = bytes(self._data[pos + 2:])
                
                # Map type to string
                type_str = {
                    self.PATTERN_EXPONENTIAL[0]: 'EXP',
                    self.PATTERN_FIBONACCI[0]: 'FIB',
                    self.PATTERN_GOLDEN[0]: 'PHI',
                    self.PATTERN_PERIODIC[0]: 'PER'
                }.get(pattern_type, chr(pattern_type + ord('A')))
                
                # Create pattern info
                pattern_info = PatternInfo(
                    type_code=type_str,
                    confidence=confidence,
                    data=pattern_data
                )
                
                # Update cache
                self._update_cache(
                    type_str,
                    (pos, pattern_info, time.time())
                )
                
                return pattern_info
                
            else:
                # Data-based decoding
                if not data or len(data) < 2:
                    return None
                    
                # Extract pattern type and confidence
                marker_type = data[0]
                pattern_type = marker_type & 0xF
                confidence = float(data[1]) / 255.0
                
                # Map type to string
                type_str = {
                    self.PATTERN_EXPONENTIAL[0]: 'EXP',
                    self.PATTERN_FIBONACCI[0]: 'FIB',
                    self.PATTERN_GOLDEN[0]: 'PHI',
                    self.PATTERN_PERIODIC[0]: 'PER'
                }.get(pattern_type, chr(pattern_type + ord('A')))
                
                return (type_str, confidence, data[2:])
                
        except Exception as e:
            self.set_metadata('pattern_decode_error', str(e))
            return None
            
    def analyze_patterns(self) -> Dict[str, List[Tuple[float, array.array]]]:
        """Analyze binary data for natural patterns."""
        patterns: Dict[str, List[Tuple[float, array.array]]] = {}
        
        try:
            # First check for encoded patterns
            pos = 0
            data_len = len(self._data)
            while pos < data_len:
                if self._data[pos] == self.MARKER_PATTERN[0]:
                    pattern = self.decode_pattern(pos)
                    if pattern:
                        type_str = pattern.type_code
                        conf = pattern.confidence
                        data = pattern.data
                        
                        # Map pattern types efficiently
                        pattern_map = {
                            chr(self.PATTERN_EXPONENTIAL[0] + ord('A')):
                                'exponential',
                            chr(self.PATTERN_FIBONACCI[0] + ord('A')):
                                'fibonacci',
                            chr(self.PATTERN_GOLDEN[0] + ord('A')):
                                'golden',
                            chr(self.PATTERN_PERIODIC[0] + ord('A')):
                                'periodic'
                        }
                        pattern_key = pattern_map.get(
                            type_str,
                            type_str.lower()
                        )
                        
                        # Use array.array directly
                        arr = array.array('B')
                        arr.frombytes(data)
                        
                        if pattern_key not in patterns:
                            patterns[pattern_key] = []
                        patterns[pattern_key].append((conf, arr))
                        
                        # Skip entire pattern
                        pos += self.PATTERN_HEADER_SIZE + len(data)
                    else:
                        pos += 1
                else:
                    pos += 1
            
            # Then analyze raw data for natural patterns
            if data_len >= 2:
                # Convert to numpy array efficiently
                raw_data = np.frombuffer(
                    bytes(self._data),
                    dtype=np.uint8
                )
                float_data = raw_data.astype(np.float64)
                arr_data = float_data + 1e-10
                
                # Check for golden ratio patterns
                ratios = np.empty(len(arr_data) * 3 - 3)
                idx = 0
                for i in range(len(arr_data) - 1):
                    a = float_data[i]
                    b = float_data[i + 1]
                    
                    ratios[idx] = abs(b / a - self.GOLDEN_RATIO)
                    idx += 1
                    ratios[idx] = abs((b + 256) / a - self.GOLDEN_RATIO)
                    idx += 1
                    ratios[idx] = abs(b / (a + 256) - self.GOLDEN_RATIO)
                    idx += 1
                
                min_deviation = np.min(ratios)
                confidence = 1.0 / (1.0 + min_deviation)
                if confidence > 0.6:
                    arr = array.array('B')
                    arr.frombytes(raw_data.tobytes())
                    patterns['golden'] = [(confidence, arr)]
                
                # Check for Fibonacci patterns efficiently
                if len(arr_data) >= 3:
                    # Use vectorized operations
                    a_data = float_data[:-2]
                    b_data = float_data[1:-1]
                    c_data = float_data[2:]
                    matches = np.abs(c_data - (a_data + b_data)) < 3
                    if np.any(matches):
                        conf = float(np.mean(matches))
                        if conf > 0.3:
                            arr = array.array('B')
                            arr.frombytes(raw_data.tobytes())
                            patterns['fibonacci'] = [(conf, arr)]
                
                # Check for exponential patterns efficiently
                if len(arr_data) >= 2:
                    log_data = np.log(arr_data)
                    rates = np.diff(log_data)
                    matches = np.abs(np.diff(rates)) < 0.3
                    if np.any(matches):
                        conf = float(np.mean(matches))
                        if conf > 0.3:
                            arr = array.array('B')
                            arr.frombytes(raw_data.tobytes())
                            patterns['exponential'] = [(conf, arr)]
                
                # Check for periodic patterns efficiently
                if len(arr_data) >= 4:
                    fft = np.fft.fft(arr_data)
                    power = np.abs(fft) ** 2
                    freq = np.argmax(power[1:]) + 1
                    conf = float(power[freq] / np.sum(power))
                    if conf > 0.1:
                        arr = array.array('B')
                        arr.frombytes(raw_data.tobytes())
                        patterns['periodic'] = [(conf, arr)]
            
            return patterns
            
        except Exception as e:
            self.set_metadata('pattern_analysis_error', str(e))
            return {} 