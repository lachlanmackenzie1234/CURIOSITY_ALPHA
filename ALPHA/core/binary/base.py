"""Base binary foundation module."""

from typing import Optional, Dict
import array


class Binary:
    """Base class for binary data handling."""
    
    def __init__(self, data: Optional[bytes] = None):
        """Initialize binary data handler."""
        self.data = array.array('B', data if data else [])
        self.metadata: Dict[str, str] = {}
    
    def to_bytes(self) -> bytes:
        """Convert to bytes."""
        return self.data.tobytes()
    
    def from_bytes(self, data: bytes) -> None:
        """Load from bytes."""
        self.data = array.array('B', data)
    
    def get_segment(self, start: int, length: int) -> bytes:
        """Get a segment of binary data."""
        return self.data[start:start + length].tobytes()
    
    def set_segment(self, start: int, data: bytes) -> None:
        """Set a segment of binary data."""
        temp = array.array('B', data)
        self.data[start:start + len(temp)] = temp
    
    def append(self, data: bytes) -> None:
        """Append binary data."""
        self.data.extend(array.array('B', data))
    
    def clear(self) -> None:
        """Clear binary data."""
        self.data = array.array('B')
        self.metadata.clear()
    
    def get_size(self) -> int:
        """Get size of binary data."""
        return len(self.data)
    
    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value."""
        return self.metadata.get(key) 