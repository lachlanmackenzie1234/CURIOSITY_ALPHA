"""Pattern type definitions."""

from enum import Enum
from typing import Any, Dict, List, Optional


class NaturalPrincipleType(str, Enum):
    """Types of natural principles that can be observed."""

    E = "e"  # Environmental principle
    R = "r"  # Resonance principle
    M = "m"  # Memory principle
    S = "s"  # Stability principle


class BinaryEncodingType(str, Enum):
    """Types of binary encodings for patterns."""

    RESONANT = "resonant"  # Preserves resonance properties
    COMPRESSED = "compressed"  # Optimized for size
    SYMBOLIC = "symbolic"  # Symbolic representation
    DIRECT = "direct"  # Direct binary mapping
