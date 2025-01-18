"""Tests for the Adaptive class."""

from ALPHA.core.binary_foundation.base import Binary
from ALPHA.core.patterns.adaptive import Adaptive


def test_learn():
    """Test learning new patterns."""
    adaptive = Adaptive()

    # Test learning a simple pattern
    pattern = b"test pattern"
    adaptive.learn(pattern)
    assert len(adaptive.patterns) > 0

    # Test learning multiple patterns
    adaptive.learn(b"another pattern")
    assert len(adaptive.patterns) > 1


def test_evolve():
    """Test pattern evolution."""
    adaptive = Adaptive()

    # Learn some patterns
    patterns = [b"test1", b"test2", b"test3"]
    for p in patterns:
        adaptive.learn(p)

    # Test evolution
    evolved = adaptive.evolve()
    assert evolved is not None
    assert isinstance(evolved, bytes)


def test_mutate():
    """Test pattern mutation."""
    adaptive = Adaptive()

    # Create a pattern
    pattern = Binary(b"test pattern")

    # Test mutation
    mutated = adaptive.mutate(pattern)
    assert mutated is not None
    assert isinstance(mutated, Binary)
    assert mutated.to_bytes() != pattern.to_bytes()


def test_combine():
    """Test pattern combination."""
    adaptive = Adaptive()

    # Create patterns
    pattern1 = Binary(b"test1")
    pattern2 = Binary(b"test2")

    # Test combination
    combined = adaptive.combine(pattern1, pattern2)
    assert combined is not None
    assert isinstance(combined, Binary)
    assert len(combined) == max(len(pattern1), len(pattern2))
