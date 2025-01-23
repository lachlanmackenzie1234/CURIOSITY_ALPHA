"""Unit tests for NEXUS protection system."""

import numpy as np
import pytest

from ALPHA.core.patterns.binary_pattern import BinaryPattern
from ALPHA.NEXUS.protection import InnerShield, NEXUSProtection, ResonanceBuffer


def create_test_pattern(data=None):
    """Create a test binary pattern."""
    if data is None:
        data = [1, 0, 1, 0, 1, 0, 1, 0]  # Balanced pattern
    return BinaryPattern(timestamp=0, data=data, source="test")


class TestInnerShield:
    """Test inner shield protection layer."""

    def test_validate_pattern(self):
        """Test pattern validation."""
        shield = InnerShield()

        # Test valid pattern
        valid_pattern = create_test_pattern()
        assert shield.validate_pattern(valid_pattern)

        # Test malformed pattern
        malformed = create_test_pattern([1, 0])
        assert not shield.validate_pattern(malformed)

        # Test disharmonious pattern
        disharmonious = create_test_pattern([1, 1, 1, 1, 1, 1, 1, 1])
        assert not shield.validate_pattern(disharmonious)

    def test_pattern_history(self):
        """Test pattern history management."""
        shield = InnerShield()
        pattern = create_test_pattern()

        # Add patterns and check history limit
        for _ in range(1100):
            shield.validate_pattern(pattern)

        assert len(shield._pattern_history) == 1000


class TestResonanceBuffer:
    """Test resonance buffer protection layer."""

    def test_establish_field(self):
        """Test resonance field establishment."""
        buffer = ResonanceBuffer()
        pattern = create_test_pattern()

        buffer.establish_field(pattern)
        assert buffer.resonance_field is not None
        assert len(buffer.active_frequencies) == 2

    def test_frequency_initialization(self):
        """Test frequency initialization."""
        buffer = ResonanceBuffer()
        pattern = create_test_pattern()

        buffer._initialize_frequencies(pattern)
        assert "primary" in buffer.active_frequencies
        assert "harmonic" in buffer.active_frequencies


class TestNEXUSProtection:
    """Test main protection system."""

    def test_initialization(self):
        """Test protection system initialization."""
        protection = NEXUSProtection()
        birth_pattern = create_test_pattern()

        protection.initialize_from_birth(birth_pattern)
        assert protection.birth_pattern is not None
        assert protection.resonance_buffer.resonance_field is not None

    def test_protect_pattern(self):
        """Test pattern protection."""
        protection = NEXUSProtection()
        birth_pattern = create_test_pattern()
        protection.initialize_from_birth(birth_pattern)

        # Test valid pattern
        valid_pattern = create_test_pattern()
        assert protection.protect_pattern(valid_pattern)

        # Test invalid pattern
        invalid_pattern = create_test_pattern([1, 1, 1, 1, 1, 1, 1, 1])
        assert not protection.protect_pattern(invalid_pattern)

    def test_resonance_check(self):
        """Test resonance checking."""
        protection = NEXUSProtection()
        birth_pattern = create_test_pattern()
        protection.initialize_from_birth(birth_pattern)

        # Test resonant pattern
        resonant = create_test_pattern()
        resonance = protection._check_resonance(resonant)
        assert resonance > 0.6

        # Test non-resonant pattern
        non_resonant = create_test_pattern([1, 1, 1, 1, 0, 0, 0, 0])
        resonance = protection._check_resonance(non_resonant)
        assert resonance < 0.6
