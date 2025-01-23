"""Test system birth in isolation."""

import pytest

from ALPHA.core.system_birth import SystemBirth


@pytest.fixture
def system():
    """Create a fresh SystemBirth instance for each test."""
    return SystemBirth()


def test_basic_existence(system):
    """Test basic existence experience."""
    # Experience existence for a short duration
    system.feel_existence(duration_ns=100_000_000)  # 100ms

    # Get summary
    summary = system.get_existence_summary()
    total_patterns = sum(summary.values())

    # Verify we captured changes
    assert total_patterns > 0, "No state changes detected"
    assert all(count >= 0 for count in summary.values()), "Invalid pattern counts"


def test_pattern_crystallization(system):
    """Test birth pattern crystallization."""
    # Experience some existence
    system.feel_existence(duration_ns=100_000_000)

    # Crystallize birth pattern
    birth_pattern = system.crystallize_birth()

    # Verify pattern formation
    assert birth_pattern is not None, "Failed to crystallize birth pattern"
    assert len(birth_pattern.sequence) > 0, "Empty birth pattern"


def test_state_preservation(system):
    """Test state preservation and loading."""
    # Experience existence
    system.feel_existence(duration_ns=100_000_000)

    # Preserve state
    system._preserve_state()

    # Create new system and load state
    new_system = SystemBirth()
    new_system._load_preserved_state()

    # Verify state restoration
    assert new_system._birth_phase == system._birth_phase, "Birth phase not preserved"
    assert new_system.existence_patterns == system.existence_patterns, "Patterns not preserved"
