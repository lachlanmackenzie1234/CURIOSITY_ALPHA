"""Tests for the BinaryTranslator class."""

from ALPHA.core.translation.translator import BinaryTranslator


def test_learn_mapping():
    """Test learning mappings between patterns and meanings."""
    translator = BinaryTranslator()

    # Test basic mapping
    pattern = b"test pattern"
    meaning = "test"
    translator.learn_mapping(pattern, meaning)
    assert meaning in translator.mappings
    assert pattern == translator.mappings[meaning]


def test_discover_structure():
    """Test finding structural relationships between patterns."""
    translator = BinaryTranslator()

    pattern1 = b"hello world"
    pattern2 = b"world hello"

    similarity = translator.discover_structure(pattern1, pattern2)
    assert 0 <= similarity <= 1.0


def test_find_similar_patterns():
    """Test finding similar patterns."""
    translator = BinaryTranslator()

    # Add some patterns
    translator.learn_mapping(b"test pattern", "test")
    translator.learn_mapping(b"test data", "data")

    similar = translator.find_similar_patterns(b"test")
    assert len(similar) > 0
    for pattern, score in similar.items():
        assert 0 <= score <= 1.0


def test_suggest_organization():
    """Test suggesting organization based on patterns."""
    translator = BinaryTranslator()

    # Add some patterns
    translator.learn_mapping(b"test pattern", "test")
    translator.learn_mapping(b"test data", "data")

    suggestions = translator.suggest_organization(b"test")
    assert len(suggestions) > 0
    for meaning, score in suggestions:
        assert isinstance(meaning, str)
        assert 0 <= score <= 1.0


def test_learn_relationship():
    """Test learning relationships between meanings."""
    translator = BinaryTranslator()

    # Add some patterns
    translator.learn_mapping(b"test pattern", "test")
    translator.learn_mapping(b"test data", "data")

    translator.learn_relationship("test", "data")
    related = translator.get_related("test")
    assert "data" in related
