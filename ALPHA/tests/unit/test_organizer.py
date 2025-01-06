"""Tests for the CodeOrganizer class."""

from ALPHA.core.analysis.organizer import CodeOrganizer


def test_learn_from_file(tmp_path):
    """Test learning patterns from a file."""
    # Create a test file
    test_file = tmp_path / "test.py"
    test_file.write_text(
        """
def test_function():
    print("Hello, World!")
    return 42
"""
    )

    organizer = CodeOrganizer()
    organizer.learn_from_file(str(test_file))
    assert len(organizer.patterns) > 0


def test_organize_directory(tmp_path):
    """Test organizing a directory."""
    # Create test files
    (tmp_path / "test1.py").write_text("def func1(): pass")
    (tmp_path / "test2.py").write_text("def func2(): pass")

    organizer = CodeOrganizer()
    results = organizer.organize_directory(str(tmp_path))
    assert len(results) > 0

    for file_path, patterns in results.items():
        assert isinstance(file_path, str)
        assert isinstance(patterns, list)


def test_find_similar_files(tmp_path):
    """Test finding similar files."""
    # Create test files with similar content
    (tmp_path / "test1.py").write_text("def similar_func(): pass")
    (tmp_path / "test2.py").write_text("def similar_func(): return True")

    organizer = CodeOrganizer()
    organizer.learn_from_file(str(tmp_path / "test1.py"))

    similar = organizer.find_similar_files(str(tmp_path / "test2.py"))
    assert len(similar) > 0
    for file_path, score in similar:
        assert isinstance(file_path, str)
        assert 0 <= score <= 1.0


def test_suggest_organization(tmp_path):
    """Test suggesting organization for files."""
    # Create test files
    (tmp_path / "test1.py").write_text("def test(): pass")

    organizer = CodeOrganizer()
    suggestions = organizer.suggest_organization(str(tmp_path / "test1.py"))
    assert isinstance(suggestions, list)
    for suggestion in suggestions:
        assert isinstance(suggestion, tuple)
        assert len(suggestion) == 2
        assert isinstance(suggestion[0], str)  # category
        assert isinstance(suggestion[1], float)  # confidence
