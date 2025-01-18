#!/usr/bin/env python3
"""Test runner for ALPHA."""

import os
import subprocess
import sys
from pathlib import Path


def setup_environment():
    """Set up the test environment."""
    # Get the project root directory
    root_dir = Path(__file__).parent.absolute()

    # Add to Python path
    sys.path.insert(0, str(root_dir))
    os.environ["PYTHONPATH"] = str(root_dir)

    # Set test mode
    os.environ["ALPHA_TEST_MODE"] = "1"

    return root_dir


def run_tests():
    """Run the test suite."""
    root_dir = setup_environment()

    # Test files to run
    test_files = [
        "tests/unit/test_pattern_first_translator.py",
        "tests/unit/test_pattern_resonance.py",
        "tests/unit/test_binary_mapping.py",
        "tests/integration/test_translation_system.py",
    ]

    # Build command
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"] + test_files

    # Run tests
    try:
        subprocess.run(cmd, cwd=root_dir, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    run_tests()
