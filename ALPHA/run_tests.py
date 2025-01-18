"""Script to run ALPHA tests."""

import os
import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with coverage."""
    # Get the project root directory
    root_dir = Path(__file__).parent
    os.chdir(root_dir)

    # Install requirements
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )

    # Install package in development mode
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

    # Run tests with coverage
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--cov=core",
            "--cov-report=term-missing",
        ],
        check=True,
    )


if __name__ == "__main__":
    run_tests()
