"""Script to run a single test file."""

import os
import sys
import unittest
from pathlib import Path


def run_test(test_file):
    """Run a single test file."""
    # Get the project root directory
    root_dir = Path(__file__).parent
    os.chdir(root_dir)

    # Add the root directory to Python path
    sys.path.insert(0, str(root_dir))

    # Load and run the test
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(root_dir), pattern=os.path.basename(test_file))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success/failure
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_single_test.py <test_file>")
        sys.exit(1)

    test_file = sys.argv[1]
    success = run_test(test_file)
    sys.exit(0 if success else 1)
