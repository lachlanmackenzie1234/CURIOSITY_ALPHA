"""Test configuration for ALPHA."""

import os
import sys
from pathlib import Path

# Get the project root directory
root_dir = str(Path(__file__).parent.parent.absolute())

# Add the root directory to Python path if not already there
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Set environment variable for test configuration
os.environ["ALPHA_TEST_MODE"] = "1" 