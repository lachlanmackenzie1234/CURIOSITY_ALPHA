#!/bin/zsh

# Set up environment
export PYTHONPATH=/Users/lachlanmackenzie/Curiosity/src

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the test runner with all arguments passed through
python3 "${SCRIPT_DIR}/test_runner.py" "$@"
