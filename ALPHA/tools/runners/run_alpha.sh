#!/bin/zsh

# Set up environment
export PYTHONPATH=/Users/lachlanmackenzie/Curiosity/src

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run analysis
python3 "${SCRIPT_DIR}/run_alpha_analysis.py" "$@"
