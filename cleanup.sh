#!/bin/zsh

# Remove any misplaced directories
rm -rf level_4 core patterns src

# Clean cache and system files
rm -rf __pycache__ .mypy_cache .pytest_cache .coverage
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name ".DS_Store" -exec rm {} +

# Ensure ALPHA package structure
mkdir -p ALPHA/core/{patterns,memory,translation,execution,binary_foundation,analysis,learning,semantic}
mkdir -p ALPHA/tests/{unit,integration,analysis}
mkdir -p ALPHA/tools/runners
mkdir -p ALPHA/utils
mkdir -p ALPHA/examples

# Create necessary __init__.py files
touch ALPHA/core/{patterns,memory,translation,execution,binary_foundation,analysis,learning,semantic}/__init__.py
touch ALPHA/tests/{unit,integration,analysis}/__init__.py
touch ALPHA/tools/__init__.py
touch ALPHA/tools/runners/__init__.py
touch ALPHA/utils/__init__.py

# Show final directory structure
echo "\nFinal directory structure:"
ls -R ALPHA/
