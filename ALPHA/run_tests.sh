#!/bin/bash

# Install requirements
python3 -m pip install -r requirements.txt

# Install package in development mode
python3 -m pip install -e .

# Run tests with coverage
python3 -m pytest tests/ -v --cov=core --cov-report=term-missing 