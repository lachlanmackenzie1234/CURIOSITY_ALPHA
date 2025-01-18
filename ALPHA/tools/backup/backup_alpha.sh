#!/bin/bash

# Configuration
BACKUP_DIR="$HOME/ALPHA_Backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ALPHA_backup_$DATE"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create backup
echo "Creating backup: $BACKUP_NAME"
tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='*.pyc' \
    ALPHA/

# Keep only last 5 backups
cd "$BACKUP_DIR"
ls -t | tail -n +6 | xargs -I {} rm -f {}

echo "Backup complete: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
echo "Remaining backups:"
ls -lh "$BACKUP_DIR"
