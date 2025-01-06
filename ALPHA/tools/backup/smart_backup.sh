#!/bin/bash

# Configuration
BACKUP_DIR="$HOME/ALPHA_Backups"
LAST_CHANGE_FILE="$BACKUP_DIR/.last_change"
ACTIVITY_TIMEOUT=3600  # 1 hour in seconds
BACKUP_INTERVAL=900    # 15 minutes in seconds
DATE=$(date +%Y%m%d)
TIME=$(date +%H%M%S)

# Create backup directory structure
mkdir -p "$BACKUP_DIR"/{daily,weekly,monthly}

# Function to create a backup
create_backup() {
    local backup_name="ALPHA_${DATE}_${TIME}"
    local backup_file="$BACKUP_DIR/daily/$backup_name.tar.gz"
    
    echo "Creating backup: $backup_name"
    tar -czf "$backup_file" \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='.mypy_cache' \
        --exclude='*.pyc' \
        --exclude="$BACKUP_DIR" \
        ALPHA/
    
    echo "$(date +%s)" > "$LAST_CHANGE_FILE"
    echo "Backup complete: $backup_name"
    
    # Run retention management
    manage_retention
}

# Function to manage backup retention
manage_retention() {
    local current_time=$(date +%s)
    
    # Process daily backups (keep 15-min backups for 24h)
    echo "Managing daily backups..."
    for backup in "$BACKUP_DIR/daily"/*.tar.gz; do
        if [ -f "$backup" ]; then
            local backup_time=$(date -r "$backup" +%s)
            local age=$((current_time - backup_time))
            
            if [ $age -gt 86400 ]; then  # Older than 24h
                # Keep only the latest backup per day
                local backup_date=$(date -r "$backup" +%Y%m%d)
                if [ -f "$BACKUP_DIR/weekly/ALPHA_${backup_date}_latest.tar.gz" ]; then
                    rm "$backup"
                else
                    mv "$backup" "$BACKUP_DIR/weekly/ALPHA_${backup_date}_latest.tar.gz"
                fi
            fi
        fi
    done
    
    # Process weekly backups (after 7 days, keep only one per week)
    echo "Managing weekly backups..."
    local current_week=$(date +%Y%W)
    for backup in "$BACKUP_DIR/weekly"/*.tar.gz; do
        if [ -f "$backup" ]; then
            local backup_time=$(date -r "$backup" +%s)
            local age=$((current_time - backup_time))
            
            if [ $age -gt 7776000 ]; then  # Older than 90 days
                local backup_month=$(date -r "$backup" +%Y%m)
                if [ -f "$BACKUP_DIR/monthly/ALPHA_${backup_month}_latest.tar.gz" ]; then
                    rm "$backup"
                else
                    mv "$backup" "$BACKUP_DIR/monthly/ALPHA_${backup_month}_latest.tar.gz"
                fi
            elif [ $age -gt 604800 ]; then  # Older than 7 days
                local backup_week=$(date -r "$backup" +%Y%W)
                if [ "$backup_week" != "$current_week" ]; then
                    # Keep only latest backup per week
                    local week_latest=$(ls -t "$BACKUP_DIR/weekly"/ALPHA_${backup_week}*.tar.gz 2>/dev/null | head -n1)
                    if [ "$backup" != "$week_latest" ]; then
                        rm "$backup"
                    fi
                fi
            fi
        fi
    done
    
    echo "Backup retention managed"
}

# Function to check if files have changed
files_changed() {
    local current_hash=$(find ALPHA -type f -not -path "*/\.*" -exec md5sum {} \; | sort | md5sum)
    local last_hash=""
    
    if [ -f "$BACKUP_DIR/.last_hash" ]; then
        last_hash=$(cat "$BACKUP_DIR/.last_hash")
    fi
    
    echo "$current_hash" > "$BACKUP_DIR/.last_hash"
    
    if [ "$current_hash" != "$last_hash" ]; then
        return 0  # Files changed
    else
        return 1  # No changes
    fi
}

# Function to check time since last activity
time_since_last_change() {
    if [ ! -f "$LAST_CHANGE_FILE" ]; then
        echo "0"
        return
    fi
    local last_change=$(cat "$LAST_CHANGE_FILE")
    local current_time=$(date +%s)
    echo $((current_time - last_change))
}

# Function to monitor directory for changes
monitor_and_backup() {
    local is_active=false
    local last_backup=0
    
    while true; do
        local current_time=$(date +%s)
        
        if files_changed; then
            if ! $is_active; then
                echo "Activity detected - Creating initial backup"
                is_active=true
                create_backup
                last_backup=$current_time
            elif [ $((current_time - last_backup)) -ge $BACKUP_INTERVAL ]; then
                echo "Creating periodic backup"
                create_backup
                last_backup=$current_time
            fi
        elif $is_active; then
            local idle_time=$(time_since_last_change)
            if [ "$idle_time" -gt "$ACTIVITY_TIMEOUT" ]; then
                echo "No activity for 1 hour - Going idle"
                create_backup  # Final backup before going idle
                is_active=false
            fi
        fi
        sleep 60  # Check every minute
    done
}

# Main backup logic
case "$1" in
    "monitor")
        monitor_and_backup
        ;;
    "force-backup")
        create_backup
        ;;
    "manage-retention")
        manage_retention
        ;;
    *)
        echo "Usage: $0 {monitor|force-backup|manage-retention}"
        exit 1
        ;;
esac 