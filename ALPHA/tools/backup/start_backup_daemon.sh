#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_SCRIPT="$SCRIPT_DIR/smart_backup.sh"
PID_FILE="$HOME/.alpha_backup.pid"
LOG_FILE="$HOME/.alpha_backup.log"

# Function to start the daemon
start_daemon() {
    if [ -f "$PID_FILE" ]; then
        echo "Backup daemon is already running."
        return
    fi

    # Start the backup monitor in the background
    nohup "$BACKUP_SCRIPT" monitor >> "$LOG_FILE" 2>&1 &

    # Save the PID
    echo $! > "$PID_FILE"
    echo "Backup daemon started. PID: $(cat "$PID_FILE")"
    echo "Log file: $LOG_FILE"
}

# Function to stop the daemon
stop_daemon() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Backup daemon is not running."
        return
    fi

    # Kill the process
    kill $(cat "$PID_FILE")
    rm -f "$PID_FILE"
    echo "Backup daemon stopped."
}

# Function to check daemon status
check_status() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Backup daemon is running. PID: $(cat "$PID_FILE")"
        echo "Recent activity:"
        tail -n 5 "$LOG_FILE"
    else
        echo "Backup daemon is not running."
        [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
    fi
}

# Main logic
case "$1" in
    "start")
        start_daemon
        ;;
    "stop")
        stop_daemon
        ;;
    "restart")
        stop_daemon
        sleep 2
        start_daemon
        ;;
    "status")
        check_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
