#!/bin/bash
# Stop Telegram Bot service

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if PID file exists
if [ ! -f "bot.pid" ]; then
    echo "⚠️  No PID file found. Bot may not be running."
    exit 0
fi

PID=$(cat bot.pid)

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "🛑 Stopping bot (PID: $PID)..."
    kill $PID
    
    # Wait for process to stop (max 10 seconds)
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "✅ Bot stopped successfully"
            rm -f bot.pid
            exit 0
        fi
        sleep 1
    done
    
    # If still running, force kill
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Bot didn't stop gracefully, forcing kill..."
        kill -9 $PID
        sleep 1
        rm -f bot.pid
        echo "✅ Bot force-stopped"
    fi
else
    echo "⚠️  Process $PID is not running. Cleaning up PID file."
    rm -f bot.pid
fi

