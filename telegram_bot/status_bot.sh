#!/bin/bash
# Check Telegram Bot status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "bot.pid" ]; then
    echo "❌ Bot is not running (no PID file found)"
    exit 1
fi

PID=$(cat bot.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Bot is running"
    echo "   PID: $PID"
    echo "   Started: $(ps -p $PID -o lstart=)"
    echo ""
    echo "Recent log entries:"
    echo "---"
    tail -n 5 logs/bot_*.log 2>/dev/null | tail -n 5 || echo "No log files found"
else
    echo "❌ Bot is not running (PID file exists but process is dead)"
    rm -f bot.pid
    exit 1
fi

