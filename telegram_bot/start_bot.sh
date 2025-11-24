#!/bin/bash
# Start Telegram Bot as background service

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if bot is already running
if [ -f "bot.pid" ]; then
    PID=$(cat bot.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "❌ Bot is already running (PID: $PID)"
        echo "   Use './stop_bot.sh' to stop it first"
        exit 1
    else
        # PID file exists but process is dead, remove it
        rm -f bot.pid
    fi
fi

# Activate conda environment
echo "🔧 Activating conda environment 'tel'..."
# Try to find conda
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "❌ Conda not found. Please activate 'tel' environment manually and run:"
    echo "   python -m bot.main --mode $MODE"
    exit 1
fi
conda activate tel

# Check mode argument
MODE=${1:-guided}
if [ "$MODE" != "guided" ] && [ "$MODE" != "rag" ]; then
    echo "⚠️  Invalid mode: $MODE. Using 'guided' instead."
    MODE="guided"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start bot in background
echo "🚀 Starting Telegram bot in $MODE mode..."
nohup python -m bot.main --mode "$MODE" > "logs/bot_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
BOT_PID=$!

# Save PID
echo $BOT_PID > bot.pid

# Wait a moment to check if it started successfully
sleep 2

if ps -p $BOT_PID > /dev/null 2>&1; then
    echo "✅ Bot started successfully!"
    echo "   PID: $BOT_PID"
    echo "   Mode: $MODE"
    echo "   Log file: logs/bot_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "To stop the bot, run: ./stop_bot.sh"
    echo "To view logs: tail -f logs/bot_*.log"
else
    echo "❌ Bot failed to start. Check logs for errors."
    rm -f bot.pid
    exit 1
fi

