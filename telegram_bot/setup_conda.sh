#!/bin/bash
# Setup script for Telegram Bot conda environment

set -e

echo "🚀 Setting up Telegram Bot conda environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^tel "; then
    echo "⚠️  Environment 'tel' already exists"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n tel -y
    else
        echo "ℹ️  Using existing environment. Activate with: conda activate tel"
        exit 0
    fi
fi

# Create environment from environment.yml
echo "📦 Creating conda environment 'tel'..."
conda env create -f environment.yml

echo ""
echo "✅ Environment 'tel' created successfully!"
echo ""
echo "📝 Next steps:"
echo "   1. Activate the environment:"
echo "      conda activate tel"
echo ""
echo "   2. Add your Telegram bot token to .env file:"
echo "      echo 'TELEGRAM_BOT_TOKEN=your_token_here' >> ../.env"
echo ""
echo "   3. Run the bot:"
echo "      python -m bot.main"
echo ""

