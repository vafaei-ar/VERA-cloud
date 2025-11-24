# VERA Cloud Telegram Bot

A Telegram bot that allows users to interact with VERA Cloud via voice messages and text chat.

## Features

- ✅ **Text Chat**: Users can send text messages
- ✅ **Voice Chat**: Users can send voice messages (automatically transcribed)
- ✅ **Voice Responses**: Bot responds with both text and voice
- ✅ **Medical Scenarios**: Supports guided and RAG-enhanced conversation modes
- ✅ **Session Management**: Each user gets their own conversation session
- ✅ **Background Service**: Can run as a background service

## Prerequisites

- Python 3.11+
- Conda (recommended) or pip
- Telegram Bot Token (get from [@BotFather](https://t.me/BotFather))
- Azure service credentials (in parent `.env` file)

## Getting Your Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow prompts to create your bot
4. Copy the token (format: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
5. Add to `.env` file: `TELEGRAM_BOT_TOKEN=your_token_here`

## Quick Start

### Option A: Using Conda (Recommended)

```bash
cd telegram_bot

# Create and activate conda environment
conda env create -f environment.yml
conda activate tel

# Or use the setup script
chmod +x setup_conda.sh
./setup_conda.sh
conda activate tel
```

### Option B: Using pip

```bash
cd telegram_bot

# First install parent app dependencies
cd ..
pip install -r requirements.txt

# Then install bot-specific dependencies
cd telegram_bot
pip install -r requirements.txt
```

### Step 2: Configure Environment

The bot will automatically use the Azure credentials from the parent `.env` file. You just need to add the Telegram bot token:

```bash
# From the telegram_bot directory, create a .env file or add to parent .env
echo "TELEGRAM_BOT_TOKEN=your_bot_token_here" >> ../.env
```

Or edit the parent `.env` file directly and add:
```bash
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

### Step 3: Run the Bot

**If using conda:**
```bash
conda activate tel
cd telegram_bot
python -m bot.main
```

**If using pip:**
```bash
# From telegram_bot directory
python -m bot.main
```

Or from the project root:
```bash
python -m telegram_bot.bot.main
```

### Step 4: Test the Bot

1. Find your bot on Telegram (search for the username you gave it)
2. Send `/start` command
3. Try sending a text message
4. Try sending a voice message

## Service Management

### Start as Background Service

```bash
./start_bot.sh          # Guided mode (default)
./start_bot.sh rag      # RAG mode
```

The bot will:
- Run in background
- Continue running after closing terminal
- Log to `logs/bot_YYYYMMDD_HHMMSS.log`
- Save PID to `bot.pid`

### Stop the Service

```bash
./stop_bot.sh
```

### Check Status

```bash
./status_bot.sh
```

### View Logs

```bash
tail -f logs/bot_*.log
```

## Configuration

### Required Environment Variables

Add to parent `.env` file:

```bash
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Azure Services (already configured)
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=...
AZURE_SEARCH_ENDPOINT=...
AZURE_SEARCH_API_KEY=...
REDIS_CONNECTION_STRING=...
```

### Optional Settings

```bash
BOT_SESSION_TIMEOUT_HOURS=24
BOT_MAX_VOICE_DURATION_SECONDS=60
BOT_ENABLE_VOICE_RESPONSES=true
```

## Architecture

- **Isolated**: All bot code in `telegram_bot/` directory
- **No modifications**: Doesn't change main app code
- **Service reuse**: Imports and uses existing Azure services
- **Independent**: Can run separately from main web app

## Troubleshooting

### Bot not responding?
- Check bot token in `.env`
- Verify internet connection
- Check logs: `tail -f logs/bot_*.log`

### Voice messages not working?
- Verify Azure Speech Service credentials
- Check REST API implementation (no SDK dependencies)

### Import errors?
- Ensure conda environment is activated: `conda activate tel`
- Verify you're in the correct directory

## Project Structure

```
telegram_bot/
├── bot/                    # Bot code
│   ├── main.py            # Entry point
│   ├── handlers/          # Message/command handlers
│   ├── services/          # Session & dialog integration
│   └── utils/             # Utilities
├── config/                # Configuration
├── start_bot.sh           # Start service
├── stop_bot.sh            # Stop service
├── status_bot.sh           # Check status
├── environment.yml        # Conda environment
└── requirements.txt       # Python dependencies
```

---

**Ready to use!** 🚀

