"""
Configuration settings for Telegram Bot
Loads from environment variables and parent .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")

# Bot Settings
BOT_SESSION_TIMEOUT_HOURS = int(os.getenv("BOT_SESSION_TIMEOUT_HOURS", "24"))
BOT_MAX_VOICE_DURATION_SECONDS = int(os.getenv("BOT_MAX_VOICE_DURATION_SECONDS", "60"))
BOT_ENABLE_VOICE_RESPONSES = os.getenv("BOT_ENABLE_VOICE_RESPONSES", "true").lower() == "true"

# Azure Service Configuration (from parent app)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
REDIS_CONNECTION_STRING = os.getenv("REDIS_CONNECTION_STRING")

# Scenario Configuration
SCENARIO_PATH = Path(__file__).parent.parent.parent / "scenarios" / "guided.yml"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

