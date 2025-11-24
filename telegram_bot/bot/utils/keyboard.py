"""
Keyboard utilities for Telegram Bot
Creates inline keyboards with buttons
"""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def create_main_keyboard() -> InlineKeyboardMarkup:
    """Create main keyboard with common commands"""
    keyboard = [
        [
            InlineKeyboardButton("🔄 Reset Session", callback_data="reset"),
            InlineKeyboardButton("📊 Status", callback_data="status")
        ],
        [
            InlineKeyboardButton("❓ Help", callback_data="help")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_reset_confirm_keyboard() -> InlineKeyboardMarkup:
    """Create confirmation keyboard for reset"""
    keyboard = [
        [
            InlineKeyboardButton("✅ Yes, Reset", callback_data="reset_confirm"),
            InlineKeyboardButton("❌ Cancel", callback_data="reset_cancel")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_simple_keyboard(buttons: list) -> InlineKeyboardMarkup:
    """
    Create a simple keyboard from a list of button tuples
    
    Args:
        buttons: List of (text, callback_data) tuples
    
    Returns:
        InlineKeyboardMarkup
    """
    keyboard = [
        [InlineKeyboardButton(text, callback_data=callback_data)]
        for text, callback_data in buttons
    ]
    return InlineKeyboardMarkup(keyboard)

