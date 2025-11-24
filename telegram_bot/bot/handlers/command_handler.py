"""
Command handlers for Telegram Bot
Handles /start, /help, /status, /reset commands
"""

import logging
from telegram import Update
from telegram.ext import ContextTypes
from bot.utils.keyboard import create_main_keyboard
from bot.utils.voice_sender import send_text_and_voice

logger = logging.getLogger(__name__)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user = update.effective_user
    logger.info(f"Start command from user {user.id} (@{user.username})")
    
    # Get session manager from context
    session_manager = context.bot_data.get("session_manager")
    
    if not session_manager:
        await update.message.reply_text(
            "❌ Bot services not initialized. Please contact administrator."
        )
        return
    
    # Check if user already has a session
    existing_session = session_manager.get_session(user.id)
    
    if existing_session:
        await update.message.reply_text(
            f"👋 Welcome back, {user.first_name}!\n\n"
            f"You already have an active session.\n"
            f"Use /status to see your session details.\n"
            f"Use /reset to start a new session.\n\n"
            f"Send me a text message or voice message to continue your conversation!"
        )
    else:
        # Get default scenario from bot_data
        default_scenario = context.bot_data.get("default_scenario", "guided.yml")
        
        # Create new session
        session_id = session_manager.create_session(
            telegram_user_id=user.id,
            patient_name=user.first_name or "User",
            honorific="",
            scenario=default_scenario
        )
        
        # Initialize dialog and send greeting with first question
        dialog_integration = context.bot_data.get("dialog_integration")
        if dialog_integration:
            dialog = dialog_integration.create_dialog(
                patient_name=session["patient_name"],
                honorific=session["honorific"],
                scenario=session["scenario"]
            )
            session_manager.update_session(user.id, dialog_engine=dialog)
            
            # Send greeting and first question
            greeting = dialog.build_greeting()
            first_question = dialog.get_current_question()
            
            keyboard = create_main_keyboard()
            if first_question:
                full_text = (
                    f"👋 Hello {user.first_name}! Welcome to VERA Cloud.\n\n"
                    f"{greeting}\n\n"
                    f"{first_question['prompt']}"
                )
                voice_text = f"{greeting}\n\n{first_question['prompt']}"  # Only bot content for voice
                await send_text_and_voice(update, context, full_text, reply_markup=keyboard, voice_text=voice_text)
            else:
                full_text = (
                    f"👋 Hello {user.first_name}! Welcome to VERA Cloud.\n\n"
                    f"{greeting}\n\n"
                    f"Send me a message to get started!"
                )
                await send_text_and_voice(update, context, full_text, reply_markup=keyboard, voice_text=greeting)
        else:
            keyboard = create_main_keyboard()
            await update.message.reply_text(
                f"👋 Hello {user.first_name}! Welcome to VERA Cloud.\n\n"
                f"✅ New session created!\n\n"
                f"Send me a message to get started!",
                reply_markup=keyboard
            )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    keyboard = create_main_keyboard()
    await update.message.reply_text(
        "📚 **VERA Cloud Bot Commands:**\n\n"
        "/start - Start a new conversation session\n"
        "/help - Show this help message\n"
        "/status - Check your current session status\n"
        "/reset - Reset your session and start fresh\n\n"
        "💬 **How to use:**\n"
        "• Send text messages for quick responses\n"
        "• Send voice messages for natural conversation\n"
        "• I'll guide you through a structured assessment\n\n"
        "❓ **Need help?** Contact support if you encounter any issues.",
        parse_mode="Markdown",
        reply_markup=keyboard
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    user = update.effective_user
    session_manager = context.bot_data.get("session_manager")
    
    if not session_manager:
        await update.message.reply_text("❌ Session manager not available")
        return
    
    session = session_manager.get_session(user.id)
    
    if not session:
        await update.message.reply_text(
            "ℹ️ You don't have an active session.\n"
            "Use /start to create a new session."
        )
    else:
        dialog = session.get("dialog_engine")
        is_complete = dialog.is_complete() if dialog else False
        
        status_text = (
            f"📊 **Session Status**\n\n"
            f"Session ID: `{session['session_id'][:8]}...`\n"
            f"Created: {session['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Last Activity: {session['last_activity'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Status: {'✅ Complete' if is_complete else '🔄 In Progress'}\n"
        )
        
        if dialog:
            progress = dialog.get_progress_percentage()
            status_text += f"Progress: {progress:.1f}%\n"
        
        keyboard = create_main_keyboard()
        await update.message.reply_text(status_text, parse_mode="Markdown", reply_markup=keyboard)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /reset command"""
    user = update.effective_user
    session_manager = context.bot_data.get("session_manager")
    
    if not session_manager:
        await update.message.reply_text("❌ Session manager not available")
        return
    
    # Get default scenario from bot_data
    default_scenario = context.bot_data.get("default_scenario", "guided.yml")
    
    # Delete existing session (this also clears the dialog engine)
    session_manager.delete_session(user.id)
    
    # Create new session with fresh dialog
    session_id = session_manager.create_session(
        telegram_user_id=user.id,
        patient_name=user.first_name or "User",
        honorific="",
        scenario=default_scenario
    )
    
    # Initialize dialog and send greeting with first question
    dialog_integration = context.bot_data.get("dialog_integration")
    if dialog_integration:
        dialog = dialog_integration.create_dialog(
            patient_name=user.first_name or "User",
            honorific="",
            scenario=default_scenario
        )
        session_manager.update_session(user.id, dialog_engine=dialog)
        
        # Send greeting and first question
        greeting = dialog.build_greeting()
        first_question = dialog.get_current_question()
        
        keyboard = create_main_keyboard()
        if first_question:
            full_text = (
                f"🔄 Session reset successfully!\n\n"
                f"{greeting}\n\n"
                f"{first_question['prompt']}"
            )
            voice_text = f"{greeting}\n\n{first_question['prompt']}"  # Only bot content for voice
            await send_text_and_voice(update, context, full_text, reply_markup=keyboard, voice_text=voice_text)
        else:
            full_text = (
                f"🔄 Session reset successfully!\n\n"
                f"{greeting}\n\n"
                f"Send me a message to start a fresh conversation!"
            )
            await send_text_and_voice(update, context, full_text, reply_markup=keyboard, voice_text=greeting)
    else:
        keyboard = create_main_keyboard()
        await update.message.reply_text(
            "🔄 Session reset successfully!\n\n"
            "✅ New session created.\n\n"
            "Send me a message to start a fresh conversation!",
            reply_markup=keyboard
        )
    
    logger.info(f"Session reset for user {user.id}, new session: {session_id}")

