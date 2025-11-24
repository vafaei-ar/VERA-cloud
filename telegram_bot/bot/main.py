"""
Telegram Bot Main Entry Point
VERA Cloud Telegram Bot
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    TELEGRAM_BOT_TOKEN,
    BOT_SESSION_TIMEOUT_HOURS,
    LOG_LEVEL
)
from bot.utils.logger import setup_logger
from bot.services.session_manager import SessionManager
from bot.services.dialog_integration import DialogIntegration
from bot.handlers.command_handler import (
    start_command,
    help_command,
    status_command,
    reset_command
)
from bot.handlers.callback_handler import callback_handler
from bot.utils.keyboard import create_main_keyboard
from bot.utils.voice_sender import send_text_and_voice, edit_text_and_send_voice

# Set up logging
logger = setup_logger("telegram_bot", LOG_LEVEL)

# Global services
session_manager: SessionManager = None
dialog_integration: DialogIntegration = None
default_scenario: str = "guided.yml"  # Default scenario mode


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    user = update.effective_user
    message_text = update.message.text
    
    logger.info(f"Text message from user {user.id}: {message_text[:50]}...")
    
    # Get or create session
    session = session_manager.get_session(user.id)
    if not session:
        # Create new session with default scenario
        session_id = session_manager.create_session(
            telegram_user_id=user.id,
            patient_name=user.first_name or "User",
            honorific="",
            scenario=default_scenario
        )
        session = session_manager.get_session(user.id)
    
    # Initialize dialog if not already done
    if not session.get("dialog_engine"):
        dialog = dialog_integration.create_dialog(
            patient_name=session["patient_name"],
            honorific=session["honorific"],
            scenario=session.get("scenario", default_scenario)
        )
        session_manager.update_session(user.id, dialog_engine=dialog)
        session = session_manager.get_session(user.id)
        
        # Send greeting and first question for new dialog
        dialog = session["dialog_engine"]
        greeting = dialog.build_greeting()
        first_question = dialog.get_current_question()
        keyboard = create_main_keyboard()
        if first_question:
            full_text = f"{greeting}\n\n{first_question['prompt']}"
            await send_text_and_voice(update, context, full_text, reply_markup=keyboard, voice_text=full_text)
        else:
            await send_text_and_voice(update, context, greeting, reply_markup=keyboard, voice_text=greeting)
        return
    
    dialog = session["dialog_engine"]
    
    # Process user message
    try:
        response_data = await dialog.process_user_response(message_text)
        response_text = response_data.get("message", "I understand. Let me continue.")
        response_type = response_data.get("type", "response")
        
        # Send response with keyboard (text + voice)
        # Voice only includes bot response, text includes full message
        keyboard = create_main_keyboard()
        await send_text_and_voice(update, context, response_text, reply_markup=keyboard, voice_text=response_text)
        
        # Check if conversation is complete
        if dialog.is_complete() or response_type == "completion":
            completion_text = (
                "✅ Thank you for completing the assessment!\n\n"
                "Your responses have been recorded. A member of our care team will review your information."
            )
            await send_text_and_voice(update, context, completion_text, reply_markup=keyboard, voice_text=completion_text)
            logger.info(f"Conversation completed for user {user.id}")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ I'm sorry, I encountered an error processing your message. "
            "Please try again or use /reset to start a new session."
        )


async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages"""
    user = update.effective_user
    voice = update.message.voice
    
    logger.info(f"Voice message from user {user.id}, duration: {voice.duration}s")
    
    # Check voice duration
    if voice.duration > 60:
        await update.message.reply_text(
            "⚠️ Voice messages longer than 60 seconds are not supported. "
            "Please send a shorter message or use text instead."
        )
        return
    
    # Send processing message
    processing_msg = await update.message.reply_text("🎤 Processing your voice message...")
    
    try:
        # Download voice file
        voice_file = await context.bot.get_file(voice.file_id)
        
        # Download as bytearray and convert to bytes
        audio_bytearray = await voice_file.download_as_bytearray()
        audio_data = bytes(audio_bytearray)
        
        # Transcribe audio
        text, confidence = await dialog_integration.transcribe_audio(audio_data)
        
        logger.info(f"Transcribed voice: {text[:50]}... (confidence: {confidence:.2f})")
        
        # Update processing message
        await processing_msg.edit_text(f"✅ Transcribed: \"{text}\"\n\nProcessing your message...")
        
        # Get or create session
        session = session_manager.get_session(user.id)
        if not session:
            session_id = session_manager.create_session(
                telegram_user_id=user.id,
                patient_name=user.first_name or "User",
                honorific="",
                scenario=default_scenario
            )
            session = session_manager.get_session(user.id)
        
        # Initialize dialog if needed
        if not session.get("dialog_engine"):
            dialog = dialog_integration.create_dialog(
                patient_name=session["patient_name"],
                honorific=session["honorific"],
                scenario=session.get("scenario", default_scenario)
            )
            session_manager.update_session(user.id, dialog_engine=dialog)
            session = session_manager.get_session(user.id)
            
            # Send greeting and first question for new dialog
            dialog = session["dialog_engine"]
            greeting = dialog.build_greeting()
            first_question = dialog.get_current_question()
            keyboard = create_main_keyboard()
            if first_question:
                full_text = f"✅ Transcribed: \"{text}\"\n\n{greeting}\n\n{first_question['prompt']}"
                voice_text = f"{greeting}\n\n{first_question['prompt']}"  # Only bot content for voice
                await edit_text_and_send_voice(processing_msg, context, full_text, reply_markup=keyboard, voice_text=voice_text)
            else:
                full_text = f"✅ Transcribed: \"{text}\"\n\n{greeting}"
                await edit_text_and_send_voice(processing_msg, context, full_text, reply_markup=keyboard, voice_text=greeting)
            return
        
        dialog = session["dialog_engine"]
        
        # Process transcribed text
        response_data = await dialog.process_user_response(text, confidence=confidence)
        response_text = response_data.get("message", "I understand. Let me continue.")
        
        # Send text response with keyboard (text + voice)
        # Text message includes transcription info, voice only includes bot response
        keyboard = create_main_keyboard()
        response_full_text = f"💬 Your message: \"{text}\"\n\n🤖 {response_text}"
        await edit_text_and_send_voice(
            processing_msg, 
            context, 
            response_full_text, 
            reply_markup=keyboard,
            voice_text=response_text  # Only bot response for voice, no metadata
        )
        
        # Check if conversation is complete
        if dialog.is_complete():
            completion_text = (
                "✅ Thank you for completing the assessment!\n\n"
                "Your responses have been recorded."
            )
            await send_text_and_voice(update, context, completion_text, reply_markup=keyboard, voice_text=completion_text)
        
    except Exception as e:
        logger.error(f"Error processing voice message: {e}", exc_info=True)
        await processing_msg.edit_text(
            "❌ I'm sorry, I couldn't process your voice message. "
            "Please try sending it again or use text instead."
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)
    
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again or use /reset to start fresh."
            )
        except:
            pass


def main():
    """Main function to run the bot"""
    global session_manager, dialog_integration, default_scenario
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="VERA Cloud Telegram Bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["guided", "rag"],
        default="guided",
        help="Conversation mode: 'guided' for guided.yml or 'rag' for rag_enhanced.yml (default: guided)"
    )
    args = parser.parse_args()
    
    # Set scenario based on mode
    if args.mode == "rag":
        default_scenario = "rag_enhanced.yml"
        logger.info("Starting bot in RAG-enhanced mode")
    else:
        default_scenario = "guided.yml"
        logger.info("Starting bot in guided mode")
    
    logger.info("Starting VERA Cloud Telegram Bot...")
    
    # Initialize services
    try:
        from config import settings
        
        # Initialize session manager
        session_manager = SessionManager(timeout_hours=BOT_SESSION_TIMEOUT_HOURS)
        logger.info("Session manager initialized")
        
        # Initialize dialog integration
        dialog_integration = DialogIntegration(settings)
        logger.info("Dialog integration initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        sys.exit(1)
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Store services in bot_data for access in handlers
    application.bot_data["session_manager"] = session_manager
    application.bot_data["dialog_integration"] = dialog_integration
    application.bot_data["default_scenario"] = default_scenario
    
    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("reset", reset_command))
    
    # Register callback handler for inline keyboard buttons
    from telegram.ext import CallbackQueryHandler
    application.add_handler(CallbackQueryHandler(callback_handler))
    
    # Register message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
    application.add_handler(MessageHandler(filters.VOICE, voice_message_handler))
    
    # Register error handler
    application.add_error_handler(error_handler)
    
    logger.info("Bot handlers registered")
    logger.info("Bot is running... Press Ctrl+C to stop")
    
    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

