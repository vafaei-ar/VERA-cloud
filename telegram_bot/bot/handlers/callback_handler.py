"""
Callback handlers for inline keyboard buttons
"""

import logging
from telegram import Update
from telegram.ext import ContextTypes
from bot.handlers.command_handler import (
    help_command,
    status_command,
    reset_command
)
from bot.utils.keyboard import create_main_keyboard, create_reset_confirm_keyboard
from bot.utils.voice_sender import send_text_and_voice

logger = logging.getLogger(__name__)

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard button callbacks"""
    query = update.callback_query
    await query.answer()  # Acknowledge the callback
    
    user = query.from_user
    callback_data = query.data
    
    logger.info(f"Callback from user {user.id}: {callback_data}")
    
    if callback_data == "help":
        # Show help
        keyboard = create_main_keyboard()
        await query.edit_message_text(
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
    
    elif callback_data == "status":
        # Show status
        session_manager = context.bot_data.get("session_manager")
        if not session_manager:
            await query.edit_message_text("❌ Session manager not available")
            return
        
        session = session_manager.get_session(user.id)
        if not session:
            keyboard = create_main_keyboard()
            await query.edit_message_text(
                "ℹ️ You don't have an active session.\n"
                "Use /start to create a new session.",
                reply_markup=keyboard
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
            await query.edit_message_text(status_text, parse_mode="Markdown", reply_markup=keyboard)
    
    elif callback_data == "reset":
        # Show reset confirmation
        keyboard = create_reset_confirm_keyboard()
        await query.edit_message_text(
            "🔄 **Reset Session**\n\n"
            "Are you sure you want to reset your current session?\n"
            "This will start a fresh conversation.",
            parse_mode="Markdown",
            reply_markup=keyboard
        )
    
    elif callback_data == "reset_confirm":
        # Confirm reset
        session_manager = context.bot_data.get("session_manager")
        if not session_manager:
            await query.edit_message_text("❌ Session manager not available")
            return
        
        # Get default scenario
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
                # Edit the message and send voice separately
                await query.edit_message_text(full_text, reply_markup=keyboard)
                # Send voice message (only bot content, no metadata)
                try:
                    dialog_integration = context.bot_data.get("dialog_integration")
                    if dialog_integration and dialog_integration.azure_speech:
                        import io
                        audio_data = await dialog_integration.synthesize_text(voice_text)
                        if audio_data and len(audio_data) > 0:
                            audio_file = io.BytesIO(audio_data)
                            audio_file.name = "voice.mp3"
                            await query.message.reply_voice(
                                voice=audio_file,
                                caption="🔊 Audio version"
                            )
                except Exception as e:
                    logger.error(f"Failed to send voice in callback: {e}")
            else:
                full_text = (
                    f"🔄 Session reset successfully!\n\n"
                    f"{greeting}\n\n"
                    f"Send me a message to start a fresh conversation!"
                )
                await query.edit_message_text(full_text, reply_markup=keyboard)
                # Send voice message (only bot content, no metadata)
                try:
                    dialog_integration = context.bot_data.get("dialog_integration")
                    if dialog_integration and dialog_integration.azure_speech:
                        import io
                        audio_data = await dialog_integration.synthesize_text(greeting)
                        if audio_data and len(audio_data) > 0:
                            audio_file = io.BytesIO(audio_data)
                            audio_file.name = "voice.mp3"
                            await query.message.reply_voice(
                                voice=audio_file,
                                caption="🔊 Audio version"
                            )
                except Exception as e:
                    logger.error(f"Failed to send voice in callback: {e}")
        else:
            keyboard = create_main_keyboard()
            await query.edit_message_text(
                "🔄 Session reset successfully!\n\n"
                "✅ New session created.\n\n"
                "Send me a message to start a fresh conversation!",
                reply_markup=keyboard
            )
        
        logger.info(f"Session reset for user {user.id}, new session: {session_id}")
    
    elif callback_data == "reset_cancel":
        # Cancel reset
        keyboard = create_main_keyboard()
        await query.edit_message_text(
            "❌ Reset cancelled.\n\n"
            "Your current session is still active.",
            reply_markup=keyboard
        )

