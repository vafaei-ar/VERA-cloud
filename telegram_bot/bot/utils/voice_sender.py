"""
Utility for sending text and voice messages together
"""

import logging
import io
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

async def send_text_and_voice(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup=None,
    voice_enabled: bool = True,
    voice_text: str = None
):
    """
    Send both text and voice message
    
    Args:
        update: Telegram update object
        context: Bot context
        text: Full text to send (includes metadata for display)
        reply_markup: Optional keyboard markup
        voice_enabled: Whether to send voice (default: True)
        voice_text: Text to use for voice (only bot response, no metadata). If None, uses text.
    """
    try:
        # Send text message (full text with metadata)
        await update.message.reply_text(text, reply_markup=reply_markup)
        
        # Send voice message if enabled (only bot response, no metadata)
        if voice_enabled:
            dialog_integration = context.bot_data.get("dialog_integration")
            if dialog_integration and dialog_integration.azure_speech:
                try:
                    # Use voice_text if provided (clean bot response), otherwise use full text
                    text_for_voice = voice_text if voice_text else text
                    # Synthesize text to speech
                    audio_data = await dialog_integration.synthesize_text(text_for_voice)
                    
                    if audio_data and len(audio_data) > 0:
                        # Convert MP3 to OGG if needed, or send as MP3
                        # Telegram accepts MP3 for voice messages
                        audio_file = io.BytesIO(audio_data)
                        audio_file.name = "voice.mp3"
                        
                        # Send as voice message
                        await update.message.reply_voice(
                            voice=audio_file,
                            caption="🔊 Audio version"
                        )
                        logger.info(f"Sent voice message for text: {text[:50]}...")
                    else:
                        logger.warning("TTS returned empty audio, skipping voice message")
                        
                except Exception as e:
                    logger.error(f"Failed to send voice message: {e}", exc_info=True)
                    # Continue without voice - text was already sent
            else:
                logger.debug("Voice disabled or TTS service not available")
                
    except Exception as e:
        logger.error(f"Error sending text and voice: {e}", exc_info=True)
        # Fallback to text only
        try:
            await update.message.reply_text(text, reply_markup=reply_markup)
        except:
            pass

async def edit_text_and_send_voice(
    message_to_edit,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup=None,
    voice_enabled: bool = True,
    voice_text: str = None
):
    """
    Edit a message with text and send voice separately
    
    Args:
        message_to_edit: Message object to edit
        context: Bot context
        text: Full text to send (includes metadata for display)
        reply_markup: Optional keyboard markup
        voice_enabled: Whether to send voice (default: True)
        voice_text: Text to use for voice (only bot response, no metadata). If None, uses text.
    """
    try:
        # Edit text message (full text with metadata)
        await message_to_edit.edit_text(text, reply_markup=reply_markup)
        
        # Send voice message if enabled (only bot response, no metadata)
        if voice_enabled:
            dialog_integration = context.bot_data.get("dialog_integration")
            if dialog_integration and dialog_integration.azure_speech:
                try:
                    # Use voice_text if provided (clean bot response), otherwise use full text
                    text_for_voice = voice_text if voice_text else text
                    # Synthesize text to speech
                    audio_data = await dialog_integration.synthesize_text(text_for_voice)
                    
                    if audio_data and len(audio_data) > 0:
                        # Send as voice message using the message's chat
                        audio_file = io.BytesIO(audio_data)
                        audio_file.name = "voice.mp3"
                        
                        # Use the message's reply_voice method
                        await message_to_edit.reply_voice(
                            voice=audio_file,
                            caption="🔊 Audio version"
                        )
                        logger.info(f"Sent voice message for text: {text[:50]}...")
                    else:
                        logger.warning("TTS returned empty audio, skipping voice message")
                        
                except Exception as e:
                    logger.error(f"Failed to send voice message: {e}", exc_info=True)
                    # Continue without voice - text was already sent
                    
    except Exception as e:
        logger.error(f"Error editing text and sending voice: {e}", exc_info=True)

