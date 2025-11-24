"""
Azure Speech Service REST API Integration for Telegram Bot
Uses REST API instead of SDK to avoid native dependency issues
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

class AzureSpeechServiceREST:
    """
    Azure Speech Service using REST API (no SDK dependencies)
    Specifically for Telegram bot to avoid native library issues
    """
    
    def __init__(self, speech_key: str, region: str):
        self.speech_key = speech_key
        self.region = region
        self.default_voice = "en-US-AriaNeural"
    
    async def synthesize_text(self, text: str, voice: str = None, rate: float = 1.0) -> bytes:
        """
        Synthesize text to speech and return audio bytes using REST API
        
        Args:
            text: Text to synthesize
            voice: Voice name (default: en-US-AriaNeural)
            rate: Speech rate (default: 1.0)
            
        Returns:
            Audio bytes in MP3 format
        """
        try:
            if not text or not text.strip():
                return b""
            
            # Use default voice if not specified
            if not voice:
                voice = self.default_voice
            
            # Create SSML for TTS
            ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{voice}">
                    <prosody rate="{rate}" pitch="0%">
                        {text}
                    </prosody>
                </voice>
            </speak>"""
            
            # Azure Speech REST API endpoint
            endpoint = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
            
            # Headers
            headers = {
                "Ocp-Apim-Subscription-Key": self.speech_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3"
            }
            
            # Make HTTP request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    content=ssml.encode('utf-8')
                )
                
                if response.status_code == 200:
                    audio_data = response.content
                    logger.info(f"Synthesized {len(text)} characters to {len(audio_data)} bytes via REST API")
                    return audio_data
                else:
                    logger.error(f"TTS REST API failed: {response.status_code} - {response.text}")
                    return b""
                    
        except Exception as e:
            logger.error(f"REST API TTS synthesis failed: {e}")
            return b""
    
    def get_available_voices(self) -> dict:
        """Get available neural voices"""
        return {
            "en-US-AriaNeural": {"gender": "female", "style": "friendly"},
            "en-US-JennyNeural": {"gender": "female", "style": "warm"},
            "en-US-GuyNeural": {"gender": "male", "style": "professional"},
            "en-US-DavisNeural": {"gender": "male", "style": "calm"},
            "en-US-EmmaNeural": {"gender": "female", "style": "empathetic"},
            "en-US-BrianNeural": {"gender": "male", "style": "authoritative"}
        }

