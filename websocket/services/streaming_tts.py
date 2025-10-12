"""
Streaming TTS Service for VERA Cloud
Real-time text-to-speech using Azure Speech Service
"""

import asyncio
import logging
from typing import Callable, Optional, Dict, Any, AsyncGenerator
import io
from azure.cognitiveservices.speech import (
    SpeechConfig, SpeechSynthesizer, AudioConfig,
    ResultReason, CancellationReason
)

logger = logging.getLogger(__name__)

class StreamingTTS:
    def __init__(self, speech_key: str, region: str):
        self.speech_key = speech_key
        self.region = region
        self.speech_config = SpeechConfig(subscription=speech_key, region=region)
        self.speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
        self.speech_config.set_speech_synthesis_output_format(
            "audio-16khz-32kbitrate-mono-mp3"
        )
        
        # State
        self.is_processing = False
        self.current_voice = "en-US-AriaNeural"
        self.current_rate = 1.0
        
        # Available voices
        self.available_voices = {
            "en-US-AriaNeural": {"gender": "female", "style": "friendly"},
            "en-US-JennyNeural": {"gender": "female", "style": "warm"},
            "en-US-GuyNeural": {"gender": "male", "style": "professional"},
            "en-US-DavisNeural": {"gender": "male", "style": "calm"},
            "en-US-EmmaNeural": {"gender": "female", "style": "empathetic"},
            "en-US-BrianNeural": {"gender": "male", "style": "authoritative"}
        }
        
        logger.info("StreamingTTS initialized")
    
    async def synthesize_text(self, text: str, voice: str = None, rate: float = 1.0) -> bytes:
        """Synthesize text to speech and return audio bytes"""
        try:
            if not text.strip():
                return b""
            
            # Configure voice and rate
            if voice and voice in self.available_voices:
                self.speech_config.speech_synthesis_voice_name = voice
                self.current_voice = voice
            else:
                self.speech_config.speech_synthesis_voice_name = self.current_voice
            
            self.current_rate = rate
            
            # Create SSML for better control
            ssml = self._create_ssml(text, voice or self.current_voice, rate)
            
            # Create synthesizer
            synthesizer = SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )
            
            # Synthesize
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Synthesized {len(text)} characters to {len(result.audio_data)} bytes")
                return result.audio_data
            else:
                logger.error(f"Speech synthesis failed: {result.reason}")
                return b""
                
        except Exception as e:
            logger.error(f"Text synthesis failed: {e}")
            return b""
    
    async def synthesize_streaming(self, text: str, voice: str = None, 
                                 rate: float = 1.0) -> AsyncGenerator[bytes, None]:
        """Stream TTS audio as it's generated"""
        try:
            if not text.strip():
                return
            
            # Configure voice and rate
            if voice and voice in self.available_voices:
                self.speech_config.speech_synthesis_voice_name = voice
                self.current_voice = voice
            else:
                self.speech_config.speech_synthesis_voice_name = self.current_voice
            
            self.current_rate = rate
            
            # Create SSML for better control
            ssml = self._create_ssml(text, voice or self.current_voice, rate)
            
            # Create synthesizer
            synthesizer = SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )
            
            # Synthesize
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == ResultReason.SynthesizingAudioCompleted:
                # Stream audio in chunks for better performance
                audio_data = result.audio_data
                chunk_size = 4096
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    yield chunk
                    await asyncio.sleep(0.01)  # Small delay for smooth playback
            else:
                logger.error(f"Streaming synthesis failed: {result.reason}")
                
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
    
    async def synthesize_with_pauses(self, text: str, voice: str = None, 
                                   rate: float = 1.0) -> bytes:
        """Synthesize text with pause markers for natural conversation flow"""
        try:
            if not text.strip():
                return b""
            
            # Configure voice and rate
            if voice and voice in self.available_voices:
                self.speech_config.speech_synthesis_voice_name = voice
                self.current_voice = voice
            else:
                self.speech_config.speech_synthesis_voice_name = self.current_voice
            
            self.current_rate = rate
            
            # Create SSML with pause processing
            ssml = self._create_ssml_with_pauses(text, voice or self.current_voice, rate)
            
            # Create synthesizer
            synthesizer = SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )
            
            # Synthesize
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Synthesized with pauses: {len(text)} chars to {len(result.audio_data)} bytes")
                return result.audio_data
            else:
                logger.error(f"Pause synthesis failed: {result.reason}")
                return b""
                
        except Exception as e:
            logger.error(f"Pause synthesis failed: {e}")
            return b""
    
    def _create_ssml(self, text: str, voice: str, rate: float) -> str:
        """Create SSML for text synthesis"""
        try:
            # Escape special characters
            escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{voice}">
                    <prosody rate="{rate}" pitch="0%">
                        {escaped_text}
                    </prosody>
                </voice>
            </speak>
            """
            
            return ssml.strip()
            
        except Exception as e:
            logger.error(f"SSML creation failed: {e}")
            return f"<speak><voice name='{voice}'>{text}</voice></speak>"
    
    def _create_ssml_with_pauses(self, text: str, voice: str, rate: float) -> str:
        """Create SSML with pause markers for natural conversation flow"""
        try:
            import re
            
            # Replace pause markers [pause=1000] with SSML breaks
            ssml_text = re.sub(
                r'\[pause=(\d+)\]',
                r'<break time="\1ms"/>',
                text
            )
            
            # Escape special characters
            escaped_text = ssml_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{voice}">
                    <prosody rate="{rate}" pitch="0%">
                        {escaped_text}
                    </prosody>
                </voice>
            </speak>
            """
            
            return ssml.strip()
            
        except Exception as e:
            logger.error(f"SSML with pauses creation failed: {e}")
            return self._create_ssml(text, voice, rate)
    
    def get_available_voices(self) -> Dict[str, Dict[str, str]]:
        """Get available neural voices"""
        return self.available_voices
    
    def set_default_voice(self, voice: str) -> bool:
        """Set default voice"""
        try:
            if voice in self.available_voices:
                self.current_voice = voice
                self.speech_config.speech_synthesis_voice_name = voice
                logger.info(f"Default voice set to: {voice}")
                return True
            else:
                logger.warning(f"Voice not available: {voice}")
                return False
        except Exception as e:
            logger.error(f"Failed to set default voice: {e}")
            return False
    
    async def get_tts_stats(self) -> Dict[str, Any]:
        """Get TTS processing statistics"""
        try:
            return {
                "is_processing": self.is_processing,
                "current_voice": self.current_voice,
                "current_rate": self.current_rate,
                "available_voices": list(self.available_voices.keys()),
                "speech_config_voice": self.speech_config.speech_synthesis_voice_name
            }
        except Exception as e:
            logger.error(f"Failed to get TTS stats: {e}")
            return {"error": str(e)}
    
    async def test_synthesis(self, text: str = "Hello, this is a test.") -> bytes:
        """Test synthesis with default settings"""
        try:
            return await self.synthesize_text(text)
        except Exception as e:
            logger.error(f"Test synthesis failed: {e}")
            return b""
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            self.is_processing = False
            logger.info("StreamingTTS cleanup completed")
        except Exception as e:
            logger.error(f"TTS cleanup failed: {e}")
