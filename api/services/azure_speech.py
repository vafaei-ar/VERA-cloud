"""
Azure Speech Service Integration for VERA Cloud
Handles real-time STT and TTS with streaming
"""

import asyncio
import logging
from typing import Optional, AsyncGenerator, Dict, Any
import io
import json
from azure.cognitiveservices.speech import (
    SpeechConfig, SpeechRecognizer, SpeechSynthesizer,
    AudioConfig, AudioDataStream,
    ResultReason, CancellationReason, PropertyId, SpeechSynthesisOutputFormat
)

logger = logging.getLogger(__name__)

class AzureSpeechService:
    def __init__(self, speech_key: str, region: str):
        self.speech_key = speech_key
        self.region = region
        self.speech_config = SpeechConfig(subscription=speech_key, region=region)
        self.speech_config.speech_recognition_language = "en-US"
        self.speech_config.enable_dictation = True
        
        # Configure for real-time processing
        self.speech_config.set_property(PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "5000")
        self.speech_config.set_property(PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "2000")
        
        # TTS configuration
        self.speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
        self.speech_config.set_speech_synthesis_output_format(
            SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
    
    async def start_streaming_recognition(self, on_partial_result, on_final_result, on_error=None):
        """Start real-time streaming speech recognition"""
        try:
            # Create audio stream
            audio_stream = PushAudioInputStream()
            audio_config = AudioConfig(stream=audio_stream)
            
            # Create recognizer
            recognizer = SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Set up event handlers
            def on_recognizing(evt):
                if evt.result.reason == ResultReason.RecognizingSpeech:
                    asyncio.create_task(on_partial_result({
                        "text": evt.result.text,
                        "confidence": 0.0,  # Partial results don't have confidence
                        "is_final": False
                    }))
            
            def on_recognized(evt):
                if evt.result.reason == ResultReason.RecognizedSpeech:
                    asyncio.create_task(on_final_result({
                        "text": evt.result.text,
                        "confidence": 0.9,  # High confidence for final results
                        "is_final": True
                    }))
            
            def on_canceled(evt):
                if evt.reason == CancellationReason.Error:
                    error_msg = f"Recognition canceled: {evt.error_details}"
                    logger.error(error_msg)
                    if on_error:
                        asyncio.create_task(on_error(error_msg))
            
            # Connect event handlers
            recognizer.recognizing.connect(on_recognizing)
            recognizer.recognized.connect(on_recognized)
            recognizer.canceled.connect(on_canceled)
            
            # Start continuous recognition
            recognizer.start_continuous_recognition()
            
            logger.info("Started streaming speech recognition")
            return audio_stream, recognizer
            
        except Exception as e:
            logger.error(f"Failed to start streaming recognition: {e}")
            raise
    
    async def stop_streaming_recognition(self, recognizer):
        """Stop streaming recognition"""
        try:
            recognizer.stop_continuous_recognition()
            logger.info("Stopped streaming speech recognition")
        except Exception as e:
            logger.error(f"Error stopping recognition: {e}")
    
    async def synthesize_text(self, text: str, voice: str = None, rate: float = 1.0) -> bytes:
        """Synthesize text to speech and return audio bytes"""
        try:
            # Configure voice if specified
            if voice:
                self.speech_config.speech_synthesis_voice_name = voice
            
            # Create SSML for better control
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{self.speech_config.speech_synthesis_voice_name}">
                    <prosody rate="{rate}" pitch="0%">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
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
    
    async def synthesize_streaming(self, text: str, voice: str = None, rate: float = 1.0) -> AsyncGenerator[bytes, None]:
        """Stream TTS audio as it's generated"""
        try:
            # Configure voice if specified
            if voice:
                self.speech_config.speech_synthesis_voice_name = voice
            
            # Create SSML for better control
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{self.speech_config.speech_synthesis_voice_name}">
                    <prosody rate="{rate}" pitch="0%">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
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
    
    def get_available_voices(self) -> Dict[str, Any]:
        """Get available neural voices"""
        return {
            "en-US-AriaNeural": {"gender": "female", "style": "friendly"},
            "en-US-JennyNeural": {"gender": "female", "style": "warm"},
            "en-US-GuyNeural": {"gender": "male", "style": "professional"},
            "en-US-DavisNeural": {"gender": "male", "style": "calm"},
            "en-US-EmmaNeural": {"gender": "female", "style": "empathetic"},
            "en-US-BrianNeural": {"gender": "male", "style": "authoritative"}
        }
    
    async def synthesize_with_pauses(self, text: str, voice: str = None, rate: float = 1.0) -> bytes:
        """Synthesize text with pause markers for natural conversation flow"""
        try:
            # Parse pause markers [pause=1000] and convert to SSML
            import re
            
            # Replace pause markers with SSML breaks
            ssml_text = re.sub(
                r'\[pause=(\d+)\]',
                r'<break time="\1ms"/>',
                text
            )
            
            # Configure voice if specified
            if voice:
                self.speech_config.speech_synthesis_voice_name = voice
            
            # Create SSML
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{self.speech_config.speech_synthesis_voice_name}">
                    <prosody rate="{rate}" pitch="0%">
                        {ssml_text}
                    </prosody>
                </voice>
            </speak>
            """
            
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
