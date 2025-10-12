"""
Streaming ASR Service for VERA Cloud
Real-time speech recognition using Azure Speech Service
"""

import asyncio
import logging
from typing import Callable, Optional, Dict, Any
import io
import wave
import numpy as np
from azure.cognitiveservices.speech import (
    SpeechConfig, SpeechRecognizer, AudioConfig,
    PushAudioInputStream, ResultReason, CancellationReason
)

logger = logging.getLogger(__name__)

class StreamingASR:
    def __init__(self, speech_key: str, region: str):
        self.speech_key = speech_key
        self.region = region
        self.speech_config = SpeechConfig(subscription=speech_key, region=region)
        self.speech_config.speech_recognition_language = "en-US"
        self.speech_config.enable_dictation = True
        
        # Configure for real-time processing
        self.speech_config.set_property("speechcontext-InitialSilenceTimeoutMs", "5000")
        self.speech_config.set_property("speechcontext-EndSilenceTimeoutMs", "2000")
        self.speech_config.set_property("speechcontext-InitialSilenceTimeoutMs", "5000")
        
        # State
        self.audio_stream = None
        self.recognizer = None
        self.is_running = False
        self.partial_callback = None
        self.final_callback = None
        self.error_callback = None
        
        logger.info("StreamingASR initialized")
    
    async def start_streaming(self, session_id: str, 
                            on_partial_result: Callable[[Dict], None],
                            on_final_result: Callable[[Dict], None],
                            on_error: Optional[Callable[[str], None]] = None):
        """Start streaming speech recognition"""
        try:
            self.partial_callback = on_partial_result
            self.final_callback = on_final_result
            self.error_callback = on_error
            
            # Create audio stream
            self.audio_stream = PushAudioInputStream()
            audio_config = AudioConfig(stream=self.audio_stream)
            
            # Create recognizer
            self.recognizer = SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Set up event handlers
            self.recognizer.recognizing.connect(self._on_recognizing)
            self.recognizer.recognized.connect(self._on_recognized)
            self.recognizer.canceled.connect(self._on_canceled)
            
            # Start continuous recognition
            self.recognizer.start_continuous_recognition()
            self.is_running = True
            
            logger.info(f"Started streaming ASR for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming ASR: {e}")
            if on_error:
                await on_error(f"ASR startup failed: {str(e)}")
            return False
    
    async def stop_streaming(self):
        """Stop streaming speech recognition"""
        try:
            if self.recognizer and self.is_running:
                self.recognizer.stop_continuous_recognition()
                self.is_running = False
                logger.info("Stopped streaming ASR")
            return True
        except Exception as e:
            logger.error(f"Error stopping ASR: {e}")
            return False
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk"""
        try:
            if self.audio_stream and self.is_running:
                # Convert raw PCM to proper format if needed
                processed_audio = self._process_audio_format(audio_data)
                self.audio_stream.write(processed_audio)
            else:
                logger.warning("ASR not running, ignoring audio chunk")
        except Exception as e:
            logger.error(f"Failed to process audio chunk: {e}")
            if self.error_callback:
                await self.error_callback(f"Audio processing failed: {str(e)}")
    
    def _process_audio_format(self, audio_data: bytes) -> bytes:
        """Process audio data to ensure correct format"""
        try:
            # Assume input is 16kHz, 16-bit, mono PCM
            # Convert to numpy array for processing
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Apply basic noise reduction (simple high-pass filter)
            if len(audio_array) > 1:
                # Simple high-pass filter to remove low-frequency noise
                filtered = np.diff(audio_array, prepend=audio_array[0])
                audio_array = filtered.astype(np.int16)
            
            return audio_array.tobytes()
            
        except Exception as e:
            logger.error(f"Audio format processing failed: {e}")
            return audio_data
    
    def _on_recognizing(self, evt):
        """Handle partial recognition results"""
        try:
            if evt.result.reason == ResultReason.RecognizingSpeech:
                result_data = {
                    "text": evt.result.text,
                    "confidence": 0.0,  # Partial results don't have confidence
                    "is_final": False,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                if self.partial_callback:
                    asyncio.create_task(self.partial_callback(result_data))
                    
        except Exception as e:
            logger.error(f"Partial result handling failed: {e}")
    
    def _on_recognized(self, evt):
        """Handle final recognition results"""
        try:
            if evt.result.reason == ResultReason.RecognizedSpeech:
                result_data = {
                    "text": evt.result.text,
                    "confidence": 0.9,  # High confidence for final results
                    "is_final": True,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                if self.final_callback:
                    asyncio.create_task(self.final_callback(result_data))
                    
        except Exception as e:
            logger.error(f"Final result handling failed: {e}")
    
    def _on_canceled(self, evt):
        """Handle recognition cancellation/errors"""
        try:
            if evt.reason == CancellationReason.Error:
                error_msg = f"Recognition canceled: {evt.error_details}"
                logger.error(error_msg)
                
                if self.error_callback:
                    asyncio.create_task(self.error_callback(error_msg))
            else:
                logger.info(f"Recognition canceled: {evt.reason}")
                
        except Exception as e:
            logger.error(f"Cancel handling failed: {e}")
    
    async def get_audio_stats(self) -> Dict[str, Any]:
        """Get audio processing statistics"""
        try:
            return {
                "is_running": self.is_running,
                "audio_stream_active": self.audio_stream is not None,
                "recognizer_active": self.recognizer is not None,
                "callbacks_configured": {
                    "partial": self.partial_callback is not None,
                    "final": self.final_callback is not None,
                    "error": self.error_callback is not None
                }
            }
        except Exception as e:
            logger.error(f"Failed to get audio stats: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.stop_streaming()
            
            if self.audio_stream:
                self.audio_stream.close()
                self.audio_stream = None
            
            if self.recognizer:
                self.recognizer = None
            
            self.partial_callback = None
            self.final_callback = None
            self.error_callback = None
            
            logger.info("StreamingASR cleanup completed")
            
        except Exception as e:
            logger.error(f"ASR cleanup failed: {e}")
