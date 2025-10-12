"""
Audio Handler for VERA Cloud WebSocket
Manages real-time audio processing and streaming
"""

import asyncio
import logging
from typing import Dict, Optional, Any
import json
import io
import wave
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioHandler:
    def __init__(self, session_id: str, streaming_asr, streaming_tts, dialog_engine, cache_service):
        self.session_id = session_id
        self.asr = streaming_asr
        self.tts = streaming_tts
        self.dialog = dialog_engine
        self.cache = cache_service
        
        # Audio processing state
        self.audio_buffer = []
        self.is_processing = False
        self.last_activity = datetime.now()
        self.session_audio = io.BytesIO()
        
        # Statistics
        self.stats = {
            "audio_chunks_received": 0,
            "transcriptions_completed": 0,
            "tts_synthesized": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"AudioHandler initialized for session {session_id}")
    
    async def handle_audio_chunk(self, audio_data: bytes, websocket) -> bool:
        """Handle incoming audio chunk from WebSocket"""
        try:
            self.stats["audio_chunks_received"] += 1
            self.last_activity = datetime.now()
            
            # Store in session audio
            self.session_audio.write(audio_data)
            
            # Process with ASR
            await self.asr.process_audio_chunk(audio_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle audio chunk: {e}")
            await self._send_error(websocket, f"Audio processing failed: {str(e)}")
            return False
    
    async def handle_transcription_result(self, result: Dict, websocket) -> bool:
        """Handle ASR transcription result"""
        try:
            self.stats["transcriptions_completed"] += 1
            
            # Send transcript update to client
            await self._send_transcript_update(websocket, result)
            
            # Process with dialog engine if final result
            if result.get("is_final", False):
                await self._process_final_transcription(result, websocket)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle transcription result: {e}")
            await self._send_error(websocket, f"Transcription processing failed: {str(e)}")
            return False
    
    async def handle_text_message(self, text: str, websocket) -> bool:
        """Handle text message (fallback when audio fails)"""
        try:
            # Process as if it were a transcription result
            result = {
                "text": text,
                "confidence": 0.8,
                "is_final": True,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            return await self.handle_transcription_result(result, websocket)
            
        except Exception as e:
            logger.error(f"Failed to handle text message: {e}")
            await self._send_error(websocket, f"Text processing failed: {str(e)}")
            return False
    
    async def synthesize_and_send(self, text: str, voice: str = None, rate: float = 1.0, 
                                 websocket = None) -> bool:
        """Synthesize text and send audio to client"""
        try:
            if not text.strip():
                return True
            
            # Check cache first
            cached_audio = await self.cache.get_tts_audio(text, voice or "default", rate)
            if cached_audio:
                self.stats["cache_hits"] += 1
                await self._send_audio_chunk(websocket, cached_audio)
                return True
            
            self.stats["cache_misses"] += 1
            
            # Synthesize with TTS
            audio_data = await self.tts.synthesize_text(text, voice, rate)
            if not audio_data:
                logger.error("TTS synthesis returned empty audio")
                return False
            
            # Cache the result
            await self.cache.set_tts_audio(text, voice or "default", rate, audio_data)
            
            # Send audio to client
            await self._send_audio_chunk(websocket, audio_data)
            self.stats["tts_synthesized"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to synthesize and send: {e}")
            await self._send_error(websocket, f"TTS synthesis failed: {str(e)}")
            return False
    
    async def synthesize_streaming(self, text: str, voice: str = None, rate: float = 1.0, 
                                 websocket = None) -> bool:
        """Stream TTS audio as it's generated"""
        try:
            if not text.strip():
                return True
            
            # Check cache first
            cached_audio = await self.cache.get_tts_audio(text, voice or "default", rate)
            if cached_audio:
                self.stats["cache_hits"] += 1
                await self._send_audio_chunk(websocket, cached_audio)
                return True
            
            self.stats["cache_misses"] += 1
            self.stats["tts_synthesized"] += 1
            
            # Stream synthesis
            async for audio_chunk in self.tts.synthesize_streaming(text, voice, rate):
                await self._send_audio_chunk(websocket, audio_chunk)
            
            # Cache the complete audio
            # Note: For streaming, we'd need to collect all chunks to cache
            # This is a simplified version
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stream synthesis: {e}")
            await self._send_error(websocket, f"Streaming TTS failed: {str(e)}")
            return False
    
    async def _process_final_transcription(self, result: Dict, websocket) -> bool:
        """Process final transcription with dialog engine"""
        try:
            user_input = result["text"]
            confidence = result["confidence"]
            
            # Process with dialog engine
            dialog_response = await self.dialog.process_user_response(user_input, confidence)
            
            # Handle different response types
            response_type = dialog_response.get("type", "unknown")
            
            if response_type == "question":
                # Send next question
                await self.synthesize_and_send(
                    dialog_response["message"], 
                    websocket=websocket
                )
                
            elif response_type == "rag_enhanced":
                # Send RAG-enhanced response
                await self.synthesize_and_send(
                    dialog_response["message"], 
                    websocket=websocket
                )
                
                # Send additional context if available
                if "rag_response" in dialog_response:
                    rag_info = dialog_response["rag_response"]
                    if rag_info.get("rag_enhanced", False):
                        await self._send_rag_context(websocket, rag_info)
            
            elif response_type == "emergency":
                # Handle emergency response
                await self.synthesize_and_send(
                    dialog_response["message"], 
                    websocket=websocket
                )
                await self._send_emergency_alert(websocket, dialog_response)
            
            elif response_type == "completion":
                # Send completion message
                await self.synthesize_and_send(
                    dialog_response["message"], 
                    websocket=websocket
                )
                await self._send_session_summary(websocket, dialog_response)
            
            elif response_type == "error":
                # Send error message
                await self.synthesize_and_send(
                    dialog_response["message"], 
                    websocket=websocket
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process final transcription: {e}")
            await self._send_error(websocket, f"Dialog processing failed: {str(e)}")
            return False
    
    async def _send_audio_chunk(self, websocket, audio_data: bytes):
        """Send audio chunk to client"""
        try:
            if websocket and not websocket.closed:
                await websocket.send_bytes(audio_data)
        except Exception as e:
            logger.error(f"Failed to send audio chunk: {e}")
    
    async def _send_transcript_update(self, websocket, result: Dict):
        """Send transcript update to client"""
        try:
            if websocket and not websocket.closed:
                message = {
                    "type": "transcript_update",
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "is_final": result["is_final"],
                    "timestamp": result.get("timestamp", asyncio.get_event_loop().time())
                }
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send transcript update: {e}")
    
    async def _send_rag_context(self, websocket, rag_info: Dict):
        """Send RAG context information to client"""
        try:
            if websocket and not websocket.closed:
                message = {
                    "type": "rag_context",
                    "rag_enhanced": rag_info.get("rag_enhanced", False),
                    "context_used": rag_info.get("context_used", 0),
                    "confidence": rag_info.get("confidence", 0.0)
                }
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send RAG context: {e}")
    
    async def _send_emergency_alert(self, websocket, dialog_response: Dict):
        """Send emergency alert to client"""
        try:
            if websocket and not websocket.closed:
                message = {
                    "type": "emergency_alert",
                    "emergency_detected": True,
                    "message": "Emergency detected. Please call 911 if needed.",
                    "medical_context": dialog_response.get("medical_context", {})
                }
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send emergency alert: {e}")
    
    async def _send_session_summary(self, websocket, dialog_response: Dict):
        """Send session summary to client"""
        try:
            if websocket and not websocket.closed:
                message = {
                    "type": "session_summary",
                    "summary": dialog_response.get("session_summary", {}),
                    "responses": dialog_response.get("responses", {}),
                    "next_action": dialog_response.get("next_action", "end_session")
                }
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send session summary: {e}")
    
    async def _send_error(self, websocket, error_message: str):
        """Send error message to client"""
        try:
            if websocket and not websocket.closed:
                message = {
                    "type": "error",
                    "message": error_message,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
    
    async def get_session_audio(self) -> bytes:
        """Get complete session audio"""
        try:
            self.session_audio.seek(0)
            return self.session_audio.getvalue()
        except Exception as e:
            logger.error(f"Failed to get session audio: {e}")
            return b""
    
    async def get_audio_stats(self) -> Dict[str, Any]:
        """Get audio processing statistics"""
        try:
            return {
                "session_id": self.session_id,
                "stats": self.stats,
                "is_processing": self.is_processing,
                "last_activity": self.last_activity.isoformat(),
                "audio_buffer_size": len(self.audio_buffer),
                "session_audio_size": len(self.session_audio.getvalue())
            }
        except Exception as e:
            logger.error(f"Failed to get audio stats: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up audio handler resources"""
        try:
            self.is_processing = False
            self.audio_buffer.clear()
            self.session_audio.close()
            logger.info(f"AudioHandler cleanup completed for session {self.session_id}")
        except Exception as e:
            logger.error(f"AudioHandler cleanup failed: {e}")
