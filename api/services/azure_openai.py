"""
Azure OpenAI Service Integration for VERA Cloud
Handles GPT-4o, Whisper, and Embeddings
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
import openai
from openai import AsyncAzureOpenAI
import json

logger = logging.getLogger(__name__)

class AzureOpenAIService:
    def __init__(self, endpoint: str, api_key: str, api_version: str = "2024-02-15-preview"):
        self.client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.whisper_deployment = "whisper-1"
        self.gpt4o_deployment = "gpt-4o"
        self.embedding_deployment = "text-embedding-3-large"
    
    async def transcribe_audio(self, audio_data: bytes, language: str = "en") -> Tuple[str, float]:
        """Transcribe audio using Azure OpenAI Whisper API"""
        try:
            # Create a file-like object from bytes
            import io
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"
            
            response = await self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.whisper_deployment,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
            
            # Extract text and calculate confidence
            text = response.text
            confidence = getattr(response, 'confidence', 0.9)  # Whisper doesn't provide confidence
            
            logger.info(f"Transcribed {len(audio_data)} bytes: {text[:50]}...")
            return text, confidence
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", 0.0
    
    async def transcribe_streaming(self, audio_chunk: bytes, language: str = "en") -> Tuple[str, float, bool]:
        """Streaming transcription for real-time ASR"""
        try:
            # For streaming, we'll use smaller chunks and return partial results
            text, confidence = await self.transcribe_audio(audio_chunk, language)
            
            # Determine if this is a final result (simple heuristic)
            is_final = len(text.strip()) > 10 and text.strip().endswith(('.', '!', '?'))
            
            return text, confidence, is_final
            
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return "", 0.0, False
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI"""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_deployment,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def generate_rag_response(self, user_input: str, context: List[Dict], 
                                  system_prompt: str = None) -> Dict:
        """Generate RAG-enhanced response using GPT-4o"""
        try:
            # Prepare context for the model
            context_text = "\n".join([f"- {item.get('content', '')}" for item in context])
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt or """You are a medical AI assistant conducting a stroke recovery follow-up call. 
                    Use the provided medical knowledge to ask informed follow-up questions. 
                    Be empathetic, professional, and focused on patient safety. 
                    Keep responses concise and conversational."""
                },
                {
                    "role": "user",
                    "content": f"Patient response: {user_input}\n\nMedical context: {context_text}"
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.gpt4o_deployment,
                messages=messages,
                temperature=0.7,
                max_tokens=200,
                stream=True
            )
            
            # Collect streaming response
            full_response = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            
            return {
                "response": full_response,
                "confidence": 0.9,
                "rag_enhanced": True,
                "context_used": len(context)
            }
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {e}")
            return {
                "response": "I understand. Let me ask you about your recovery progress.",
                "confidence": 0.5,
                "rag_enhanced": False,
                "context_used": 0
            }
    
    async def generate_follow_up_question(self, user_input: str, current_question: str, 
                                        medical_context: List[Dict] = None) -> str:
        """Generate contextual follow-up question"""
        try:
            context_text = ""
            if medical_context:
                context_text = f"\nMedical context: {json.dumps(medical_context, indent=2)}"
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a medical AI assistant. Generate a natural follow-up question based on the patient's response.
                    Current question was: {current_question}
                    Be empathetic and professional. Keep it conversational and focused on stroke recovery.
                    {context_text}"""
                },
                {
                    "role": "user",
                    "content": f"Patient said: {user_input}"
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.gpt4o_deployment,
                messages=messages,
                temperature=0.8,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Follow-up question generation failed: {e}")
            return "Thank you for sharing that. How are you feeling overall with your recovery?"
