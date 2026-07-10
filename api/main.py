"""
VERA Cloud API - Main FastAPI Application
Azure-optimized version of VERA with cloud services
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import yaml
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import Azure services
from .services.azure_openai import AzureOpenAIService
from .services.azure_speech import AzureSpeechService
from .services.azure_search import AzureSearchService
from .services.redis_cache import RedisCacheService
from .services.enhanced_dialog import EnhancedDialog
from .services.outcomes import (
    record_user_urgency, USER_URGENCY_VALUES,
    load_outcome, build_clinician_summary, record_field_action,
    record_reminder, record_session_flags,
)
from .services.resources import lookup_resources, list_regions, NEED_CATEGORIES
from .services.patient_data import load_patient_context
from .services.audit import write_access
from .services.roles import normalize_role
from .services.flagging import evaluate as flag_evaluate
from .services import faq as faq_service

# Load configuration
config_path = Path(__file__).parent.parent / "config" / "azure.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set up logging
logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format=config["logging"]["format"],
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import WebSocket services
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from websocket.services.streaming_asr import StreamingASR
    from websocket.services.streaming_tts import StreamingTTS
    from websocket.handlers.audio_handler import AudioHandler
except ImportError as e:
    logger.warning(f"WebSocket services not available: {e}")
    StreamingASR = None
    StreamingTTS = None
    AudioHandler = None

# Global services
azure_openai: Optional[AzureOpenAIService] = None
azure_speech: Optional[AzureSpeechService] = None
azure_search: Optional[AzureSearchService] = None
redis_cache: Optional[RedisCacheService] = None

# Session management
active_sessions: Dict[str, Dict] = {}
websocket_connections: Dict[str, WebSocket] = {}

# Store conversation data for each session
conversation_data: Dict[str, Dict] = {}  # session_id -> {audio_segments: [], transcript: []}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting VERA Cloud API...")
    
    try:
        # Initialize Azure services
        await initialize_azure_services()
        logger.info("Azure services initialized successfully")
        
        # Warm up caches
        if config["caching"]["cache_warmup"]:
            await warm_up_caches()
            logger.info("Caches warmed up successfully")
        
        logger.info("VERA Cloud API started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start VERA Cloud API: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down VERA Cloud API...")
        await cleanup_services()
        logger.info("VERA Cloud API shutdown complete")

async def initialize_azure_services():
    """Initialize Azure services"""
    global azure_openai, azure_speech, azure_search, redis_cache
    
    try:
        # Azure OpenAI
        azure_openai = AzureOpenAIService(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=config["azure_openai"]["api_version"]
        )
        
        # Azure Speech Service
        azure_speech = AzureSpeechService(
            speech_key=os.getenv("AZURE_SPEECH_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION")
        )
        
        # Azure AI Search
        azure_search = AzureSearchService(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            api_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=config["azure_search"]["index_name"]
        )
        
        # Redis is an optional performance optimization. Keep the application
        # functional without a dedicated cache so small/dev deployments avoid
        # the fixed cost of a managed Redis instance.
        redis_connection_string = os.getenv("REDIS_CONNECTION_STRING", "").strip()
        if redis_connection_string:
            redis_cache = RedisCacheService(
                connection_string=redis_connection_string,
                ttl_seconds=config["redis"]["ttl_seconds"]
            )
        else:
            redis_cache = None
            logger.info("Redis cache disabled (REDIS_CONNECTION_STRING is not set)")
        
    except Exception as e:
        logger.error(f"Failed to initialize Azure services: {e}")
        raise

async def warm_up_caches():
    """Warm up caches with common responses"""
    try:
        if redis_cache:
            # Clear any existing TTS cache to ensure fresh audio
            await redis_cache.clear_cache()
            logger.info("TTS cache cleared")
            
            # Pre-cache common TTS responses
            common_responses = [
                "Hello, I'm your AI stroke navigator.",
                "Thank you for your time today.",
                "I understand. Let me ask you about your recovery.",
                "That's helpful information. Thank you for sharing."
            ]
            
            for response in common_responses:
                await azure_speech.synthesize_text(response)
                # Cache will be populated by the TTS service
            
            logger.info("Cache warmup completed")
    except Exception as e:
        logger.warning(f"Cache warmup failed: {e}")

async def cleanup_services():
    """Clean up services on shutdown"""
    try:
        if redis_cache:
            await redis_cache.close()
        logger.info("Services cleaned up successfully")
    except Exception as e:
        logger.error(f"Service cleanup failed: {e}")

# Create FastAPI app
app = FastAPI(
    title="VERA Cloud API",
    description="Voice-Enabled Recovery Assistant - Cloud Version",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["security"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if config["performance"]["enable_compression"]:
    app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Pydantic models
class StartRequest(BaseModel):
    honorific: str
    patient_name: str
    scenario: str = "guided.yml"
    voice: Optional[str] = None
    rate: float = 1.0
    role: str = "survivor"  # A.6 — survivor | caregiver | clinician
    patient_id: Optional[str] = None      # B.3 — optional PATID
    caregiver_consent: bool = False        # C.1 — consent gate for caregiver role
    empathy: bool = False                  # optional empathetic acknowledgments (DRAFT)

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    version: str

class UrgencyRequest(BaseModel):
    # A.3 — patient self-reported urgency: "routine" | "soon" | "urgent"
    urgency: str
    role: Optional[str] = None

# Routes
@app.get("/")
async def root():
    """Serve the main application"""
    return FileResponse("frontend/static/index.html")

@app.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    try:
        services = {}
        
        # Check Azure OpenAI
        services["azure_openai"] = "available" if azure_openai else "disabled"
        
        # Check Azure Speech
        services["azure_speech"] = "available" if azure_speech else "disabled"
        
        # Check Azure Search
        services["azure_search"] = "available" if azure_search else "disabled"
        
        # Check Redis
        services["redis"] = "available" if redis_cache else "disabled"
        
        # Redis is optional; health reflects only dependencies required to serve
        # a check-in. Its state remains visible in the response for diagnostics.
        required_services = ("azure_openai", "azure_speech", "azure_search")
        overall_status = "healthy" if all(
            services[name] in ["available", "healthy"]
            for name in required_services
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            services=services,
            version="2.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            services={"error": str(e)},
            version="2.0.0"
        )

@app.post("/api/start")
async def start_session(request: StartRequest):
    """Start a new VERA session"""
    try:
        if not azure_openai or not azure_speech or not azure_search:
            raise HTTPException(status_code=503, detail="Azure services not initialized")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Load scenario — resolve to the role-matched file (caregiver/survivor)
        scenario_file = resolve_scenario_file(request.scenario, request.role)
        if not scenario_file:
            raise HTTPException(status_code=404, detail="Scenario not found")
        scenario_path = Path(__file__).parent.parent / "scenarios" / scenario_file
        
        # B.3 / C.1 — load patient context per role + consent gating.
        # Rules: no PATID -> generic; caregiver without consent -> generic (+ note);
        # survivor/caregiver-with-consent/clinician + PATID -> load context.
        patient_context = None
        context_loaded = False
        consent_note = None
        if request.patient_id:
            if request.role == "caregiver" and not request.caregiver_consent:
                consent_note = "caregiver without consent"
            else:
                patient_context = load_patient_context(request.patient_id)
                context_loaded = patient_context is not None
            # B.5 — audit every history-load attempt (access event only, no clinical content).
            write_access(
                patid=request.patient_id, role=request.role,
                consent=request.caregiver_consent, context_loaded=context_loaded,
                session_id=session_id, note=consent_note,
            )

        # Create dialog engine
        dialog = EnhancedDialog(
            azure_openai_service=azure_openai,
            azure_search_service=azure_search,
            redis_cache_service=redis_cache,
            scenario_path=str(scenario_path),
            honorific=request.honorific,
            patient_name=request.patient_name,
            role=request.role,
            patient_context=patient_context,
            empathy=request.empathy,
        )
        
        # Build greeting
        greeting_text = dialog.build_greeting()
        
        # Store session
        active_sessions[session_id] = {
            "dialog": dialog,
            "voice": request.voice,
            "rate": request.rate,
            "role": dialog.role,
            "patient_id": request.patient_id,
            "context_loaded": context_loaded,
            "caregiver_consent": request.caregiver_consent,
            "consent_note": consent_note,
            "start_time": datetime.now().isoformat(),
            "scenario": request.scenario
        }
        
        # Initialize conversation data storage
        conversation_data[session_id] = {
            "audio_segments": [],
            "transcript": [],
            "start_time": datetime.now().isoformat()
        }
        
        logger.info(f"Started session {session_id} for {request.honorific} {request.patient_name}")
        
        return {
            "session_id": session_id,
            "greeting_text": greeting_text,
            "scenario": request.scenario,
            "mode": dialog.mode,
            "role": dialog.role,
            "patient_id": request.patient_id,
            "context_loaded": context_loaded,
            "caregiver_consent": request.caregiver_consent,
            "consent_note": consent_note
        }
        
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

class SessionStartRequest(BaseModel):
    # C.1 — canonical session-initiation contract.
    patient_name: str
    role: str                              # survivor | caregiver | clinician
    patient_id: Optional[str] = None       # PATID (optional)
    caregiver_consent: bool = False        # only relevant when role == caregiver
    honorific: str = ""
    scenario: str = "guided.yml"
    voice: Optional[str] = None
    rate: float = 1.0
    empathy: bool = False                  # optional empathetic acknowledgments (DRAFT)


@app.post("/session/start")
async def session_start(request: SessionStartRequest):
    """C.1 — start a check-in. History-loading rules (also enforced in start_session):

    - no PATID            -> generic mode (no patient context, no Tier-2 context flags;
                              Tier-1 red-flag guidance still works)
    - survivor + PATID    -> load context
    - caregiver + PATID   -> load only if caregiver_consent is true; otherwise generic
                              mode + 'caregiver without consent' audit note
    - clinician + PATID   -> load (treated as authorized); audited

    Returns a session object carrying role, whether context loaded, and consent.
    """
    sr = StartRequest(
        honorific=request.honorific,
        patient_name=request.patient_name,
        scenario=request.scenario,
        voice=request.voice,
        rate=request.rate,
        role=request.role,
        patient_id=request.patient_id,
        caregiver_consent=request.caregiver_consent,
        empathy=request.empathy,
    )
    return await start_session(sr)


@app.websocket("/ws/audio/{session_id}")
async def audio_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time audio processing"""
    logger.info(f"WebSocket connection attempt for session {session_id}")
    logger.info(f"Active sessions: {list(active_sessions.keys())}")
    
    if session_id not in active_sessions:
        logger.error(f"Session {session_id} not found in active sessions")
        await websocket.close(code=1008, reason="Session not found")
        return
    
    await websocket.accept()
    websocket_connections[session_id] = websocket
    logger.info(f"WebSocket connected for session {session_id}")
    
    try:
        session = active_sessions[session_id]
        dialog = session["dialog"]
        voice = session["voice"]
        rate = session["rate"]
        
        # Send initial greeting with TTS
        greeting_text = dialog.build_greeting()
        greeting_audio_duration = 0
        
        # Generate TTS audio for greeting
        if azure_speech:
            try:
                audio_data = await azure_speech.synthesize_text(greeting_text, voice=voice, rate=rate)
                import base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8') if isinstance(audio_data, bytes) else audio_data
                
                # Calculate audio duration for dynamic delay
                # MP3 at 16kHz, 32kbps mono = ~4KB per second
                greeting_audio_duration = len(audio_data) / 4000  # Rough estimate for MP3

                # Store audio segment with sequence number
                if session_id in conversation_data:
                    timestamp = datetime.now().isoformat()
                    sequence = len(conversation_data[session_id]["audio_segments"])
                    conversation_data[session_id]["audio_segments"].append({
                        "type": "bot",
                        "text": greeting_text,
                        "audio_data": audio_base64,
                        "timestamp": timestamp,
                        "sequence": sequence
                    })
                    logger.info(f"Stored greeting audio segment {sequence} for session {session_id} at {timestamp}")
                    conversation_data[session_id]["transcript"].append({
                        "speaker": "bot",
                        "text": greeting_text,
                        "timestamp": datetime.now().isoformat()
                    })

                await websocket.send_text(json.dumps({
                    "type": "audio",
                    "text": greeting_text,
                    "audio_data": audio_base64,
                    "progress": 0
                }))
            except Exception as e:
                logger.error(f"TTS failed: {e}")
                await websocket.send_text(json.dumps({
                    "type": "greeting",
                    "text": greeting_text,
                    "message": "Welcome to VERA Cloud! (TTS temporarily unavailable)",
                    "progress": 0
                }))
        else:
            await websocket.send_text(json.dumps({
                "type": "greeting",
                "text": greeting_text,
                "message": "Welcome to VERA Cloud! Voice features are currently in development mode.",
                "progress": 0
            }))
        
        # Calculate dynamic delay based on audio duration
        if greeting_audio_duration > 0:
            # Use actual audio duration
            dynamic_delay = max(greeting_audio_duration + 0.5, 3)  # Add 1.5s buffer, minimum 3s
            logger.info(f"Audio duration: {greeting_audio_duration:.1f}s, waiting {dynamic_delay:.1f}s total")
        else:
            # Fallback to text-based estimation
            text_duration = len(greeting_text) * 0.08  # ~0.08 seconds per character
            dynamic_delay = max(text_duration + 1, 3)
            logger.info(f"Text-based estimate: {dynamic_delay:.1f}s (text length: {len(greeting_text)} chars)")
        
        await asyncio.sleep(dynamic_delay)
        
        # Send first question after greeting
        first_question = dialog.get_current_question()
        if first_question:
            question_text = first_question["prompt"]
            logger.info(f"Sending first question: {question_text}")
            if azure_speech:
                try:
                    audio_data = await azure_speech.synthesize_text(question_text, voice=voice, rate=rate)
                    import base64
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8') if isinstance(audio_data, bytes) else audio_data
                    
                    # Store audio segment with sequence number
                    if session_id in conversation_data:
                        timestamp = datetime.now().isoformat()
                        sequence = len(conversation_data[session_id]["audio_segments"])
                        conversation_data[session_id]["audio_segments"].append({
                            "type": "bot",
                            "text": question_text,
                            "audio_data": audio_base64,
                            "timestamp": timestamp,
                            "sequence": sequence
                        })
                        logger.info(f"Stored bot audio segment {sequence} for session {session_id}: {question_text[:50]}... at {timestamp}")
                        conversation_data[session_id]["transcript"].append({
                            "speaker": "bot",
                            "text": question_text,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    await websocket.send_text(json.dumps({
                        "type": "audio",
                        "text": question_text,
                        "audio_data": audio_base64,
                        "progress": 0
                    }))
                except Exception as e:
                    logger.error(f"TTS failed for question: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "text": question_text,
                        "status": "question",
                        "progress": 0
                    }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "text": question_text,
                    "status": "question",
                    "progress": 0
                }))
        
        # Message processing loop
        while True:
            try:
                # Check if WebSocket is still open
                if websocket.client_state != websocket.client_state.CONNECTED:
                    logger.info(f"WebSocket disconnected for session {session_id}")
                    break
                
                # Try to receive any message (text or binary)
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                    
                    # Check if it's binary data
                    if message["type"] == "websocket.receive":
                        if "bytes" in message:
                            # Handle binary audio data
                            audio_data = message["bytes"]
                            logger.info(f"Received user audio data: {len(audio_data)} bytes")
                            
                            # Store user audio temporarily until text response is processed
                            websocket._pending_user_audio = audio_data
                            logger.info(f"Stored pending user audio for session {session_id}: {len(audio_data)} bytes")
                            
                            # Acknowledge audio receipt
                            if websocket.client_state == websocket.client_state.CONNECTED:
                                await websocket.send_text(json.dumps({
                                    "type": "response",
                                    "text": "I received your audio message. Processing...",
                                    "status": "audio_received"
                                }))
                            continue
                        
                        elif "text" in message:
                            # Handle text message
                            message_text = message["text"]
                        else:
                            logger.warning(f"Received unknown message type: {message}")
                            continue
                    else:
                        logger.warning(f"Received unexpected message: {message}")
                        continue
                    try:
                        data = json.loads(message_text)
                    except json.JSONDecodeError:
                        # Handle non-JSON text messages
                        logger.info(f"Received plain text: {message_text}")
                        await websocket.send_text(json.dumps({
                            "type": "response",
                            "text": f"Thank you for saying: '{message_text}'. This is a test response.",
                            "status": "received"
                        }))
                        continue
                    
                    if data.get("type") == "text_input":
                        # Handle text input
                        user_text = data.get("text", "")
                        logger.info(f"Received text input: '{user_text}' (length: {len(user_text)})")
                        
                        # Store user input in transcript
                        if session_id in conversation_data:
                            timestamp = datetime.now().isoformat()
                            conversation_data[session_id]["transcript"].append({
                                "speaker": "user",
                                "text": user_text,
                                "timestamp": timestamp
                            })
                            
                            # Check if we have pending user audio to store
                            if hasattr(websocket, '_pending_user_audio'):
                                audio_data = websocket._pending_user_audio
                                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                                sequence = len(conversation_data[session_id]["audio_segments"])
                                
                                conversation_data[session_id]["audio_segments"].append({
                                    "type": "user",
                                    "text": user_text,
                                    "audio_data": audio_base64,
                                    "timestamp": timestamp,
                                    "sequence": sequence
                                })
                                logger.info(f"Stored user audio segment {sequence} for text: '{user_text}' at {timestamp}")
                                delattr(websocket, '_pending_user_audio')
                        
                        # Process with dialog engine
                        response_data = await dialog.process_user_response(user_text)
                        logger.info(f"Dialog response: {response_data}")
                        response = response_data.get("message", "I understand. Let me continue with the next question.")
                        response_type = response_data.get("type", "response")
                        progress = response_data.get("progress", 0)
                        
                        # Check if conversation is complete
                        logger.info(f"Checking completion: dialog.is_complete()={dialog.is_complete()}, response_type='{response_type}'")
                        if dialog.is_complete() or response_type == "completion":
                            logger.info("Conversation completed, triggering completion handler")
                            await handleConversationComplete(session_id, websocket)
                            break
                        
                        # Check if WebSocket is still open before sending response
                        if websocket.client_state != websocket.client_state.CONNECTED:
                            logger.warning(f"WebSocket closed before sending response for session {session_id}")
                            break
                        
                        # Handle different response types
                        if response_type == "denial_with_continuation":
                            # Send denial message first
                            if azure_speech:
                                try:
                                    audio_data = await azure_speech.synthesize_text(response, voice=voice, rate=rate)
                                    import base64
                                    audio_base64 = base64.b64encode(audio_data).decode('utf-8') if isinstance(audio_data, bytes) else audio_data
                                    
                                    # Store audio segment with sequence number
                                    if session_id in conversation_data:
                                        timestamp = datetime.now().isoformat()
                                        sequence = len(conversation_data[session_id]["audio_segments"])
                                        conversation_data[session_id]["audio_segments"].append({
                                            "type": "bot",
                                            "text": response,
                                            "audio_data": audio_base64,
                                            "timestamp": timestamp,
                                            "sequence": sequence
                                        })
                                        logger.info(f"Stored denial response audio segment {sequence} for session {session_id} at {timestamp}")
                                        conversation_data[session_id]["transcript"].append({
                                            "speaker": "bot",
                                            "text": response,
                                            "timestamp": datetime.now().isoformat()
                                        })
                                    
                                    await websocket.send_text(json.dumps({
                                        "type": "audio",
                                        "text": response,
                                        "audio_data": audio_base64,
                                        "progress": progress
                                    }))
                                except Exception as e:
                                    logger.error(f"TTS failed: {e}")
                                    await websocket.send_text(json.dumps({
                                        "type": "response",
                                        "text": response,
                                        "status": "denial",
                                        "progress": progress
                                    }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "response",
                                    "text": response,
                                    "status": "denial",
                                    "progress": progress
                                }))
                            
                            # Wait for denial message to finish playing
                            denial_duration = len(response) * 0.08  # Estimate speech duration
                            await asyncio.sleep(max(denial_duration + 1, 2))
                            next_question = response_data.get("next_question", "")
                            if next_question:
                                if azure_speech:
                                    try:
                                        audio_data = await azure_speech.synthesize_text(next_question, voice=voice, rate=rate)
                                        import base64
                                        audio_base64 = base64.b64encode(audio_data).decode('utf-8') if isinstance(audio_data, bytes) else audio_data
                                        
                                        # Store audio segment with sequence number
                                        if session_id in conversation_data:
                                            timestamp = datetime.now().isoformat()
                                            sequence = len(conversation_data[session_id]["audio_segments"])
                                            conversation_data[session_id]["audio_segments"].append({
                                                "type": "bot",
                                                "text": next_question,
                                                "audio_data": audio_base64,
                                                "timestamp": timestamp,
                                                "sequence": sequence
                                            })
                                            logger.info(f"Stored next question audio segment {sequence} for session {session_id} at {timestamp}")
                                            conversation_data[session_id]["transcript"].append({
                                                "speaker": "bot",
                                                "text": next_question,
                                                "timestamp": datetime.now().isoformat()
                                            })
                                        
                                        await websocket.send_text(json.dumps({
                                            "type": "audio",
                                            "text": next_question,
                                            "audio_data": audio_base64,
                                            "progress": progress
                                        }))
                                    except Exception as e:
                                        logger.error(f"TTS failed for next question: {e}")
                                        await websocket.send_text(json.dumps({
                                            "type": "response",
                                            "text": next_question,
                                            "status": "question",
                                            "progress": progress
                                        }))
                                else:
                                    await websocket.send_text(json.dumps({
                                        "type": "response",
                                        "text": next_question,
                                        "status": "question",
                                        "progress": progress
                                    }))
                        else:
                            # Regular response
                            if azure_speech:
                                try:
                                    audio_data = await azure_speech.synthesize_text(response, voice=voice, rate=rate)
                                    import base64
                                    audio_base64 = base64.b64encode(audio_data).decode('utf-8') if isinstance(audio_data, bytes) else audio_data
                                    
                                    # Store audio segment with sequence number
                                    if session_id in conversation_data:
                                        timestamp = datetime.now().isoformat()
                                        sequence = len(conversation_data[session_id]["audio_segments"])
                                        conversation_data[session_id]["audio_segments"].append({
                                            "type": "bot",
                                            "text": response,
                                            "audio_data": audio_base64,
                                            "timestamp": timestamp,
                                            "sequence": sequence
                                        })
                                        logger.info(f"Stored regular response audio segment {sequence} for session {session_id} at {timestamp}")
                                        conversation_data[session_id]["transcript"].append({
                                            "speaker": "bot",
                                            "text": response,
                                            "timestamp": datetime.now().isoformat()
                                        })
                                    
                                    await websocket.send_text(json.dumps({
                                        "type": "audio",
                                        "text": response,
                                        "audio_data": audio_base64,
                                        "progress": progress
                                    }))
                                except Exception as e:
                                    logger.error(f"TTS failed: {e}")
                                    await websocket.send_text(json.dumps({
                                        "type": "response",
                                        "text": response,
                                        "status": "received",
                                        "progress": progress
                                    }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "response",
                                    "text": response,
                                    "status": "received",
                                    "progress": progress
                                }))
                    
                except asyncio.TimeoutError:
                    # No message received within timeout, continue waiting
                    logger.debug(f"No message received for session {session_id} within timeout")
                    continue
                except Exception as e:
                    logger.error(f"WebSocket error for session {session_id}: {e}")
                    break
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket error for session {session_id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                }))
                break
    
    except Exception as e:
        logger.error(f"WebSocket setup failed for session {session_id}: {e}")
        await websocket.close(code=1011, reason="Internal server error")
    finally:
        # Cleanup WebSocket connections and active sessions
        if session_id in websocket_connections:
            del websocket_connections[session_id]
        if session_id in active_sessions:
            del active_sessions[session_id]
        # Keep conversation_data for download purposes - don't delete it immediately

@app.get("/api/sessions/{session_id}/download")
async def download_session(session_id: str):
    """Download session data"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        dialog = session["dialog"]
        
        # Get session summary
        summary = dialog._generate_session_summary()
        
        # Create download package
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add transcript
            zip_file.writestr("transcript.json", json.dumps(summary, indent=2))
            zip_file.writestr("transcript.txt", f"VERA Session Transcript\n\n{summary.get('patient_name', 'Patient')}\n\n")
            
            # Add session audio if available
            # Note: In a real implementation, you'd store audio in Azure Blob Storage
        
        zip_buffer.seek(0)
        
        return FileResponse(
            io.BytesIO(zip_buffer.getvalue()),
            media_type="application/zip",
            filename=f"vera_session_{session_id}.zip"
        )
        
    except Exception as e:
        logger.error(f"Download failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.post("/api/session/{session_id}/urgency")
async def set_session_urgency(session_id: str, request: UrgencyRequest):
    """A.3 — record the patient's self-reported urgency as its own outcome field.

    This is stored separately from any model/rule flag and never suppresses
    automatic red-flag routing (which is handled independently by the flagging
    layer). Returns the accepted value.
    """
    try:
        role = request.role
        if role is None and session_id in active_sessions:
            role = active_sessions[session_id].get("role")
        record = record_user_urgency(session_id, request.urgency, role=role)
        return {
            "session_id": session_id,
            "user_reported_urgency": record["user_reported_urgency"],
            "note": "User urgency is advisory and does not override automatic red-flag routing.",
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to record urgency for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to record urgency")


class ReminderRequest(BaseModel):
    text: str


@app.post("/api/session/{session_id}/reminder")
async def post_reminder(session_id: str, request: ReminderRequest):
    """A.7 — save a voice-captured reminder to the session."""
    try:
        record = record_reminder(session_id, request.text)
        return {"session_id": session_id, "reminders": record.get("reminders", [])}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Reminder save failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save reminder")


@app.get("/api/session/{session_id}/clinician-summary")
async def get_clinician_summary(session_id: str):
    """A.11 — concise, prioritized clinician summary (flags + urgency first;
    routine items grouped for collapsing; DRAFT role routing attached)."""
    try:
        record = load_outcome(session_id)
        return build_clinician_summary(record)
    except Exception as e:
        logger.error(f"Clinician summary failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to build clinician summary")


class FieldActionRequest(BaseModel):
    field: str


@app.post("/api/session/{session_id}/field-action")
async def post_field_action(session_id: str, request: FieldActionRequest):
    """A.11 — record that a clinician acted on a summary field (workflow-fit metric)."""
    try:
        counts = record_field_action(request.field)
        return {"field": request.field, "count": counts.get(request.field, 0)}
    except Exception as e:
        logger.error(f"Field-action record failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to record field action")


@app.get("/api/resources")
async def get_resources(region: Optional[str] = None, need: Optional[str] = None):
    """A.8 — opt-in local resource lookup by region (county) + optional need.

    Separate from the core check-in. Returns information-only resources with a
    clear non-endorsement disclaimer.
    """
    try:
        return lookup_resources(region, need)
    except Exception as e:
        logger.error(f"Resource lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Resource lookup failed")


@app.get("/api/resource-regions")
async def get_resource_regions():
    """A.8 — list regions available in the resource directory."""
    return {"regions": list_regions(), "needs": list(NEED_CATEGORIES)}


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
async def ask_vera(request: AskRequest):
    """Ask-VERA — RETRIEVAL ONLY. Gated by env ASK_ENABLED (default off).
    DRAFT, pending clinical sign-off (see Kura docs/ASK_VERA_SPEC.md).

    Pipeline: (1) flag the question first — a red/urgent flag returns safety
    guidance, never an answer; (2) return a clinician-approved FAQ answer
    verbatim if one matches; (3) otherwise refuse. No generated medical text.
    """
    if os.getenv("ASK_ENABLED", "false").strip().lower() not in ("1", "true", "yes", "on"):
        raise HTTPException(status_code=403, detail="Ask-VERA is disabled")

    q = (request.question or "").strip()
    if not q:
        raise HTTPException(status_code=422, detail="Empty question")

    # 1) Safety: never chat past a red flag described in the question.
    result = flag_evaluate(q)
    tier = result.get("overall_tier")
    if tier == 1:
        return {"kind": "emergency", "answer": result.get("guidance"), "tier": 1}
    if tier == 2:
        return {"kind": "urgent", "answer": result.get("guidance"), "tier": 2}

    # 2) Clinician-approved answer, verbatim.
    hit = faq_service.lookup(q)
    if hit:
        return {
            "kind": "answer",
            "answer": hit.get("answer"),
            "disclaimer": faq_service.disclaimer(),
            "topic": hit.get("id"),
        }

    # 3) Refuse anything not approved.
    return {"kind": "refusal", "answer": faq_service.REFUSAL}


# Friendly, role-neutral labels for the patient-facing scenario picker.
SCENARIO_LABELS = {
    "guided": "General check-in (guided)",
    "rag_enhanced": "General check-in (AI-enhanced)",
    "micro_routine": "Routine check-in",
    "micro_worsening": "Worsening check-in",
    "micro_redflag": "Red-flag check-in (demo)",
}


def _scenario_base(stem: str) -> str:
    """Strip a trailing _survivor/_caregiver role suffix to get the base name."""
    for r in ("survivor", "caregiver"):
        if stem.endswith("_" + r):
            return stem[: -(len(r) + 1)]
    return stem


def resolve_scenario_file(scenario_arg: str, role: str) -> Optional[str]:
    """Resolve a scenario request (base name or filename) + role to an actual file.

    Role is authoritative: caregiver -> <base>_caregiver.yml when it exists;
    otherwise falls back to <base>_survivor.yml, then <base>.yml.
    """
    sdir = Path(__file__).parent.parent / "scenarios"
    role_n = normalize_role(role)
    stem = scenario_arg[:-4] if scenario_arg.endswith(".yml") else scenario_arg
    base = _scenario_base(stem)
    candidates = []
    if role_n != "survivor":
        candidates.append(f"{base}_{role_n}.yml")
    candidates += [f"{base}_survivor.yml", f"{base}.yml"]
    for c in candidates:
        if (sdir / c).exists():
            return c
    orig = scenario_arg if scenario_arg.endswith(".yml") else scenario_arg + ".yml"
    return orig if (sdir / orig).exists() else None


@app.get("/api/scenarios")
async def get_scenarios():
    """Get available scenarios as role-neutral BASE check-ins.

    Survivor/caregiver variants are collapsed into one base entry; the role chosen
    at session start selects the wording.
    """
    try:
        scenarios_dir = Path(__file__).parent.parent / "scenarios"
        bases = {}
        for scenario_file in scenarios_dir.glob("*.yml"):
            stem = scenario_file.stem
            base = _scenario_base(stem)
            try:
                with open(scenario_file, "r") as f:
                    meta = (yaml.safe_load(f) or {}).get("meta", {})
            except Exception:
                continue
            entry = bases.setdefault(base, {
                "base": base, "roles": [],
                "description": meta.get("description", ""),
                "mode": meta.get("mode", "guided"),
            })
            role = "caregiver" if stem.endswith("_caregiver") else "survivor"
            if role not in entry["roles"]:
                entry["roles"].append(role)

        order = list(SCENARIO_LABELS.keys())
        scenarios = [
            {"base": b, "name": SCENARIO_LABELS.get(b, b),
             "description": e["description"], "mode": e["mode"], "roles": sorted(e["roles"])}
            for b, e in bases.items()
        ]
        scenarios.sort(key=lambda x: order.index(x["base"]) if x["base"] in order else 99)
        return {"scenarios": scenarios}
        
    except Exception as e:
        logger.error(f"Failed to get scenarios: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scenarios: {str(e)}")

@app.get("/api/voices")
async def get_voices():
    """Get available TTS voices"""
    try:
        if azure_speech:
            voices = azure_speech.get_available_voices()
            return {"voices": voices}
        else:
            return {"voices": {}}
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")

@app.post("/api/test-tts")
async def test_tts(request: dict):
    """Test TTS functionality"""
    text = request.get("text", "Hello, this is a test of the text-to-speech system.")

    if not azure_speech:
        return {"error": "Azure Speech service not available"}

    try:
        audio_data = await azure_speech.synthesize_text(text)
        import base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8') if isinstance(audio_data, bytes) else audio_data

        return {
            "text": text,
            "audio_data": audio_base64,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"TTS test failed: {e}")
        return {"error": f"TTS failed: {str(e)}"}

def build_response_time_message(cfg: dict, priority: bool = False) -> str:
    """Build the configurable response-time expectation message (A.2).

    Driven by config['response_expectations'] so each deployment can set its own
    timeframe and urgent instructions without code changes.

    When `priority` is True (a Tier-1/Tier-2 flag fired, or the patient marked the
    check-in urgent), the routine "not watched in real time / reply within N
    business days" wording is SUPPRESSED — telling a patient who just reported
    red-flag stroke symptoms to expect a reply in 2 business days is unsafe and
    contradictory. Instead we surface the urgent/911 instruction only.

    # ===== DRAFT CLINICAL LOGIC — REQUIRES DR. ZAND SIGN-OFF =====
    # Rationale: focus group (Laura) feared an unmonitored inbox; set expectations
    #   plainly and always route urgent issues to a human / 911. The priority
    #   branch ensures the routine timeframe is never read back on a flagged call.
    # Source: focus group June 2026.
    # DO NOT treat the timeframe or urgent wording as clinically validated.
    """
    re_cfg = (cfg or {}).get("response_expectations", {}) or {}
    urgent = re_cfg.get(
        "urgent_instructions",
        "For anything urgent, call your care team. If it is an emergency, call 911.",
    )
    if priority:
        # No routine timeframe and no "not watched in real time" line on a flagged
        # call — only the escalation instruction.
        return re_cfg.get(
            "priority_instructions",
            f"Because of what you told me, a member of your care team will be "
            f"notified to follow up. {urgent}",
        )
    days = re_cfg.get("routine_response_business_days", 2)
    not_realtime = "" if re_cfg.get("monitored_real_time", False) else \
        "This check-in is not watched in real time. "
    plural = "day" if days == 1 else "days"
    return f"{not_realtime}For routine questions, expect a reply within {days} business {plural}. {urgent}"
    # ===== END DRAFT CLINICAL LOGIC =====


async def handleConversationComplete(session_id: str, websocket: WebSocket):
    """Handle conversation completion"""
    try:
        logger.info(f"Handling conversation completion for session {session_id}")

        # Get the wrapup message from the scenario
        is_priority = False
        if session_id in active_sessions:
            dialog = active_sessions[session_id]["dialog"]
            # C.3 — persist accumulated flags so the clinician summary (A.11) reflects them.
            try:
                record_session_flags(session_id, getattr(dialog, "flags", []))
            except Exception as _e:
                logger.error(f"Failed to persist session flags: {_e}")
            wrapup_message = dialog.scenario.get("wrapup", {}).get("message",
                "Thank you for your time. A member of our care team will review your responses.")
            # A.2/A.3 — a flagged or self-reported-urgent call must NOT hear the
            # routine "2 business days / not watched in real time" wording.
            try:
                tier = dialog.overall_tier() if hasattr(dialog, "overall_tier") else 3
                is_priority = tier in (1, 2)
            except Exception:
                is_priority = False
            try:
                outcome = load_outcome(session_id)
                if (outcome or {}).get("user_reported_urgency") == "urgent":
                    is_priority = True
            except Exception:
                pass
        else:
            wrapup_message = "Thank you for your time. A member of our care team will review your responses."

        # A.2: append the configurable response-time expectation to the spoken wrapup,
        # and send it as a separate field so the UI can show it on-screen too.
        response_time_message = build_response_time_message(config, priority=is_priority)
        wrapup_message = f"{wrapup_message} {response_time_message}"

        logger.info(f"Wrapup message: {wrapup_message}")
        
        if azure_speech and session_id in active_sessions:
            session = active_sessions[session_id]
            voice = session.get("voice", "en-US-AriaNeural")
            rate = session.get("rate", 1.0)
            
            try:
                audio_data = await azure_speech.synthesize_text(wrapup_message, voice=voice, rate=rate)
                import base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8') if isinstance(audio_data, bytes) else audio_data
                
                # Store final audio segment with sequence number
                if session_id in conversation_data:
                    timestamp = datetime.now().isoformat()
                    sequence = len(conversation_data[session_id]["audio_segments"])
                    conversation_data[session_id]["audio_segments"].append({
                        "type": "bot",
                        "text": wrapup_message,
                        "audio_data": audio_base64,
                        "timestamp": timestamp,
                        "sequence": sequence
                    })
                    logger.info(f"Stored completion audio segment {sequence} for session {session_id} at {timestamp}")
                    conversation_data[session_id]["transcript"].append({
                        "speaker": "bot",
                        "text": wrapup_message,
                        "timestamp": datetime.now().isoformat()
                    })
                
                await websocket.send_text(json.dumps({
                    "type": "completion",
                    "response_time_message": response_time_message,
                    "text": wrapup_message,
                    "audio_data": audio_base64,
                    "progress": 100.0
                }))
            except Exception as e:
                logger.error(f"TTS failed for completion: {e}")
                await websocket.send_text(json.dumps({
                    "type": "completion",
                    "response_time_message": response_time_message,
                    "text": wrapup_message,
                    "progress": 100.0
                }))
        else:
            await websocket.send_text(json.dumps({
                "type": "completion",
                "text": wrapup_message,
                "progress": 100.0
            }))
            
    except Exception as e:
        logger.error(f"Conversation completion handling failed: {e}")

@app.get("/api/debug-session/{session_id}")
async def debug_session(session_id: str):
    """Debug session state"""
    logger.info(f"Debug request for session: {session_id}")
    logger.info(f"Active sessions: {list(active_sessions.keys())}")
    
    if session_id not in active_sessions:
        return {"error": "Session not found", "active_sessions": list(active_sessions.keys())}
    
    session = active_sessions[session_id]
    dialog = session["dialog"]
    
    return {
        "current_index": dialog.current_index,
        "scenario_flow_length": len(dialog.scenario.get("flow", [])),
        "current_question": dialog.get_current_question(),
        "is_complete": dialog.is_complete()
    }

@app.get("/api/conversation-data/{session_id}")
async def get_conversation_data(session_id: str):
    """Get recorded conversation data for a session"""
    if session_id not in conversation_data:
        return {"error": "Conversation data not found for this session"}
    
    data = conversation_data[session_id]
    
    # Calculate total duration
    if data["transcript"]:
        start_time = datetime.fromisoformat(data["start_time"])
        end_time = datetime.fromisoformat(data["transcript"][-1]["timestamp"])
        duration = (end_time - start_time).total_seconds()
    else:
        duration = 0
    
    return {
        "session_id": session_id,
        "duration_seconds": duration,
        "audio_segments": data["audio_segments"],
        "transcript": data["transcript"],
        "summary": {
            "total_messages": len(data["transcript"]),
            "bot_messages": len([t for t in data["transcript"] if t["speaker"] == "bot"]),
            "user_messages": len([t for t in data["transcript"] if t["speaker"] == "user"]),
            "audio_segments": len(data["audio_segments"])
        }
    }

@app.get("/api/conversation-audio/{session_id}")
async def get_conversation_audio(session_id: str):
    """Get concatenated audio for a session (bot audio only due to format compatibility)"""
    if session_id not in conversation_data:
        return {"error": "Conversation data not found for this session"}
    
    data = conversation_data[session_id]
    audio_segments = data.get("audio_segments", [])
    
    # Filter audio segments (both bot and user)
    valid_audio_segments = [seg for seg in audio_segments if seg.get("audio_data")]
    
    if not valid_audio_segments:
        return {"error": "No audio segments found for this session"}
    
    # Sort by timestamp to maintain chronological order
    valid_audio_segments.sort(key=lambda x: x.get("timestamp", ""))
    
    try:
        # Properly merge all audio segments using pydub
        import base64
        import io
        from pydub import AudioSegment
        
        audio_segments_list = []
        bot_segments = 0
        user_segments = 0
        
        for i, segment in enumerate(valid_audio_segments):
            try:
                audio_data = base64.b64decode(segment["audio_data"])
                segment_type = segment.get('type', 'unknown')
                
                logger.info(f"Processing segment {i}: type={segment_type}, size={len(audio_data)} bytes")
                
                if segment_type == 'bot':
                    # Bot audio is already in MP3 format
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                    audio_segments_list.append(audio_segment)
                    bot_segments += 1
                    logger.info(f"Loaded bot segment {i}: {len(audio_data)} bytes, duration: {len(audio_segment)}ms")
                else:
                    # User audio is in Opus format, convert to MP3
                    try:
                        logger.info(f"Converting user segment {i} from Opus to MP3...")
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp4")
                        logger.info(f"Loaded user audio: {len(audio_data)} bytes, duration: {len(audio_segment)}ms")
                        
                        # Normalize audio level to ensure it's audible
                        audio_segment = audio_segment.normalize()
                        
                        audio_segments_list.append(audio_segment)
                        user_segments += 1
                        logger.info(f"Added user segment {i}: duration: {len(audio_segment)}ms")
                        
                    except Exception as e:
                        logger.error(f"Failed to convert user segment {i} to MP3: {e}")
                        logger.error(f"Audio data length: {len(audio_data)} bytes")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to process segment {i}: {e}")
                continue
        
        # Merge all audio segments properly
        if audio_segments_list:
            logger.info(f"Merging {len(audio_segments_list)} audio segments...")
            logger.info(f"Segment breakdown: {len([s for s in valid_audio_segments if s.get('type') == 'bot'])} bot, {len([s for s in valid_audio_segments if s.get('type') == 'user'])} user")
            
            concatenated_audio_segment = audio_segments_list[0]
            logger.info(f"Starting with segment 0: {len(concatenated_audio_segment)}ms")
            
            for i, segment in enumerate(audio_segments_list[1:], 1):
                concatenated_audio_segment += segment
                logger.info(f"Added segment {i}, total duration: {len(concatenated_audio_segment)}ms")
            
            # Export as MP3
            mp3_buffer = io.BytesIO()
            concatenated_audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")
            concatenated_audio = mp3_buffer.getvalue()
            
            logger.info(f"Audio merge complete: {bot_segments} bot segments, {user_segments} user segments")
            logger.info(f"Final audio duration: {len(concatenated_audio_segment)}ms, size: {len(concatenated_audio)} bytes")
        else:
            logger.error("No valid audio segments found")
            return {"error": "No valid audio segments found"}
        
        # Return the concatenated audio as a file response
        from fastapi.responses import Response
        return Response(
            content=concatenated_audio,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=conversation-audio-complete-{session_id}.mp3"
            }
        )
        
    except Exception as e:
        logger.error(f"Error concatenating audio for session {session_id}: {e}")
        return {"error": f"Failed to concatenate audio: {str(e)}"}

@app.get("/api/conversation-audio-with-user/{session_id}")
async def get_conversation_audio_with_user(session_id: str):
    """Get conversation audio including user segments (separate files)"""
    if session_id not in conversation_data:
        return {"error": "Conversation data not found for this session"}
    
    data = conversation_data[session_id]
    audio_segments = data.get("audio_segments", [])
    
    # Filter audio segments (both bot and user)
    valid_audio_segments = [seg for seg in audio_segments if seg.get("audio_data")]
    
    if not valid_audio_segments:
        return {"error": "No audio segments found for this session"}
    
    # Sort by timestamp to maintain chronological order
    valid_audio_segments.sort(key=lambda x: x.get("timestamp", ""))
    
    try:
        import base64
        import zipfile
        import io
        from pydub import AudioSegment
        
        # Create a ZIP file containing all audio segments
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
            for i, segment in enumerate(valid_audio_segments):
                try:
                    audio_data = base64.b64decode(segment["audio_data"])
                    segment_type = segment.get('type', 'unknown')
                    timestamp = segment.get('timestamp', 'unknown')
                    
                    if segment_type == 'bot':
                        # Bot audio is already in MP3 format
                        filename = f"segment_{i:03d}_bot_{timestamp}.mp3"
                        zip_file.writestr(filename, audio_data)
                        logger.info(f"Added bot segment {i} as MP3: {len(audio_data)} bytes")
                    else:
                        # User audio is in Opus format, convert to MP3
                        try:
                            logger.info(f"Attempting to convert user segment {i} from Opus to MP3...")
                            
                            # Create AudioSegment from MP4 data (which contains Opus codec)
                            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp4")
                            logger.info(f"Successfully loaded Opus audio: {len(audio_data)} bytes, duration: {len(audio_segment)}ms")
                            
                            # Export as MP3
                            mp3_buffer = io.BytesIO()
                            audio_segment.export(mp3_buffer, format="mp3", bitrate="128k")
                            mp3_data = mp3_buffer.getvalue()
                            
                            filename = f"segment_{i:03d}_user_{timestamp}.mp3"
                            zip_file.writestr(filename, mp3_data)
                            logger.info(f"Successfully converted user segment {i} from Opus to MP3: {len(audio_data)} -> {len(mp3_data)} bytes")
                            
                        except Exception as e:
                            # Fallback: save as raw Opus
                            logger.error(f"Failed to convert user segment {i} to MP3: {e}")
                            logger.error(f"Audio data length: {len(audio_data)} bytes")
                            filename = f"segment_{i:03d}_user_{timestamp}.opus"
                            zip_file.writestr(filename, audio_data)
                            logger.info(f"Saved user segment {i} as raw Opus due to conversion failure")
                            
                except Exception as e:
                    logger.error(f"Failed to process segment {i}: {e}")
                    continue
        
        zip_buffer.seek(0)
        
        # Return the ZIP file
        from fastapi.responses import Response
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=conversation-audio-segments-{session_id}.zip"
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating audio segments for session {session_id}: {e}")
        return {"error": f"Failed to create audio segments: {str(e)}"}

@app.post("/api/test-dialog/{session_id}")
async def test_dialog(session_id: str, request: dict):
    """Test dialog processing for a session"""
    if session_id not in active_sessions:
        return {"error": "Session not found"}
    
    session = active_sessions[session_id]
    dialog = session["dialog"]
    user_text = request.get("text", "")
    
    try:
        response_data = await dialog.process_user_response(user_text)
        return {
            "user_input": user_text,
            "response": response_data,
            "current_index": dialog.current_index,
            "is_complete": dialog.is_complete()
        }
    except Exception as e:
        logger.error(f"Dialog test failed: {e}")
        return {"error": f"Dialog processing failed: {str(e)}"}

@app.get("/api/debug-audio/{session_id}")
async def debug_audio(session_id: str):
    """Debug audio segments for a session"""
    if session_id not in conversation_data:
        return {"error": "Conversation data not found for this session"}
    
    data = conversation_data[session_id]
    audio_segments = data.get("audio_segments", [])
    
    # Analyze segments
    bot_segments = [s for s in audio_segments if s.get("type") == "bot"]
    user_segments = [s for s in audio_segments if s.get("type") == "user"]
    
    # Calculate total sizes
    import base64
    bot_total_size = sum(len(base64.b64decode(s.get("audio_data", ""))) for s in bot_segments)
    user_total_size = sum(len(base64.b64decode(s.get("audio_data", ""))) for s in user_segments)
    
    return {
        "session_id": session_id,
        "total_segments": len(audio_segments),
        "bot_segments": len(bot_segments),
        "user_segments": len(user_segments),
        "bot_total_size_bytes": bot_total_size,
        "user_total_size_bytes": user_total_size,
            "segments": [
                {
                    "index": i,
                    "sequence": s.get("sequence", i),
                    "type": s.get("type"),
                    "timestamp": s.get("timestamp"),
                    "size_bytes": len(base64.b64decode(s.get("audio_data", ""))),
                    "text_preview": s.get("text", "")[:50] + "..." if len(s.get("text", "")) > 50 else s.get("text", "")
                }
                for i, s in enumerate(audio_segments)
            ]
    }

@app.delete("/api/cleanup-conversation-data")
async def cleanup_conversation_data():
    """Clean up old conversation data (call this periodically)"""
    try:
        # Keep only recent sessions (last 24 hours)
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        sessions_to_remove = []
        for session_id, data in conversation_data.items():
            if data.get("start_time"):
                start_time = datetime.fromisoformat(data["start_time"])
                if start_time < cutoff_time:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del conversation_data[session_id]
        
        return {
            "message": f"Cleaned up {len(sessions_to_remove)} old sessions",
            "remaining_sessions": len(conversation_data)
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"error": f"Cleanup failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config["app"]["host"],
        port=config["app"]["port"],
        log_level=config["app"]["log_level"].lower(),
        reload=config["app"]["debug"]
    )
