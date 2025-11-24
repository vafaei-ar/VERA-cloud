"""
Dialog Integration Service
Integrates Telegram Bot with VERA EnhancedDialog engine
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path to import VERA services
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from api.services.azure_openai import AzureOpenAIService
from api.services.azure_search import AzureSearchService
from api.services.redis_cache import RedisCacheService
from api.services.enhanced_dialog import EnhancedDialog
from bot.services.azure_speech_rest import AzureSpeechServiceREST
import yaml

logger = logging.getLogger(__name__)

class DialogIntegration:
    """Manages integration with VERA dialog engine"""
    
    def __init__(self, config):
        """
        Initialize dialog integration with Azure services
        
        Args:
            config: Configuration object with Azure credentials
        """
        self.config = config
        self.azure_openai: Optional[AzureOpenAIService] = None
        self.azure_speech: Optional[AzureSpeechServiceREST] = None
        self.azure_search: Optional[AzureSearchService] = None
        self.redis_cache: Optional[RedisCacheService] = None
        
        # Load Azure config
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "azure.yaml"
        with open(config_path, "r") as f:
            self.azure_config = yaml.safe_load(f)
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize Azure services"""
        try:
            # Azure OpenAI
            self.azure_openai = AzureOpenAIService(
                endpoint=self.config.AZURE_OPENAI_ENDPOINT,
                api_key=self.config.AZURE_OPENAI_API_KEY,
                api_version=self.azure_config["azure_openai"]["api_version"]
            )
            # Override whisper deployment name from config
            # Try "whisper" first, then fallback to "whisper-1" if not found
            whisper_deployment = self.azure_config["azure_openai"]["deployments"].get("whisper")
            if not whisper_deployment:
                whisper_deployment = "whisper-1"  # Default fallback
            self.azure_openai.whisper_deployment = whisper_deployment
            logger.info(f"Using Whisper deployment: {whisper_deployment}")
            
            # Azure Speech Service (REST API for Telegram bot - no SDK dependencies)
            self.azure_speech = AzureSpeechServiceREST(
                speech_key=self.config.AZURE_SPEECH_KEY,
                region=self.config.AZURE_SPEECH_REGION
            )
            
            # Azure AI Search
            self.azure_search = AzureSearchService(
                endpoint=self.config.AZURE_SEARCH_ENDPOINT,
                api_key=self.config.AZURE_SEARCH_API_KEY,
                index_name=self.azure_config["azure_search"]["index_name"]
            )
            
            # Redis Cache
            self.redis_cache = RedisCacheService(
                connection_string=self.config.REDIS_CONNECTION_STRING,
                ttl_seconds=self.azure_config["redis"]["ttl_seconds"]
            )
            
            logger.info("Azure services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure services: {e}")
            raise
    
    def create_dialog(self, patient_name: str, honorific: str = "", scenario: str = "guided.yml") -> EnhancedDialog:
        """
        Create a new EnhancedDialog instance
        
        Args:
            patient_name: Patient name
            honorific: Honorific (Mr., Mrs., etc.)
            scenario: Scenario file name
        
        Returns:
            EnhancedDialog instance
        """
        scenario_path = Path(__file__).parent.parent.parent.parent / "scenarios" / scenario
        
        if not scenario_path.exists():
            logger.warning(f"Scenario {scenario} not found, using default")
            scenario_path = self.config.SCENARIO_PATH
        
        dialog = EnhancedDialog(
            azure_openai_service=self.azure_openai,
            azure_search_service=self.azure_search,
            redis_cache_service=self.redis_cache,
            scenario_path=str(scenario_path),
            honorific=honorific,
            patient_name=patient_name
        )
        
        logger.info(f"Created dialog for {honorific} {patient_name}")
        return dialog
    
    async def synthesize_text(self, text: str, voice: str = None, rate: float = 1.0) -> bytes:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            voice: Voice name (optional)
            rate: Speaking rate (optional)
        
        Returns:
            Audio data as bytes
        """
        if not self.azure_speech:
            raise RuntimeError("Azure Speech service not initialized")
        
        return await self.azure_speech.synthesize_text(text, voice=voice, rate=rate)
    
    async def transcribe_audio(self, audio_data: bytes, language: str = "en") -> tuple:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio data as bytes
            language: Language code (default: "en")
        
        Returns:
            Tuple of (text, confidence)
        """
        if not self.azure_openai:
            raise RuntimeError("Azure OpenAI service not initialized")
        
        return await self.azure_openai.transcribe_audio(audio_data, language=language)

