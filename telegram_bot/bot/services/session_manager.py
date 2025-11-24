"""
Session Manager for Telegram Bot
Maps Telegram users to VERA sessions
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages Telegram user sessions and their VERA dialog instances"""
    
    def __init__(self, timeout_hours: int = 24):
        """
        Initialize session manager
        
        Args:
            timeout_hours: Hours before session expires
        """
        self.sessions: Dict[int, Dict] = {}  # telegram_user_id -> session_data
        self.timeout_hours = timeout_hours
        logger.info(f"SessionManager initialized with {timeout_hours}h timeout")
    
    def create_session(self, telegram_user_id: int, patient_name: str = "", honorific: str = "", scenario: str = "guided.yml") -> str:
        """
        Create a new session for a Telegram user
        
        Args:
            telegram_user_id: Telegram user ID
            patient_name: Patient name for VERA session
            honorific: Honorific (Mr., Mrs., etc.)
            scenario: Scenario file name (e.g., "guided.yml" or "rag_enhanced.yml")
        
        Returns:
            Session ID (UUID string)
        """
        session_id = str(uuid.uuid4())
        
        self.sessions[telegram_user_id] = {
            "session_id": session_id,
            "patient_name": patient_name or f"User_{telegram_user_id}",
            "honorific": honorific or "",
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "dialog_engine": None,  # Will be set when dialog is initialized
            "scenario": scenario
        }
        
        logger.info(f"Created session {session_id} for Telegram user {telegram_user_id} with scenario {scenario}")
        return session_id
    
    def get_session(self, telegram_user_id: int) -> Optional[Dict]:
        """
        Get session for a Telegram user
        
        Args:
            telegram_user_id: Telegram user ID
        
        Returns:
            Session data dict or None if not found/expired
        """
        if telegram_user_id not in self.sessions:
            return None
        
        session = self.sessions[telegram_user_id]
        
        # Check if session expired
        if self._is_expired(session):
            logger.info(f"Session expired for user {telegram_user_id}")
            del self.sessions[telegram_user_id]
            return None
        
        # Update last activity
        session["last_activity"] = datetime.now()
        return session
    
    def update_session(self, telegram_user_id: int, **kwargs):
        """
        Update session data
        
        Args:
            telegram_user_id: Telegram user ID
            **kwargs: Key-value pairs to update
        """
        if telegram_user_id in self.sessions:
            self.sessions[telegram_user_id].update(kwargs)
            self.sessions[telegram_user_id]["last_activity"] = datetime.now()
            logger.debug(f"Updated session for user {telegram_user_id}")
    
    def delete_session(self, telegram_user_id: int):
        """
        Delete session for a Telegram user
        
        Args:
            telegram_user_id: Telegram user ID
        """
        if telegram_user_id in self.sessions:
            session_id = self.sessions[telegram_user_id]["session_id"]
            del self.sessions[telegram_user_id]
            logger.info(f"Deleted session {session_id} for user {telegram_user_id}")
    
    def _is_expired(self, session: Dict) -> bool:
        """Check if session has expired"""
        last_activity = session["last_activity"]
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)
        
        timeout = timedelta(hours=self.timeout_hours)
        return datetime.now() - last_activity > timeout
    
    def cleanup_expired_sessions(self):
        """Remove all expired sessions"""
        expired_users = [
            user_id for user_id, session in self.sessions.items()
            if self._is_expired(session)
        ]
        
        for user_id in expired_users:
            self.delete_session(user_id)
        
        if expired_users:
            logger.info(f"Cleaned up {len(expired_users)} expired sessions")
        
        return len(expired_users)

