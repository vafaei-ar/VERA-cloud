"""
Enhanced Dialog Engine for VERA Cloud
Supports both guided and RAG-enhanced conversation modes
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import yaml
from datetime import datetime
import json

from .prompt_guidelines import assistant_guidelines, is_boilerplate_only
from .roles import normalize_role, greeting_framing, prompt_framing
from . import flagging
from .empathy import acknowledge as empathy_ack

logger = logging.getLogger(__name__)

class EnhancedDialog:
    def __init__(self, azure_openai_service, azure_search_service, redis_cache_service,
                 scenario_path: str, honorific: str = "", patient_name: str = "",
                 role: str = "survivor", patient_context=None, empathy: bool = False):
        self.openai = azure_openai_service
        self.search = azure_search_service
        self.cache = redis_cache_service
        self.honorific = honorific
        self.patient_name = patient_name
        self.role = normalize_role(role)  # A.6 — survivor | caregiver | clinician
        # Optional empathetic acknowledgments (DRAFT). Off unless explicitly enabled.
        self.empathy = bool(empathy)
        # B.3 — optional PatientContext (None => generic mode, no history-dependent flags)
        self.patient_context = patient_context
        # C.3 — flags accumulated across the session (highest severity wins).
        self.flags = []
        
        # Load scenario
        self.scenario = self._load_scenario(scenario_path)
        self.mode = self.scenario.get("meta", {}).get("mode", "guided")
        
        # Dialog state
        self.current_index = 0
        self.responses = {}
        self.session_context = {
            "start_time": datetime.now().isoformat(),
            "recovery_stage": "early",  # early, mid, late
            "risk_level": "low",  # low, medium, high
            "emergency_detected": False
        }
        
        logger.info(f"Initialized dialog in {self.mode} mode")
    
    def _load_scenario(self, scenario_path: str) -> Dict:
        """Load scenario from YAML file"""
        try:
            with open(scenario_path, 'r') as f:
                scenario = yaml.safe_load(f)
            return scenario
        except Exception as e:
            logger.error(f"Failed to load scenario: {e}")
            return {"meta": {"mode": "guided"}, "flow": []}
    
    def build_greeting(self) -> str:
        """Build personalized greeting"""
        try:
            template = self.scenario.get("greeting", {}).get("template", "")
            variables = self.scenario.get("greeting", {}).get("variables", [])
            
            # Compute time of day
            hour = datetime.now().hour
            if hour < 12:
                timeofday = "morning"
            elif hour < 17:
                timeofday = "afternoon"
            else:
                timeofday = "evening"
            
            # Replace variables
            greeting = template.format(
                timeofday=timeofday,
                honorific=self.honorific,
                patient_name=self.patient_name,
                organization=self.scenario.get("meta", {}).get("organization", ""),
                site=self.scenario.get("meta", {}).get("site", "")
            )

            # A.6 — append role framing (caregiver/clinician); clinical content unchanged.
            framing = greeting_framing(self.role)
            if framing:
                greeting = f"{greeting.rstrip()} {framing}"

            return greeting
        except Exception as e:
            logger.error(f"Failed to build greeting: {e}")
            return "Hello! I'm your AI stroke navigator. How are you feeling today?"
    
    def get_current_question(self) -> Optional[Dict]:
        """Get current question from flow"""
        try:
            if self.current_index < len(self.scenario.get("flow", [])):
                question = self.scenario["flow"][self.current_index]
                if question.get("type") != "section":
                    return question
            return None
        except Exception as e:
            logger.error(f"Failed to get current question: {e}")
            return None
    
    def advance_to_next(self) -> bool:
        """Advance to next question in flow"""
        try:
            old_index = self.current_index
            self.current_index += 1
            logger.info(f"Advanced from index {old_index} to {self.current_index}")
            
            # Skip section markers
            while (self.current_index < len(self.scenario.get("flow", [])) and 
                   (self.scenario["flow"][self.current_index].get("type") == "section" or 
                    self.scenario["flow"][self.current_index].get("section"))):
                logger.info(f"Skipping section marker at index {self.current_index}: {self.scenario['flow'][self.current_index]}")
                self.current_index += 1
            
            logger.info(f"Final index: {self.current_index}, flow length: {len(self.scenario.get('flow', []))}")
            return self.current_index < len(self.scenario.get("flow", []))
        except Exception as e:
            logger.error(f"Failed to advance to next question: {e}")
            return False
    
    def get_progress_percentage(self) -> float:
        """Get conversation progress as percentage"""
        try:
            total_questions = len([q for q in self.scenario.get("flow", []) if q.get("type") != "section" and not q.get("section")])
            if total_questions == 0:
                return 0.0
            
            # Count actual questions answered (excluding section markers)
            questions_answered = 0
            for i in range(min(self.current_index, len(self.scenario.get("flow", [])))):
                item = self.scenario["flow"][i]
                if item.get("type") != "section" and not item.get("section"):
                    questions_answered += 1
            
            progress = min(100.0, (questions_answered / total_questions) * 100)
            return progress
        except Exception as e:
            logger.error(f"Progress calculation failed: {e}")
            return 0.0
    
    def is_complete(self) -> bool:
        """Check if dialog is complete"""
        return self.current_index >= len(self.scenario.get("flow", []))
    
    async def process_user_response(self, user_input: str, confidence: float = 0.0) -> Dict:
        """Process user response and generate next action"""
        try:
            current_question = self.get_current_question()
            if not current_question:
                return await self._handle_completion()
            
            # Store user response
            self.responses[current_question["key"]] = {
                "text": user_input,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update session context
            await self._update_session_context(user_input, current_question)

            # C.3 — evaluate flags on the patient's words and accumulate (skip the
            # consent confirm step). Tier-1 is independent of context/urgency (B.4).
            if current_question.get("key") != "consent":
                self._accumulate_flags(user_input)

            # Process based on mode and question type
            if self.mode == "rag_enhanced" and current_question.get("type") != "confirm":
                response = await self._process_rag_enhanced(user_input, current_question)
            else:
                response = await self._process_guided(user_input, current_question)

            # Optional empathy: prepend ONE short acknowledgment to the next
            # question, when enabled and the patient expressed distress. Never on
            # consent, completion, emergency, or denial turns. DRAFT.
            if self.empathy and current_question.get("key") != "consent":
                skip = {"completion", "emergency", "denial_with_continuation", "error"}
                ack = empathy_ack(user_input)
                msg = response.get("message") if isinstance(response, dict) else None
                if ack and msg and response.get("type") not in skip:
                    response["message"] = f"{ack} {msg}"
            return response
                
        except Exception as e:
            logger.error(f"Failed to process user response: {e}")
            return {
                "type": "error",
                "message": "I'm sorry, I didn't catch that. Could you please repeat?",
                "next_action": "wait_for_response"
            }
    
    async def _process_guided(self, user_input: str, current_question: Dict) -> Dict:
        """Process response in guided mode (original behavior)"""
        try:
            question_type = current_question.get("type", "free")
            logger.info(f"Processing guided response: user_input='{user_input}', question_type='{question_type}', current_question={current_question}")
            
            if question_type == "confirm":
                return await self._handle_confirm_response(user_input, current_question)
            else:
                return await self._handle_free_response(user_input, current_question)
                
        except Exception as e:
            logger.error(f"Guided processing failed: {e}")
            return await self._get_fallback_response()
    
    async def _process_rag_enhanced(self, user_input: str, current_question: Dict) -> Dict:
        """Process response in RAG-enhanced mode"""
        try:
            # Get contextual knowledge
            rag_config = current_question.get("rag_enhancement", {})
            context = await self.search.get_contextual_knowledge(
                user_input, 
                current_question["key"], 
                self.session_context
            )
            
            # Check for emergency keywords
            if context["knowledge"]["emergency"]:
                self.session_context["emergency_detected"] = True
                return await self._handle_emergency_response(user_input, context)
            
            # Check if we should limit follow-up questions
            follow_up_count = self.responses.get(f"{current_question['key']}_followups", 0)
            max_followups = 2  # Limit to 2 follow-up questions per main question
            
            if follow_up_count < max_followups:
                # Generate RAG-enhanced response with follow-up
                rag_response = await self.openai.generate_rag_response(
                    user_input=user_input,
                    context=context["knowledge"]["general"],
                    system_prompt=self._build_rag_system_prompt(current_question, context)
                )

                # A.4 — flag responses whose only substantive content is a
                # variability disclaimer (logged for QA; does not block).
                try:
                    if is_boilerplate_only(rag_response.get("response", "")):
                        rag_response["boilerplate_flag"] = True
                        logger.warning(
                            "Boilerplate-only response detected for question "
                            f"'{current_question.get('key')}'"
                        )
                except Exception as _e:  # pragma: no cover - defensive
                    logger.debug(f"boilerplate check skipped: {_e}")

                # Generate follow-up question
                follow_up = await self.openai.generate_follow_up_question(
                    user_input=user_input,
                    current_question=current_question["prompt"],
                    medical_context=context["knowledge"]["general"]
                )
                
                # Increment follow-up counter
                self.responses[f"{current_question['key']}_followups"] = follow_up_count + 1
                
                return {
                    "type": "rag_enhanced",
                    "message": follow_up,
                    "rag_response": rag_response,
                    "medical_context": context,
                    "next_action": "wait_for_response",
                    "progress": self.get_progress_percentage()
                }
            else:
                # Max follow-ups reached, advance to next question
                logger.info(f"Max follow-ups reached for question {current_question['key']}, advancing to next question")
                
                # Generate acknowledgment response
                acknowledgment = await self.openai.generate_acknowledgment_response(
                    user_input=user_input,
                    medical_context=context["knowledge"]["general"]
                )
                
                # Advance to next question
                if self.advance_to_next():
                    next_question = self.get_current_question()
                    if next_question:
                        progress = self.get_progress_percentage()
                        return {
                            "type": "rag_enhanced",
                            "message": f"{acknowledgment} {next_question['prompt']}",
                            "rag_response": {"acknowledgment": acknowledgment},
                            "medical_context": context,
                            "next_action": "wait_for_response",
                            "progress": progress
                        }
                
                # No more questions, handle completion
                return await self._handle_completion()
            
        except Exception as e:
            logger.error(f"RAG processing failed: {e}")
            return await self._process_guided(user_input, current_question)
    
    async def _handle_confirm_response(self, user_input: str, current_question: Dict) -> Dict:
        """Handle yes/no confirmation responses"""
        try:
            # Simple pattern matching for yes/no
            user_lower = user_input.lower()
            yes_patterns = ["yes", "yeah", "yep", "i consent", "ok", "okay", "sure", "agree"]
            no_patterns = ["no", "don't", "decline", "disagree", "not", "refuse"]
            
            is_yes = any(pattern in user_lower for pattern in yes_patterns)
            is_no = any(pattern in user_lower for pattern in no_patterns)
            
            if is_yes:
                # Advance to next question
                if self.advance_to_next():
                    next_question = self.get_current_question()
                    if next_question:
                        return {
                            "type": "question",
                            "message": next_question["prompt"],
                            "next_action": "wait_for_response",
                            "progress": self.get_progress_percentage()
                        }
                return await self._handle_completion()
            
            elif is_no:
                # Handle denial - provide explanation but continue conversation
                on_deny = current_question.get("on_deny", "I understand. Thank you for your time.")
                
                # Advance to next question after denial
                if self.advance_to_next():
                    next_question = self.get_current_question()
                    if next_question:
                        return {
                            "type": "denial_with_continuation",
                            "message": on_deny,
                            "next_question": next_question["prompt"],
                            "next_action": "wait_for_response",
                            "progress": self.get_progress_percentage()
                        }
                
                # If no next question, end session
                return {
                    "type": "denial",
                    "message": on_deny,
                    "next_action": "end_session"
                }
            
            else:
                # Unclear response
                return {
                    "type": "clarification",
                    "message": "I didn't quite catch that. Could you please say yes or no?",
                    "next_action": "wait_for_response",
                    "progress": self.get_progress_percentage()
                }
                
        except Exception as e:
            logger.error(f"Confirm response handling failed: {e}")
            return await self._get_fallback_response()
    
    async def _handle_free_response(self, user_input: str, current_question: Dict) -> Dict:
        """Handle free-form text responses"""
        try:
            logger.info(f"Handling free response: user_input='{user_input}', current_question={current_question}")
            # Advance to next question
            if self.advance_to_next():
                next_question = self.get_current_question()
                logger.info(f"Advanced to next question: {next_question}")
                if next_question and next_question.get("prompt"):
                    return {
                        "type": "question",
                        "message": next_question["prompt"],
                        "next_action": "wait_for_response",
                        "progress": self.get_progress_percentage()
                    }
                elif next_question and next_question.get("type") == "section":
                    # If we landed on a section marker, advance again to get the actual question
                    logger.info(f"Landed on section marker, advancing again")
                    if self.advance_to_next():
                        next_question = self.get_current_question()
                        logger.info(f"Advanced to actual question: {next_question}")
                        if next_question and next_question.get("prompt"):
                            return {
                                "type": "question",
                                "message": next_question["prompt"],
                                "next_action": "wait_for_response",
                                "progress": self.get_progress_percentage()
                            }
            logger.info("No next question, handling completion")
            return await self._handle_completion()
            
        except Exception as e:
            logger.error(f"Free response handling failed: {e}")
            return await self._get_fallback_response()
    
    async def _handle_emergency_response(self, user_input: str, context: Dict) -> Dict:
        """Handle emergency detection"""
        try:
            emergency_message = self.scenario.get("emergency_disclaimer", 
                "If you are experiencing a medical emergency, please call 911 immediately.")
            
            return {
                "type": "emergency",
                "message": emergency_message,
                "emergency_detected": True,
                "next_action": "end_session",
                "medical_context": context
            }
            
        except Exception as e:
            logger.error(f"Emergency response handling failed: {e}")
            return await self._get_fallback_response()
    
    async def _handle_completion(self) -> Dict:
        """Handle dialog completion"""
        try:
            wrapup_message = self.scenario.get("wrapup", {}).get("message", 
                "Thank you for your time. A member of our care team will review your responses.")
            
            return {
                "type": "completion",
                "message": wrapup_message,
                "next_action": "end_session",
                "responses": self.responses,
                "session_summary": self._generate_session_summary(),
                "progress": 100.0
            }
            
        except Exception as e:
            logger.error(f"Completion handling failed: {e}")
            return {
                "type": "completion",
                "message": "Thank you for your time.",
                "next_action": "end_session",
                "progress": 100.0
            }
    
    async def _update_session_context(self, user_input: str, current_question: Dict):
        """Update session context based on user input"""
        try:
            # Update recovery stage based on responses
            if "discharge" in user_input.lower() or "hospital" in user_input.lower():
                self.session_context["recovery_stage"] = "early"
            elif "weeks" in user_input.lower() or "months" in user_input.lower():
                self.session_context["recovery_stage"] = "mid"
            
            # Update risk level based on symptoms
            risk_keywords = ["pain", "headache", "weakness", "dizzy", "confused"]
            if any(keyword in user_input.lower() for keyword in risk_keywords):
                self.session_context["risk_level"] = "medium"
            
            # Check for emergency keywords
            emergency_keywords = ["emergency", "911", "severe", "sudden", "can't move"]
            if any(keyword in user_input.lower() for keyword in emergency_keywords):
                self.session_context["risk_level"] = "high"
                self.session_context["emergency_detected"] = True
                
        except Exception as e:
            logger.error(f"Session context update failed: {e}")
    
    def _patient_background_block(self) -> str:
        """B.3 — background-only patient context for the system prompt.

        Used to ask better questions and personalize plain-language information.
        Explicitly NOT for diagnosis and NOT to be read back to the patient verbatim.
        Empty string in generic mode (no PatientContext).
        """
        ctx = self.patient_context
        if ctx is None:
            return ("No patient medical history is loaded for this call (generic mode). "
                    "Do not assume any history.")
        try:
            summary = ctx.to_background_summary()
        except Exception:
            summary = ""
        if not summary:
            return ""
        return (
            "BACKGROUND (patient history, for your awareness only): " + summary + ". "
            "Use this only to ask better, more relevant questions and to personalize "
            "general information. Do NOT diagnose, do NOT make treatment decisions, and "
            "do NOT read the medical record back to the patient verbatim."
        )

    def _accumulate_flags(self, user_input: str) -> None:
        """C.3 — merge any non-routine flags from this response into self.flags."""
        try:
            result = self.evaluate_flags(user_input)
            for f in result.get("flags", []):
                if f.get("rule_id") == "t3_routine":
                    continue
                if not any(e.get("rule_id") == f.get("rule_id") for e in self.flags):
                    self.flags.append(f)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"flag accumulation failed: {e}")

    def overall_tier(self) -> int:
        """Highest-severity tier across accumulated flags (3 = routine if none)."""
        return min([f.get("tier", 3) for f in self.flags], default=3)

    def evaluate_flags(self, user_text: str, user_urgency: Optional[str] = None) -> Dict:
        """B.3/B.4 — evaluate flags for a patient response using loaded context.

        In generic mode (no PatientContext) only context-free rules (incl. Tier-1)
        can fire. Tier-1 always fires from the words alone.
        """
        return flagging.evaluate(user_text, context=self.patient_context,
                                 user_urgency=user_urgency)

    def _build_rag_system_prompt(self, current_question: Dict, context: Dict) -> str:
        """Build system prompt for RAG responses"""
        try:
            base_prompt = """You are a medical AI assistant conducting a stroke recovery follow-up call.
            Use the provided medical knowledge to ask informed follow-up questions.
            Be empathetic, professional, and focused on patient safety."""

            if context["knowledge"]["emergency"]:
                base_prompt += "\n\nWARNING: Emergency keywords detected. Prioritize patient safety."

            if context["knowledge"]["medication"]:
                base_prompt += "\n\nFocus on medication adherence and side effects."

            if context["knowledge"]["lifestyle"]:
                base_prompt += "\n\nEmphasize lifestyle modifications and daily activities."

            # Shared behavioral guidelines (human oversight A.1, etc.) — single source of truth.
            base_prompt += "\n\n" + assistant_guidelines()

            # A.6 — role framing (wording only; clinical content unchanged).
            base_prompt += "\n\n" + prompt_framing(self.role)

            # B.3 — inject patient history as BACKGROUND only (if loaded).
            base_prompt += "\n\n" + self._patient_background_block()

            return base_prompt
            
        except Exception as e:
            logger.error(f"System prompt building failed: {e}")
            return "You are a medical AI assistant. Be empathetic and professional."
    
    def _generate_session_summary(self) -> Dict:
        """Generate session summary for care team"""
        try:
            return {
                "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "patient_name": self.patient_name,
                "honorific": self.honorific,
                "role": self.role,
                "duration_minutes": self._calculate_duration(),
                "questions_answered": len(self.responses),
                "recovery_stage": self.session_context["recovery_stage"],
                "risk_level": self.session_context["risk_level"],
                "emergency_detected": self.session_context["emergency_detected"],
                "flags": self.flags,                       # C.3 — accumulated B.4 flags
                "overall_tier": self.overall_tier(),       # C.3
                "context_loaded": self.patient_context is not None,
                "responses": self.responses,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Session summary generation failed: {e}")
            return {"error": "Failed to generate summary"}
    
    def _calculate_duration(self) -> float:
        """Calculate session duration in minutes"""
        try:
            start_time = datetime.fromisoformat(self.session_context["start_time"])
            duration = datetime.now() - start_time
            return round(duration.total_seconds() / 60, 2)
        except Exception:
            return 0.0
    
    async def _get_fallback_response(self) -> Dict:
        """Get fallback response when processing fails"""
        return {
            "type": "error",
            "message": "I'm sorry, I didn't catch that. Could you please repeat?",
            "next_action": "wait_for_response",
            "progress": self.get_progress_percentage()
        }
