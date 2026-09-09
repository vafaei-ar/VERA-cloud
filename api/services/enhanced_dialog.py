"""
Enhanced Dialog Engine for VERA Cloud
Supports both guided and RAG-enhanced conversation modes
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import yaml
from datetime import datetime, timezone
import json
import re

from .prompt_guidelines import assistant_guidelines, is_boilerplate_only
from .roles import normalize_role, greeting_framing, prompt_framing
from . import flagging
from .empathy import acknowledge as empathy_ack
from .clinical_policy import load_policy
from . import faq

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
        self.policy = load_policy()
        self.consent = "pending"
        self.terminal_state = None
        self.user_urgency = None
        self.callback_requested = False
        self.followup = None
        self.active_pathway = None
        self.turn_receipts = {}
        self.audio_turns = {}
        self.transcription_reviews = []
        self.stroke_type = "unknown"
        
        # Load scenario
        self.scenario = self._load_scenario(scenario_path)
        # Do not assume ischemic stroke or conflate answer consent with audio.
        for question in self.scenario.get("flow", []):
            if question.get("key") == "consent":
                question["prompt"] = self.policy["messages"]["consent"]
            if question.get("key") == "know_ischemic":
                question["prompt"] = "Would you like a short explanation of stroke?"
                question["on_deny"] = "That's okay."
            if question.get("key") == "urgency_self_report":
                question["prompt"] = self.policy["messages"]["urgency_choices"]
        if self.role == "caregiver":
            flow = self.scenario.get("flow", [])
            urgency_index = next((i for i, q in enumerate(flow) if q.get("key") == "urgency_self_report"), len(flow))
            flow.insert(urgency_index, {"key": "caregiver_support", "type": "free", "prompt": self.policy["messages"]["caregiver_support"]})
        self.mode = self.scenario.get("meta", {}).get("mode", "guided")
        
        # Dialog state
        self.current_index = 0
        self.responses = {}
        self.session_context = {
            "start_time": datetime.now(timezone.utc).isoformat(),
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
        """Short introduction without inferring a person's stroke type."""
        return self.policy["messages"]["introduction"]

    def snapshot(self):
        """JSON-only state; keep the exact policy and question flow on reconnect."""
        fields = ("honorific", "patient_name", "role", "empathy", "flags", "policy", "consent",
                  "terminal_state", "user_urgency", "callback_requested", "followup", "active_pathway",
                  "turn_receipts", "audio_turns", "transcription_reviews", "stroke_type", "scenario", "mode",
                  "current_index", "responses", "session_context")
        return {"schema_version": 1, "state": {key: getattr(self, key) for key in fields},
                "patient_context": self.patient_context.to_dict() if self.patient_context else None}

    @classmethod
    def restore(cls, snapshot, openai=None, search=None, cache=None):
        from .patient_data import PatientContext
        if snapshot.get("schema_version") != 1:
            raise ValueError("Unsupported saved conversation")
        state = snapshot["state"]
        # Do not silently run old clinical settings through changed detection code.
        if state["policy"]["engine_digest"] != load_policy()["engine_digest"]:
            raise ValueError("Clinical engine changed; a new check-in is required")
        if not 0 <= state["current_index"] <= len(state["scenario"]["flow"]):
            raise ValueError("Invalid saved question position")
        restored = cls.__new__(cls)
        restored.__dict__.update(state)
        restored.openai, restored.search, restored.cache = openai, search, cache
        restored.patient_context = PatientContext(**snapshot["patient_context"]) if snapshot.get("patient_context") else None
        return restored

    def current_prompt(self):
        if self.followup:
            branch = self.policy["symptom_followups"].get("pathways", {}).get(self.active_pathway, {})
            timing = self.followup == self.policy["symptom_followups"]["max_questions"]
            return branch.get("timing_prompt" if timing else "impact_prompt",
                              self.policy["messages"]["symptom_timing" if timing else "symptom_impact"])
        question = self.get_current_question()
        return question["prompt"] if question else "Your check-in answers are saved."

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
            if self.terminal_state:
                return {"type": "session_ended", "state": self.terminal_state, "next_action": "end_session", "message": "This check-in has ended."}
            current_question = self.get_current_question()
            if not current_question:
                return await self._handle_completion()

            if current_question.get("key") == "consent":
                # Consent is processed BEFORE recording any clinical response.
                return await self._handle_confirm_response(user_input, current_question)
            if self.consent != "accepted":
                return {"type": "clarification", "message": self.policy["messages"]["consent"]}
            normalized = user_input.lower().strip().rstrip(".!?")
            if normalized in {"stop", "stop the check-in", "end check-in", "withdraw consent"}:
                self.terminal_state = "withdrawn"
                return {"type": "session_ended", "state": "withdrawn", "next_action": "end_session", "message": "The check-in has stopped. Answers already shared remain saved; contact the study team about deletion."}
            # Safety precedes FAQ detours and all symptom clarification.
            self._accumulate_flags(user_input)
            if self.followup and self.active_pathway:
                self._evaluate_pathway_answer(user_input)
            if self.overall_tier() == 1:
                return self._emergency_stop(user_input)
            if normalized in {"call me", "call back", "callback", "speak to a person", "talk to a person", "human help"}:
                self.callback_requested = True
                return {"type": "response", "message": self.policy["messages"]["callback"] + " " + self.current_prompt()}
            if normalized in {"repeat", "repeat that", "replay"}:
                return {"type": "question", "message": self.current_prompt()}
            if normalized in {"skip", "skip this question"}:
                self.responses[current_question["key"]] = {"skipped": True}
                self.followup = None
                self.active_pathway = None
                return await self._handle_free_response(user_input, current_question)
            if normalized in {"my stroke was hemorrhagic", "my stroke was ischemic", "i don't know my stroke type"}:
                self.stroke_type = "hemorrhagic" if "hemorrhagic" in normalized else "ischemic" if "ischemic" in normalized else "unknown"
                self.responses["stroke_type_correction"] = {"type": self.stroke_type, "source": "respondent"}
                return {"type": "response", "message": self.policy["education"][self.stroke_type] + " " + self.current_prompt()}
            if user_input.strip().endswith("?") and current_question.get("key") != "urgency_self_report":
                hit = faq.lookup(user_input)
                return {"type": "response", "message": (hit["answer"] if hit else faq.REFUSAL) + " Back to the check-in: " + self.current_prompt()}
            if current_question.get("key") == "urgency_self_report":
                if normalized in {"not sure", "i don't know", "i am not sure"}:
                    normalized = "unsure"
                values = re.findall(r"\b(routine|soon|urgent|unsure)\b", normalized)
                if len(set(values)) != 1 or re.search(r"\bnot\b", normalized):
                    return {"type": "clarification", "message": self.policy["messages"]["urgency_choices"]}
                self.user_urgency = values[0]
            if self.followup:
                self.responses.setdefault(current_question["key"] + "_details", []).append(user_input)
                self.followup -= 1
                if self.followup:
                    pathway = self.policy["symptom_followups"].get("pathways", {}).get(self.active_pathway, {})
                    return {"type": "question", "message": pathway.get("impact_prompt", self.policy["messages"]["symptom_impact"])}
                self.followup = None
                self.active_pathway = None
                return await self._handle_free_response(user_input, current_question)
            
            # Store user response
            self.responses[current_question["key"]] = {
                "text": user_input,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Update session context
            await self._update_session_context(user_input, current_question)

            # C.3 — evaluate flags on the patient's words and accumulate (skip the
            # consent confirm step). Tier-1 is independent of context/urgency (B.4).
            followups = self.policy["symptom_followups"]
            if followups["enabled"] and current_question.get("key") in followups["question_keys"] and followups["max_questions"]:
                # Only clear negative/no-change answers skip the clarification.
                if normalized not in {"no", "none", "no new symptoms", "no changes", "fine", "good", "same", "unchanged"}:
                    self.followup = followups["max_questions"]
                    pathways = followups.get("pathways", {})
                    self.active_pathway = next((name for name, branch in pathways.items()
                        if any(re.search(r"\b" + re.escape(term) + r"\b", normalized) for term in branch["triggers"])), None)
                    branch = pathways.get(self.active_pathway, {})
                    if self.active_pathway:
                        self._evaluate_pathway_answer(user_input)
                        if self.overall_tier() == 1:
                            return self._emergency_stop(user_input)
                    return {"type": "question", "message": branch.get("timing_prompt", self.policy["messages"]["symptom_timing"])}

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
            logger.info("Processing guided question type: %s", question_type)
            
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
            user_lower = user_input.lower().strip().rstrip(".!?")
            is_yes = user_lower in {"yes", "yeah", "yep", "i consent", "ok", "okay", "sure", "i agree", "yes please"}
            is_no = user_lower in {"no", "no thanks", "no thank you", "i decline", "i do not consent", "i don't consent", "decline", "not now"}
            if current_question.get("key") == "consent":
                if is_no:
                    self.consent = "declined"
                    self.terminal_state = "declined"
                    return {"type": "session_ended", "state": "declined", "message": self.policy["messages"]["declined"], "next_action": "end_session"}
                if is_yes:
                    self.consent = "accepted"
            
            if is_yes:
                education = self.policy["education"].get(self.stroke_type, self.policy["education"]["unknown"]) + " " if current_question.get("key") == "know_ischemic" else ""
                # Advance to next question
                if self.advance_to_next():
                    next_question = self.get_current_question()
                    if next_question:
                        return {
                            "type": "question",
                            "message": education + next_question["prompt"],
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
                            "type": "question",
                            "message": on_deny + " " + next_question["prompt"],
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
            logger.info("Processing free response for question %s", current_question.get("key"))
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
            self.terminal_state = "escalated"
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
                    f["detected_at"] = datetime.now(timezone.utc).isoformat()
                    self.flags.append(f)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"flag accumulation failed: {e}")
            raise

    def _evaluate_pathway_answer(self, text):
        """DRAFT deterministic clarification. Baseline labels never erase flags."""
        branch = self.policy["symptom_followups"]["pathways"][self.active_pathway]
        value = text.lower()
        unknown = bool(re.search(r"\b(unsure|unknown|not sure|don't know|cannot tell)\b", value))
        baseline = not unknown and bool(re.search(r"\b(unchanged|usual|same|baseline)\b", value))
        changed = any(re.search(r"\b" + term + r"\b", value) and not flagging._negated_near(value, term) for term in ("new", "worse", "worsening"))
        label = "unknown" if unknown else "new_or_worse" if changed else "baseline_reported" if baseline else "unclassified"
        self.responses.setdefault("symptom_clarifications", []).append({"pathway": self.active_pathway, "reported_timing": label, "text": text})
        immediate = any(re.search(r"\b" + re.escape(term) + r"\b", value) and not flagging._negated_near(value, term) for term in branch["immediate_terms"])
        tier = 1 if immediate else branch["new_or_worse_tier"] if changed else branch["unknown_tier"] if unknown else None
        if tier:
            rule_id = f"pathway_{self.active_pathway}_{label}_{tier}"
            if not any(flag.get("rule_id") == rule_id for flag in self.flags):
                self.flags.append({"tier": tier, "rule_id": rule_id, "category": self.active_pathway,
                    "reason": "DRAFT symptom pathway: " + ("emergency qualifier" if immediate else label),
                    "matched": text, "detected_at": datetime.now(timezone.utc).isoformat()})

    def _emergency_stop(self, user_input):
        self.responses.setdefault("safety_reports", []).append(user_input)
        self.terminal_state = "escalated"
        return {"type": "emergency_alert", "state": "escalated", "next_action": "end_session", "message": self.policy["messages"]["emergency"]}

    def overall_tier(self) -> int:
        """Highest-severity tier across accumulated flags (3 = routine if none)."""
        return min([f.get("tier", 3) for f in self.flags], default=3)

    def evaluate_flags(self, user_text: str, user_urgency: Optional[str] = None) -> Dict:
        """B.3/B.4 — evaluate flags for a patient response using loaded context.

        In generic mode (no PatientContext) only context-free rules (incl. Tier-1)
        can fire. Tier-1 always fires from the words alone.
        """
        return flagging.evaluate(user_text, context=self.patient_context,
                                 user_urgency=user_urgency, policy=self.policy)

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
                "session_id": f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
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
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Session summary generation failed: {e}")
            return {"error": "Failed to generate summary"}
    
    def _calculate_duration(self) -> float:
        """Calculate session duration in minutes"""
        try:
            start_time = datetime.fromisoformat(self.session_context["start_time"])
            duration = datetime.now(timezone.utc) - start_time
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
