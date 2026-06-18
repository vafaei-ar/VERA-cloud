"""
Shared system-prompt guidelines for AI-SoNar dialog.

Single source of truth for the behavioral rules that must apply everywhere the
system prompt is constructed. Before this module, the same rules were duplicated
as inline string literals in azure_openai.py and enhanced_dialog.py, which let
them drift apart. Importing from here keeps them consistent.

SAFETY: Guidelines that touch escalation, red flags, human-review pathways, or
response-time promises are DRAFT clinical content and are logged in
CLINICAL_REVIEW_NEEDED.md. Do not treat them as clinically validated.

This module has NO Azure / network dependencies so it can be imported and unit
tested in isolation.
"""

# ===== DRAFT CLINICAL LOGIC — REQUIRES DR. ZAND SIGN-OFF =====
# Rationale: focus group (Phil, Margie) asked that human oversight be visible and
#   that the assistant point worried users to the human-review pathway instead of
#   only disclaiming its own limits. Over-disclaiming was reported to backfire.
# Source: focus group June 2026; standard care-team escalation practice (BE-FAST).
# DO NOT treat as clinically validated.
HUMAN_OVERSIGHT = (
    "A member of the patient's care team reviews what is shared after this call. "
    "When the patient expresses worry or describes a concerning symptom, reassure "
    "them plainly that a real person on the care team will see what they shared and "
    "follow up — and for anything urgent, that they should call their care team or "
    "911 right away. State the human-review pathway clearly when it is relevant. "
    "Do not repeat 'I am not a doctor' style disclaimers more than once; say it "
    "plainly a single time if needed and then be helpful."
)
# ===== END DRAFT CLINICAL LOGIC =====


# Plain-language constraint (A.9). Not clinical content, but safety-relevant:
# stroke survivors (incl. aphasia) need simple wording and slow pacing.
PLAIN_LANGUAGE = (
    "Use plain, everyday language at about a 6th-grade reading level. Keep "
    "sentences short — one idea per sentence. Prefer common words over medical "
    "terms; if a medical term is necessary, explain it in a few simple words. "
    "Keep each turn brief so the patient is not overwhelmed. Speak slowly and "
    "calmly. If the patient seems confused or asks, happily repeat or rephrase "
    "what you said in simpler words. Never rush the patient."
)


# Specific-not-boilerplate constraint (A.4). Focus group (Phil): "all strokes are
# different" as a COMPLETE answer makes people tune out. Acknowledge variability at
# most once, then say something concrete.
SPECIFICITY = (
    "Do not answer only with a generic disclaimer like 'all strokes are different' "
    "or 'every recovery is unique'. You may acknowledge that variability at most "
    "once, but every response must then add something concrete and relevant — for "
    "example, common effects for the patient's stroke type or affected area, or a "
    "practical next step. Make clear that any such information is general "
    "information, not personal medical advice. If you do not have enough specific "
    "information to be concrete, say so plainly and offer to note the question for "
    "the care team."
)

# Phrases that, on their own, signal a content-free variability hedge (A.4 detector).
_BOILERPLATE_MARKERS = (
    "all strokes are different",
    "every stroke is different",
    "everyone is different",
    "every recovery is different",
    "every recovery is unique",
    "each person is different",
    "each patient is different",
    "it depends on the person",
    "it varies from person to person",
    "everybody is different",
)


def is_boilerplate_only(text: str, min_substantive_words: int = 12) -> bool:
    """Heuristic (A.4): True if a response is essentially only a variability hedge.

    Strips out known boilerplate phrases; if little substantive content remains,
    the response is flagged as boilerplate-only. Conservative on purpose — it is a
    lightweight check used to log/flag, not to block.
    """
    if not text:
        return False
    lowered = text.lower()
    if not any(m in lowered for m in _BOILERPLATE_MARKERS):
        return False
    stripped = lowered
    for m in _BOILERPLATE_MARKERS:
        stripped = stripped.replace(m, " ")
    # Remove non-alphabetic noise and count remaining words.
    remaining = [w for w in "".join(
        c if c.isalpha() else " " for c in stripped
    ).split() if len(w) > 2]
    return len(remaining) < min_substantive_words


def assistant_guidelines() -> str:
    """Return the shared behavioral guidelines block appended to system prompts.

    Built from the constants above so all prompt-construction sites stay in sync.
    """
    parts = [HUMAN_OVERSIGHT, PLAIN_LANGUAGE, SPECIFICITY]
    return "\n\n".join(p for p in parts if p)
