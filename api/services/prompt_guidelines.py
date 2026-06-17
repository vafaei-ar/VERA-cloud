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


def assistant_guidelines() -> str:
    """Return the shared behavioral guidelines block appended to system prompts.

    Built from the constants above so all prompt-construction sites stay in sync.
    Later focus-group tasks (A.4 specificity, A.9 plain language) extend this.
    """
    parts = [HUMAN_OVERSIGHT]
    return "\n\n".join(p for p in parts if p)
