"""
Role / track support (A.6) — Survivor, Caregiver, Clinician.

Only the *framing/wording* changes by role; clinical content stays identical.
This module is the single source of truth for role names and role-specific
framing, reused by the dialog engine (A.6) and the Part C scenario system.

No Azure / network dependencies — safe to import and unit test in isolation.
"""

from typing import Optional

ROLES = ("survivor", "caregiver", "clinician")
DEFAULT_ROLE = "survivor"


def normalize_role(role: Optional[str]) -> str:
    """Return a known role, defaulting to survivor for unknown/empty input."""
    r = (role or "").strip().lower()
    return r if r in ROLES else DEFAULT_ROLE


def greeting_framing(role: Optional[str]) -> str:
    """Extra greeting sentence for the role (empty for survivor).

    Plain language (A.9). Clinical content is unchanged — this only sets who
    the assistant is talking to.
    """
    r = normalize_role(role)
    if r == "caregiver":
        return ("I understand you are helping care for your family member. "
                "I will ask how they are doing, and you can answer for them.")
    if r == "clinician":
        return "This is a clinician check-in, so I will keep it short and to the point."
    return ""


def prompt_framing(role: Optional[str]) -> str:
    """System-prompt instruction so generated wording matches the role.

    Framing only — never changes clinical thresholds or content.
    """
    r = normalize_role(role)
    if r == "caregiver":
        return ("The person on this call is a CAREGIVER answering about their family "
                "member. Address the caregiver, and refer to the patient as 'your "
                "family member' rather than 'you'. Clinical content is unchanged.")
    if r == "clinician":
        return ("The person on this call is a CLINICIAN. Be concise and use a brief, "
                "factual, data-entry style. Clinical content is unchanged.")
    return ("The person on this call is the stroke survivor. Address them directly "
            "as 'you'.")
