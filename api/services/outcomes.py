"""
Outcomes store for AI-SoNar check-ins.

A small, dependency-free JSON-per-session store for assessment outcomes:
user-reported urgency (A.3), model/clinical flags (Part B.4), the clinician
summary (A.11), and related metadata. Each session is one JSON file under
OUTCOMES_PATH (default: <repo>/outcomes/sessions/), so it works with no DB and
is easy to inspect during synthetic testing.

SAFETY / SEPARATION OF SIGNALS (A.3):
`user_reported_urgency` is stored as its OWN field, completely separate from any
model-derived or rule-based flag. User urgency may ADD signal but must NEVER
suppress an automatic red-flag escalation. This module only records data; it does
not decide escalation. Flagging logic (Part B.4) reads these fields but treats
Tier-1 red flags as independent of user urgency.

No Azure / network dependencies — safe to import and unit test in isolation.
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Allowed user-reported urgency values (A.3). Ordered low -> high.
USER_URGENCY_VALUES = ("routine", "soon", "urgent")


def _outcomes_dir() -> Path:
    """Directory where per-session outcome files live (configurable)."""
    default = Path(__file__).resolve().parent.parent.parent / "outcomes" / "sessions"
    return Path(os.environ.get("OUTCOMES_PATH", str(default)))


def _session_file(session_id: str) -> Path:
    # Basic hardening so a session_id cannot escape the outcomes dir.
    safe = "".join(c for c in str(session_id) if c.isalnum() or c in ("-", "_"))
    return _outcomes_dir() / f"{safe}.json"


def load_outcome(session_id: str) -> Dict[str, Any]:
    """Load a session's outcome record, or an empty scaffold if none exists."""
    path = _session_file(session_id)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to read outcome {path}: {e}")
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "user_reported_urgency": None,   # A.3 — separate from model flags
        "flags": [],                     # Part B.4 — model/rule flags
        "updated_at": None,
    }


def save_outcome(record: Dict[str, Any]) -> Path:
    """Persist a session outcome record to its JSON file."""
    record["updated_at"] = datetime.now().isoformat()
    path = _session_file(record["session_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, path)
    return path


def record_session_flags(session_id: str, flags: list) -> Dict[str, Any]:
    """C.3 — persist the session's accumulated flags onto the outcome record.

    This is what the clinician summary (A.11) reads. User urgency, stored
    separately by record_user_urgency, is left untouched.
    """
    record = load_outcome(session_id)
    record["flags"] = list(flags or [])
    save_outcome(record)
    logger.info(f"Recorded {len(record['flags'])} flag(s) for session {session_id}")
    return record


def record_reminder(session_id: str, text: str) -> Dict[str, Any]:
    """A.7 — append a patient voice reminder to the session outcome record.

    Reminders are informational notes for the patient/care team (e.g. an
    appointment). Not clinical content; stored alongside the outcome.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty reminder text.")
    record = load_outcome(session_id)
    record.setdefault("reminders", []).append({
        "text": text,
        "created_at": datetime.now().isoformat(),
    })
    save_outcome(record)
    logger.info(f"Recorded reminder for session {session_id}")
    return record


def _field_usage_file() -> Path:
    return _outcomes_dir() / "_field_usage.json"


def record_field_action(field: str) -> Dict[str, int]:
    """A.11 — increment a counter for a clinician-acted-on field.

    Feeds Aim-2 workflow-fit evidence: which summary fields clinicians actually
    use, so low-value fields can be dropped later. Aggregate only, no PHI.
    """
    path = _field_usage_file()
    counts: Dict[str, int] = {}
    if path.exists():
        try:
            with open(path, "r") as f:
                counts = json.load(f)
        except Exception:  # pragma: no cover - defensive
            counts = {}
    counts[field] = int(counts.get(field, 0)) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(counts, f, indent=2)
    os.replace(tmp, path)
    return counts


# ===== DRAFT CLINICAL LOGIC — REQUIRES DR. ZAND SIGN-OFF =====
# Rationale: oversight is a team (Lisa asked about nurses/PAs/speech therapists,
#   not only physicians). Route alerts to the most appropriate role. This encodes
#   clinical urgency and MUST be reviewed.
# Source: focus group June 2026.
# DO NOT treat as clinically validated.
def suggest_route(flags: list, user_urgency: Optional[str] = None) -> Dict[str, Any]:
    """Suggest which care-team role should see this check-in (DRAFT, A.11).

    `flags` is a list of dicts that may include 'tier' (1=red,2=urgent,3=routine)
    and 'category' (e.g. 'medication', 'rehab', 'bp', 'mood'). Returns a role plus
    priority. Tier-1 always routes to emergency/physician regardless of category or
    user urgency.
    """
    flags = flags or []
    tiers = [f.get("tier") for f in flags if isinstance(f, dict)]
    categories = {f.get("category") for f in flags if isinstance(f, dict)}

    if 1 in tiers:
        return {"role": "physician", "priority": "emergency",
                "reason": "Tier-1 red flag — emergency guidance + physician/care-team escalation."}
    if 2 in tiers:
        if "medication" in categories:
            return {"role": "nurse_or_pharmacist", "priority": "urgent",
                    "reason": "Tier-2 medication concern."}
        if "rehab" in categories or "mobility" in categories:
            return {"role": "therapist", "priority": "urgent",
                    "reason": "Tier-2 rehabilitation/mobility concern."}
        return {"role": "nurse_or_navigator", "priority": "urgent",
                "reason": "Tier-2 urgent clinician review."}
    # No model red/urgent flags: user-reported urgency can raise visibility but
    # never creates a Tier-1 escalation on its own.
    if user_urgency == "urgent":
        return {"role": "nurse_or_navigator", "priority": "review_soon",
                "reason": "Patient marked this urgent (advisory; no automatic red flag)."}
    return {"role": "navigator", "priority": "routine",
            "reason": "No red/urgent flags; routine follow-up."}
# ===== END DRAFT CLINICAL LOGIC =====


def build_clinician_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """A.11 — concise, prioritized clinician summary.

    Leads with flagged items + user-reported urgency; routine/normal items are
    grouped separately so the UI can collapse them and avoid noise. Routing
    (DRAFT) is attached. This function only organizes/presents existing data.
    """
    flags = record.get("flags", []) or []
    priority_flags = [f for f in flags if isinstance(f, dict) and f.get("tier") in (1, 2)]
    routine_flags = [f for f in flags if isinstance(f, dict) and f.get("tier") not in (1, 2)]
    urgency = record.get("user_reported_urgency")

    return {
        "session_id": record.get("session_id"),
        "user_reported_urgency": urgency,           # A.3 — shown alongside model flags
        "priority_items": priority_flags,           # lead with these
        "routine_items": routine_flags,             # collapsible "all normal" noise
        "has_priority": bool(priority_flags) or urgency == "urgent",
        "suggested_route": suggest_route(flags, urgency),  # DRAFT routing
        "generated_at": datetime.now().isoformat(),
    }


def record_user_urgency(session_id: str, urgency: str,
                        role: Optional[str] = None) -> Dict[str, Any]:
    """Record the patient's self-reported urgency (A.3) as its own field.

    Raises ValueError for an unrecognized urgency value. Does NOT touch the
    `flags` list — user urgency never overrides automatic flagging.
    """
    u = (urgency or "").strip().lower()
    if u not in USER_URGENCY_VALUES:
        raise ValueError(
            f"Invalid urgency '{urgency}'. Expected one of {USER_URGENCY_VALUES}."
        )
    record = load_outcome(session_id)
    record["user_reported_urgency"] = u
    record["user_reported_urgency_at"] = datetime.now().isoformat()
    if role:
        record["user_reported_urgency_role"] = role
    save_outcome(record)
    logger.info(f"Recorded user urgency '{u}' for session {session_id}")
    return record
