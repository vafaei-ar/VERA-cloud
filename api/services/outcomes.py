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
