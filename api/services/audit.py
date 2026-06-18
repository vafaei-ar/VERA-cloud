"""
Access audit log (B.5).

Writes one append-only record every time a patient's history is loaded (or a load
is attempted). The audit line captures the ACCESS EVENT only — timestamp, PATID,
caller role, consent flag, and whether context was loaded. It contains NO clinical
content (no diagnoses, meds, vitals, labs, or flags).

Records are JSON lines under AUDIT_PATH (default: <repo>/audit/access_log.jsonl).
No Azure / network dependencies — safe to import and unit test.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def audit_path() -> Path:
    default = Path(__file__).resolve().parent.parent.parent / "audit"
    return Path(os.environ.get("AUDIT_PATH", str(default)))


def _audit_file() -> Path:
    return audit_path() / "access_log.jsonl"


def write_access(patid: Optional[str], role: Optional[str], consent: Optional[bool],
                 context_loaded: bool, session_id: Optional[str] = None,
                 note: Optional[str] = None) -> Dict[str, Any]:
    """Append an access-event audit record. No clinical content is recorded.

    Returns the written record (also useful for tests).
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "event": "patient_history_access",
        "session_id": session_id,
        "patid": patid,
        "role": role,
        "consent": bool(consent) if consent is not None else None,
        "context_loaded": bool(context_loaded),
        "note": note,
    }
    try:
        path = _audit_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:  # pragma: no cover - defensive; never break the call
        logger.error(f"Failed to write audit record: {e}")
    return record
