"""Ask-VERA — retrieval-only patient FAQ (DRAFT; requires Dr. Zand sign-off).

No language model. Deterministic keyword match against config/faq.yaml. Returns
a clinician-approved answer verbatim, or None so the caller can refuse. The YAML
file is the entire review surface. No Azure / network deps.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Shown when nothing approved matches the question.
REFUSAL = (
    "I'm sorry, I can only share a few approved topics. For anything else, "
    "please contact your care team. If this is an emergency, call 911."
)


def _faq_path() -> Path:
    default = Path(__file__).resolve().parent.parent.parent / "config" / "faq.yaml"
    return Path(os.environ.get("FAQ_PATH", str(default)))


def _load() -> Dict[str, Any]:
    p = _faq_path()
    if p.exists():
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def disclaimer() -> str:
    return _load().get("disclaimer", "")


def lookup(question: str) -> Optional[Dict[str, Any]]:
    """Return the best-matching approved entry, or None.

    Match = the question text contains one of an entry's `keywords`. The entry
    with the most keyword hits wins; ties keep the first. Deterministic.
    """
    q = (question or "").lower()
    if not q.strip():
        return None
    entries: List[Dict[str, Any]] = _load().get("entries", []) or []
    best: Optional[Dict[str, Any]] = None
    best_hits = 0
    for entry in entries:
        kws = [str(k).lower() for k in entry.get("keywords", []) or []]
        hits = sum(1 for k in kws if k and k in q)
        if hits > best_hits:
            best_hits, best = hits, entry
    return best if best_hits > 0 else None
