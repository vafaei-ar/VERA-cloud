"""Optional empathetic acknowledgments (DRAFT — requires clinician sign-off).

When enabled per-session, the dialog may prepend ONE short, non-medical
acknowledgment before the next question if the patient expresses distress
(e.g. "I have pain" -> "I'm sorry to hear you're in pain." -> next question).

Templated only — NO language model, no advice, no diagnosis, no medical content.
Off by default. The phrase list below is the entire review surface.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

# (cue keywords) -> short acknowledgment. DRAFT wording for clinical review.
_ACKS: List[Tuple[Tuple[str, ...], str]] = [
    (("pain", "hurts", "hurting", "sore", "aches", "aching"), "I'm sorry to hear you're in pain."),
    (("tired", "exhausted", "fatigue", "no energy", "worn out"), "That sounds tiring."),
    (("sad", "down", "depressed", "feeling low", "crying"), "I'm sorry you're feeling low."),
    (("worried", "anxious", "scared", "afraid", "nervous", "stressed"), "I understand this can feel worrying."),
    (("lonely", "alone", "isolated"), "I'm sorry you're feeling alone."),
    (("frustrated", "angry", "upset"), "I hear you — that sounds frustrating."),
    (("struggling", "hard time", "difficult", "overwhelmed", "can't cope"), "That sounds hard."),
]


def acknowledge(text: str) -> Optional[str]:
    """Return one short empathetic sentence if a distress cue is present, else None.
    Deterministic; first matching cue wins."""
    t = (text or "").lower()
    if not t.strip():
        return None
    for cues, phrase in _ACKS:
        if any(c in t for c in cues):
            return phrase
    return None
