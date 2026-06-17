"""
Tiered flagging engine (B.4).

================================================================================
# ===== DRAFT CLINICAL LOGIC — REQUIRES DR. ZAND SIGN-OFF =====
# Rationale: provide a realistic, literature-aligned DEFAULT so the system is
#   functional and testable. Tier-1 from stroke warning signs (BE-FAST); Tier-2
#   context-raised using PatientContext; Tier-3 routine.
# Source: focus group June 2026 + standard stroke guidance (BE-FAST). NOT validated.
# DO NOT treat any threshold, red-flag definition, escalation level, or guidance
#   message in this module as clinically final. Every rule here is logged in
#   CLINICAL_REVIEW_NEEDED.md and must be reviewed.
# ===== END DRAFT CLINICAL LOGIC =====
================================================================================

Design guarantees:
- Tier-1 red flags fire from the patient's words ALONE — independent of patient
  context and independent of the user's self-reported urgency.
- User-reported urgency is advisory: it is carried through but NEVER lowers a tier
  and never suppresses a Tier-1 escalation.
- Rules are data-driven and auditable; each returned flag cites the rule/context
  that fired.

No Azure / network dependencies — safe to import and unit test in isolation.
"""

import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

TIER_RED = 1       # emergency — call 911 + automatic escalation
TIER_URGENT = 2    # urgent clinician review (same / next business day)
TIER_ROUTINE = 3   # routine

# ----------------------------------------------------------------------------
# ===== DRAFT CLINICAL LOGIC — REQUIRES DR. ZAND SIGN-OFF =====
# Tier-1 red-flag patterns (BE-FAST + other emergencies). Phrase lists are matched
# against the patient's words. These fire regardless of context/urgency.
TIER1_RULES = [
    ("t1_weakness_onesided", "sudden one-sided weakness or numbness",
     ["one side", "one-sided", "left side", "right side", "side of my body",
      "arm went weak", "leg went weak", "can't move my", "cannot move my",
      "numb on one", "weakness on one"]),
    ("t1_face_droop", "facial droop",
     ["face is drooping", "face drooping", "face droop", "drooping face",
      "side of my face", "mouth is drooping"]),
    ("t1_speech", "new trouble speaking or understanding speech",
     ["can't speak", "cannot speak", "slurred", "slurring", "trouble speaking",
      "can't get my words", "words won't come", "can't understand"]),
    ("t1_headache_worst", "sudden severe headache (worst ever)",
     ["worst headache", "worst ever", "thunderclap", "sudden severe headache"]),
    ("t1_vision", "sudden vision loss or double vision",
     ["lost my vision", "can't see", "cannot see", "double vision",
      "vision went", "blind in"]),
    ("t1_balance", "sudden loss of balance or coordination",
     ["lost my balance", "can't keep my balance", "can't walk", "cannot walk",
      "keep falling", "lost coordination", "room is spinning and i fell"]),
    ("t1_loc", "loss of consciousness",
     ["passed out", "blacked out", "lost consciousness", "fainted"]),
    ("t1_chest_pain", "chest pain",
     ["chest pain", "pain in my chest", "chest is tight", "crushing chest"]),
    ("t1_fall_injury", "a fall with injury",
     ["fell and hit my head", "fell and i'm bleeding", "fell and hurt",
      "fall and hit my head", "injured in a fall"]),
]

# Tier-2 symptom patterns (worsening but non-emergent). Raise to urgent on their own.
TIER2_SYMPTOM_RULES = [
    ("t2_fatigue_dizzy", "increasing fatigue or dizziness", "worsening",
     ["more tired", "increasing fatigue", "very dizzy", "getting dizzy",
      "dizzier", "lightheaded"]),
    ("t2_headache_new", "new or more frequent headaches", "worsening",
     ["new headache", "more headaches", "headaches more often", "frequent headache"]),
    ("t2_side_effects", "possible medication side effects", "medication",
     ["side effect", "since starting the", "the medicine makes me", "reaction to the medicine"]),
    ("t2_missed_doses", "missed medication doses", "medication",
     ["missed a dose", "missed my dose", "missed doses", "forgot to take",
      "skipped my", "ran out of my"]),
    ("t2_new_mild", "new mild symptoms", "worsening",
     ["new symptom", "something new", "didn't have before"]),
]

# Context-raised Tier-2 patterns (need PatientContext to fire).
BLEEDING_TERMS = ["bleeding", "blood in", "black stool", "tarry", "bruising",
                  "bruise", "coughing blood", "nosebleed", "blood in my stool",
                  "blood when"]
FALL_TERMS = ["i fell", "i fall", "had a fall", "slipped and fell", "fell down"]
MISSED_DOSE_TERMS = ["missed a dose", "missed my dose", "missed doses",
                     "forgot to take", "skipped my", "ran out of my"]
GLUCOSE_TERMS = ["shaky", "very sweaty", "cold sweat", "very thirsty",
                 "blurry vision", "peeing a lot", "confused and sweaty"]

# Hypertensive-urgency thresholds (systolic/diastolic).
BP_URGENT_SYSTOLIC = 180
BP_URGENT_DIASTOLIC = 120

NEGATIONS = ["no ", "not ", "n't", "without", "denies", "deny", "any ", "haven't",
             "don't have", "do not have"]
# ===== END DRAFT CLINICAL LOGIC =====
# ----------------------------------------------------------------------------


def _contains(text: str, phrase: str) -> bool:
    return phrase in text


def _negated_near(text: str, phrase: str, window: int = 18) -> bool:
    """Light negation guard: True if a negation word appears shortly BEFORE phrase.

    Conservative — only suppresses when negation is close before the phrase, so we
    still err toward flagging when unsure (safety).
    """
    idx = text.find(phrase)
    if idx < 0:
        return False
    pre = text[max(0, idx - window):idx]
    return any(neg in pre for neg in NEGATIONS)


def _match_rules(text: str, rules) -> List[Dict[str, Any]]:
    hits = []
    for rule in rules:
        if len(rule) == 3:
            rule_id, reason, phrases = rule
            category = None
        else:
            rule_id, reason, category, phrases = rule
        for p in phrases:
            if _contains(text, p) and not _negated_near(text, p):
                hit = {"rule_id": rule_id, "reason": reason, "matched": p}
                if category:
                    hit["category"] = category
                hits.append(hit)
                break
    return hits


def _tier1(text: str) -> List[Dict[str, Any]]:
    flags = _match_rules(text, TIER1_RULES)
    for f in flags:
        f["tier"] = TIER_RED
        f["category"] = "neuro_emergency"
    return flags


def _tier2(text: str, context: Optional[Any]) -> List[Dict[str, Any]]:
    flags = _match_rules(text, TIER2_SYMPTOM_RULES)
    for f in flags:
        f["tier"] = TIER_URGENT

    # Context-raised rules require PatientContext.
    if context is not None:
        on_anticoag = bool(getattr(context, "on_anticoagulant", False))
        comorbid = list(getattr(context, "comorbidities", []) or [])
        max_sys = getattr(context, "max_systolic", None)
        max_dia = getattr(context, "max_diastolic", None)

        if on_anticoag and any(_contains(text, t) and not _negated_near(text, t)
                               for t in BLEEDING_TERMS):
            flags.append({"tier": TIER_URGENT, "rule_id": "t2_anticoag_bleeding",
                          "category": "anticoag",
                          "reason": "On an anticoagulant and reports possible bleeding/bruising."})
        if on_anticoag and any(_contains(text, t) and not _negated_near(text, t)
                               for t in FALL_TERMS):
            flags.append({"tier": TIER_URGENT, "rule_id": "t2_anticoag_fall",
                          "category": "anticoag",
                          "reason": "On an anticoagulant and reports a fall (bleeding risk)."})
        if on_anticoag and any(_contains(text, t) and not _negated_near(text, t)
                               for t in MISSED_DOSE_TERMS):
            flags.append({"tier": TIER_URGENT, "rule_id": "t2_anticoag_missed",
                          "category": "medication",
                          "reason": "On an anticoagulant and reports missed doses (recurrence risk)."})
        if (max_sys is not None and max_sys >= BP_URGENT_SYSTOLIC) or \
           (max_dia is not None and max_dia >= BP_URGENT_DIASTOLIC):
            flags.append({"tier": TIER_URGENT, "rule_id": "t2_bp_hypertensive_urgency",
                          "category": "bp",
                          "reason": f"Recent BP in hypertensive-urgency range "
                                    f"(>= {BP_URGENT_SYSTOLIC}/{BP_URGENT_DIASTOLIC})."})
        if "diabetes" in comorbid and any(_contains(text, t) and not _negated_near(text, t)
                                          for t in GLUCOSE_TERMS):
            flags.append({"tier": TIER_URGENT, "rule_id": "t2_glucose",
                          "category": "glucose",
                          "reason": "Diabetic with symptoms suggesting low/high blood sugar."})
    return flags


# ===== DRAFT CLINICAL LOGIC — REQUIRES DR. ZAND SIGN-OFF =====
# Plain-language guidance messages per tier (A.9). DRAFT wording.
GUIDANCE = {
    TIER_RED: ("This may be an emergency. Please hang up and call 911 right now. "
               "Your care team will also be alerted."),
    TIER_URGENT: ("Thank you for sharing this. Your care team should look at this "
                  "soon — the same day or the next business day. If anything gets "
                  "worse or feels like an emergency, call 911."),
    TIER_ROUTINE: ("Thank you. Nothing here needs urgent attention. Your care team "
                   "will review your check-in as part of routine follow-up."),
}
# ===== END DRAFT CLINICAL LOGIC =====


def evaluate(user_text: str, context: Optional[Any] = None,
             user_urgency: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate flags for a patient's response.

    Returns a dict with the overall tier, the list of flags (each citing the rule
    /context that fired), DRAFT guidance text, and the advisory user urgency.

    Guarantees:
    - Tier-1 fires from `user_text` alone (context/urgency cannot prevent it).
    - `user_urgency` is advisory: included in the result, never lowers the tier.
    """
    text = (user_text or "").lower()
    flags: List[Dict[str, Any]] = []

    tier1 = _tier1(text)
    flags.extend(tier1)

    # Tier-2 only matters if there is no Tier-1 emergency in play, but we still
    # record them; the overall tier is the most severe (lowest number).
    if not tier1:
        flags.extend(_tier2(text, context))

    if flags:
        overall = min(f["tier"] for f in flags)
    else:
        overall = TIER_ROUTINE
        flags.append({"tier": TIER_ROUTINE, "rule_id": "t3_routine",
                      "category": "routine", "reason": "No red or urgent flags detected."})

    return {
        "overall_tier": overall,
        "flags": flags,
        "guidance": GUIDANCE.get(overall, GUIDANCE[TIER_ROUTINE]),
        "user_reported_urgency": user_urgency,  # advisory only; never lowers tier
        "tier1_independent": True,
    }
