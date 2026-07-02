"""Unit tests for the tiered flagging engine (api/services/flagging.py).

Purpose
-------
Lock in the behavior of every DRAFT clinical rule so that:
1. Refactors cannot silently change what gets flagged.
2. Once Dr. Zand signs off on a rule, the corresponding test becomes the
   guarantee that the approved behavior is preserved.

These tests assert the engine's *documented safety guarantees*:
- Tier-1 fires from the patient's words alone (context/urgency cannot prevent it).
- User-reported urgency is advisory and never lowers a tier.
- Negation ("no chest pain") suppresses a match, but only conservatively.

NOTE: passing tests do NOT mean the rules are clinically correct — that is
Dr. Zand's sign-off (see CLINICAL_REVIEW_NEEDED.md). Tests only pin behavior.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from api.services.flagging import (
    TIER_RED,
    TIER_URGENT,
    TIER_ROUTINE,
    TIER1_RULES,
    TIER2_SYMPTOM_RULES,
    BP_URGENT_SYSTOLIC,
    BP_URGENT_DIASTOLIC,
    GUIDANCE,
    evaluate,
)


# ---------------------------------------------------------------------------
# Lightweight stand-in for api.services.patient_data.PatientContext.
# The engine only reads attributes via getattr, so any object with these
# fields works; this keeps the test import-independent of patient_data.
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    on_anticoagulant: bool = False
    comorbidities: List[str] = field(default_factory=list)
    max_systolic: Optional[int] = None
    max_diastolic: Optional[int] = None


def rule_ids(result):
    return {f["rule_id"] for f in result["flags"]}


# ---------------------------------------------------------------------------
# Tier-1 red flags — every rule, representative phrase(s)
# ---------------------------------------------------------------------------
TIER1_CASES = [
    ("t1_weakness_onesided", "my left side went numb, weakness on one side"),
    ("t1_weakness_onesided", "I can't move my arm this morning"),
    ("t1_face_droop", "my face is drooping on the right"),
    ("t1_speech", "my words are slurring and I can't get my words out"),
    ("t1_speech", "I have trouble speaking since breakfast"),
    ("t1_headache_worst", "this is the worst headache of my life"),
    ("t1_headache_worst", "a thunderclap headache hit me"),
    ("t1_vision", "I suddenly have double vision"),
    ("t1_vision", "I can't see out of my left eye"),
    ("t1_balance", "I can't walk straight and keep falling"),
    ("t1_loc", "I passed out for a minute earlier"),
    ("t1_loc", "I think I fainted in the kitchen"),
    ("t1_chest_pain", "I have chest pain when I breathe"),
    ("t1_chest_pain", "my chest is tight and heavy"),
    ("t1_fall_injury", "I fell and hit my head on the counter"),
]


@pytest.mark.parametrize("expected_rule,text", TIER1_CASES)
def test_tier1_rules_fire(expected_rule, text):
    result = evaluate(text)
    assert result["overall_tier"] == TIER_RED
    assert expected_rule in rule_ids(result)


def test_every_tier1_rule_is_covered_by_a_case():
    """If someone adds a new Tier-1 rule, force them to add a test for it."""
    tested = {rule for rule, _ in TIER1_CASES}
    defined = {r[0] for r in TIER1_RULES}
    assert defined == tested, (
        f"Tier-1 rules without a test case: {defined - tested}. "
        "Add a case to TIER1_CASES for each new rule."
    )


def test_tier1_flags_are_labeled_neuro_emergency():
    result = evaluate("my face is drooping")
    t1 = [f for f in result["flags"] if f["tier"] == TIER_RED]
    assert t1 and all(f["category"] == "neuro_emergency" for f in t1)
    assert all("matched" in f and "reason" in f for f in t1)  # auditable


# ---------------------------------------------------------------------------
# Safety guarantees
# ---------------------------------------------------------------------------
def test_tier1_fires_regardless_of_context():
    """Context must never suppress a Tier-1 emergency."""
    healthy_ctx = Ctx()  # nothing concerning in context
    result = evaluate("I can't speak properly, slurred speech", context=healthy_ctx)
    assert result["overall_tier"] == TIER_RED


def test_user_urgency_never_lowers_tier():
    """Patient saying 'it's nothing' must not downgrade an emergency."""
    result = evaluate("my face is drooping", user_urgency="not urgent")
    assert result["overall_tier"] == TIER_RED
    assert result["user_reported_urgency"] == "not urgent"


def test_user_urgency_alone_does_not_raise_tier():
    """Urgency is advisory: it is carried through but does not create flags."""
    result = evaluate("I feel fine today", user_urgency="emergency")
    assert result["overall_tier"] == TIER_ROUTINE
    assert result["user_reported_urgency"] == "emergency"


def test_tier1_takes_priority_over_tier2():
    text = "I missed a dose and now my face is drooping"
    result = evaluate(text, context=Ctx(on_anticoagulant=True))
    assert result["overall_tier"] == TIER_RED


# ---------------------------------------------------------------------------
# Negation guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "no chest pain at all",
    "I don't have chest pain",
    "without any double vision",
    "I haven't passed out",
])
def test_negated_symptoms_do_not_fire_tier1(text):
    result = evaluate(text)
    assert result["overall_tier"] == TIER_ROUTINE, (
        f"Negated phrase incorrectly flagged: {text!r} -> {rule_ids(result)}"
    )


def test_negation_is_conservative_when_far_from_phrase():
    """Negation far before the phrase must NOT suppress (err toward flagging)."""
    text = "no problems with my meds, but tonight I got sudden chest pain"
    result = evaluate(text)
    assert result["overall_tier"] == TIER_RED


# ---------------------------------------------------------------------------
# Tier-2 symptom rules (no context needed)
# ---------------------------------------------------------------------------
TIER2_CASES = [
    ("t2_fatigue_dizzy", "I've been feeling more tired and lightheaded"),
    ("t2_headache_new", "I'm getting headaches more often now"),
    ("t2_side_effects", "I think it's a side effect of the new pill"),
    ("t2_missed_doses", "I forgot to take my blood pressure pill twice"),
    ("t2_new_mild", "there's something new I didn't have before"),
]


@pytest.mark.parametrize("expected_rule,text", TIER2_CASES)
def test_tier2_symptom_rules_fire(expected_rule, text):
    result = evaluate(text)
    assert result["overall_tier"] == TIER_URGENT
    assert expected_rule in rule_ids(result)


def test_every_tier2_symptom_rule_is_covered_by_a_case():
    tested = {rule for rule, _ in TIER2_CASES}
    defined = {r[0] for r in TIER2_SYMPTOM_RULES}
    assert defined == tested, (
        f"Tier-2 rules without a test case: {defined - tested}. "
        "Add a case to TIER2_CASES for each new rule."
    )


# ---------------------------------------------------------------------------
# Context-raised Tier-2 rules
# ---------------------------------------------------------------------------
def test_anticoag_bleeding_flags_only_with_context():
    text = "I noticed bruising on my arm"
    assert evaluate(text)["overall_tier"] == TIER_ROUTINE  # no context
    result = evaluate(text, context=Ctx(on_anticoagulant=True))
    assert result["overall_tier"] == TIER_URGENT
    assert "t2_anticoag_bleeding" in rule_ids(result)


def test_anticoag_fall_flags_only_with_context():
    text = "I slipped and fell in the bathroom"
    assert evaluate(text)["overall_tier"] == TIER_ROUTINE
    result = evaluate(text, context=Ctx(on_anticoagulant=True))
    assert "t2_anticoag_fall" in rule_ids(result)
    assert result["overall_tier"] == TIER_URGENT


def test_anticoag_missed_dose_adds_anticoag_specific_flag():
    text = "I missed my dose yesterday"
    result = evaluate(text, context=Ctx(on_anticoagulant=True))
    ids = rule_ids(result)
    assert "t2_anticoag_missed" in ids
    assert "t2_missed_doses" in ids  # base rule also fires
    assert result["overall_tier"] == TIER_URGENT


def test_glucose_symptoms_require_diabetes_comorbidity():
    text = "I feel shaky and very sweaty"
    assert evaluate(text, context=Ctx())["overall_tier"] == TIER_ROUTINE
    result = evaluate(text, context=Ctx(comorbidities=["diabetes"]))
    assert "t2_glucose" in rule_ids(result)
    assert result["overall_tier"] == TIER_URGENT


# ---------------------------------------------------------------------------
# BP thresholds — exact boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sys_bp,dia_bp,should_flag", [
    (BP_URGENT_SYSTOLIC, None, True),        # 180 fires
    (BP_URGENT_SYSTOLIC - 1, None, False),   # 179 does not
    (None, BP_URGENT_DIASTOLIC, True),       # 120 fires
    (None, BP_URGENT_DIASTOLIC - 1, False),  # 119 does not
    (200, 130, True),
    (None, None, False),
])
def test_bp_hypertensive_urgency_boundaries(sys_bp, dia_bp, should_flag):
    result = evaluate("I feel okay today",
                      context=Ctx(max_systolic=sys_bp, max_diastolic=dia_bp))
    flagged = "t2_bp_hypertensive_urgency" in rule_ids(result)
    assert flagged == should_flag
    assert result["overall_tier"] == (TIER_URGENT if should_flag else TIER_ROUTINE)


# ---------------------------------------------------------------------------
# Routine default & result contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "I feel fine, thanks for checking in",
    "everything is good, took all my medications",
    "",
    None,
])
def test_benign_or_empty_input_is_routine(text):
    result = evaluate(text)
    assert result["overall_tier"] == TIER_ROUTINE
    assert rule_ids(result) == {"t3_routine"}


def test_result_contract():
    result = evaluate("my face is drooping", user_urgency="high")
    assert set(result) == {"overall_tier", "flags", "guidance",
                           "user_reported_urgency", "tier1_independent"}
    assert result["tier1_independent"] is True
    assert result["guidance"] == GUIDANCE[TIER_RED]


@pytest.mark.parametrize("text,tier", [
    ("my face is drooping", TIER_RED),
    ("I've been feeling more tired lately", TIER_URGENT),
    ("all good today", TIER_ROUTINE),
])
def test_guidance_matches_overall_tier(text, tier):
    result = evaluate(text)
    assert result["overall_tier"] == tier
    assert result["guidance"] == GUIDANCE[tier]


def test_911_language_only_in_red_and_urgent_guidance():
    assert "911" in GUIDANCE[TIER_RED]
    assert "911" in GUIDANCE[TIER_URGENT]   # "if worse, call 911"
    assert "911" not in GUIDANCE[TIER_ROUTINE]
