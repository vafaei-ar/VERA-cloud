"""Versioned, validated clinical review surface; snapshot once per session."""
import hashlib
import json
import os
from pathlib import Path

import yaml


def load_policy(path=None):
    path = path or os.getenv("CLINICAL_POLICY_PATH") or Path(__file__).resolve().parents[2] / "config/clinical_policy.yaml"
    with open(path) as source:
        policy = yaml.safe_load(source)
    if not isinstance(policy, dict) or not policy.get("version"):
        raise ValueError("Clinical policy requires a version")
    if policy.get("status") not in {"draft", "approved"}:
        raise ValueError("Clinical policy status must be draft or approved")
    for key in ("introduction", "consent", "declined", "emergency", "urgent", "routine", "callback", "symptom_timing", "symptom_impact", "caregiver_support"):
        if not isinstance(policy.get("messages", {}).get(key), str) or not policy["messages"][key].strip():
            raise ValueError(f"Missing clinical message: {key}")
    for key in ("ischemic", "hemorrhagic", "unknown"):
        if not policy.get("education", {}).get(key):
            raise ValueError(f"Missing stroke education: {key}")
    if not isinstance(policy["messages"].get("urgency_choices"), str) or not policy["messages"]["urgency_choices"].strip():
        raise ValueError("Missing urgency choice explanation")
    if not 0.5 <= policy["speech"]["rate"] <= 1.5 or not 3 <= policy["speech"]["silence_seconds"] <= 30:
        raise ValueError("Speech settings outside supported range")
    if not 0 <= policy["symptom_followups"]["max_questions"] <= 2:
        raise ValueError("Symptom follow-ups must be bounded to 0–2")
    for pathway in policy["symptom_followups"].get("pathways", {}).values():
        for key in ("triggers", "immediate_terms"):
            if not isinstance(pathway.get(key), list) or not all(isinstance(value, str) and value for value in pathway[key]):
                raise ValueError("Pathway terms must be nonempty strings")
        for key in ("timing_prompt", "impact_prompt"):
            if not pathway.get(key):
                raise ValueError("Pathway prompts are required")
        if pathway.get("new_or_worse_tier") not in (1, 2) or pathway.get("unknown_tier") != 2:
            raise ValueError("Pathway escalation must be Tier 1/2; unknown stays visible for review")
    days = policy.get("response_windows", {}).get("routine_business_days")
    if days is not None and (type(days) is not int or not 1 <= days <= 14):
        raise ValueError("Routine response target must be null or 1–14 business days")
    if not all(policy.get("routing", {}).get(k) for k in ("emergency", "urgent", "routine", "medication", "rehab")):
        raise ValueError("Incomplete clinical routing")
    flagging = policy.get("flagging", {})
    if not 100 <= flagging.get("bp_urgent_systolic", 180) <= 250 or not 60 <= flagging.get("bp_urgent_diastolic", 120) <= 180:
        raise ValueError("Unsupported BP thresholds")
    for rule, phrases in flagging.get("phrase_overrides", {}).items():
        if not rule.startswith(("t1_", "t2_")) or not isinstance(phrases, list) or not phrases or any(not isinstance(p, str) or not p.strip() for p in phrases):
            raise ValueError("Phrase overrides require named rules and nonempty phrase lists")
    engine_source = b"".join(Path(__file__).with_name(name).read_bytes()
                             for name in ("flagging.py", "enhanced_dialog.py"))
    policy["engine_digest"] = hashlib.sha256(engine_source).hexdigest()
    policy["digest"] = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    return policy


def response_message(policy, priority=False):
    message = policy["messages"]["urgent" if priority else "routine"]
    days = policy.get("response_windows", {}).get("routine_business_days")
    if not priority and days is not None:
        message += f" The service target for routine replies is {days} business days; a reply within that time is not guaranteed."
    return message
