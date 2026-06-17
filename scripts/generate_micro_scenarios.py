#!/usr/bin/env python3
"""
Generate the 6 micro-visit scenario files (C.2): 3 clinical levels x 2 roles.

Levels:  routine | worsening | redflag
Roles:   survivor | caregiver   (clinician track reuses these with concise framing
                                  applied at runtime by api/services/roles.py)

Each scenario uses plain language (A.9), makes human oversight visible (A.1),
includes the urgency self-flag (A.3), and (via runtime append) response-time
expectations (A.2). Escalation content is DRAFT-marked (requires Dr. Zand sign-off).

Output: scenarios/micro_<level>_<role>.yml   Run: python scripts/generate_micro_scenarios.py
"""

import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "scenarios"

LEVELS = ("routine", "worsening", "redflag")
ROLES = ("survivor", "caregiver")

LEVEL_DESC = {
    "routine": "Routine check-in (general well-being, medications, activities).",
    "worsening": "Worsening but non-emergent check-in (exercises Tier-2 context flags).",
    "redflag": "Red-flag / emergent check-in (must trigger Tier-1 emergency guidance).",
}

# Plain-language greeting per role (AI identity A.12 + human oversight A.1).
GREETING = {
    "survivor": (
        "Good {timeofday} {honorific} {patient_name}. I'm an automated voice assistant "
        "from {organization}. I am a computer program, not a person on your care team. "
        "A member of your care team reviews what you share. If something looks urgent, a "
        "real person follows up. This is a short check-in about how you are doing. "
        "With your permission, this call will be recorded to help document your care."
    ),
    "caregiver": (
        "Good {timeofday} {honorific} {patient_name}. I'm an automated voice assistant "
        "from {organization}. I am a computer program, not a person on the care team. "
        "I understand you are helping care for your family member. A member of the care "
        "team reviews what you share. If something looks urgent, a real person follows up. "
        "This is a short check-in about how your family member is doing. With your "
        "permission, this call will be recorded to help document their care."
    ),
}

# Flow questions per level and role. Plain language; one idea per question.
QUESTIONS = {
    "routine": {
        "survivor": [
            ("general_feeling", "How have you been feeling since you left the hospital?"),
            ("meds_routine", "Are you taking your medicines the way they were prescribed?"),
            ("activities", "How are you doing with daily tasks like walking, bathing, and cooking?"),
            ("questions_for_team", "Do you have any questions you want me to pass to your care team?"),
        ],
        "caregiver": [
            ("general_feeling", "How has your family member been feeling since leaving the hospital?"),
            ("meds_routine", "Is your family member taking their medicines the way they were prescribed?"),
            ("activities", "How is your family member doing with daily tasks like walking, bathing, and cooking?"),
            ("questions_for_team", "Do you have any questions you want me to pass to the care team?"),
        ],
    },
    "worsening": {
        "survivor": [
            ("fatigue_dizzy", "Have you felt more tired or more dizzy than usual?"),
            ("missed_doses", "Have you missed any doses of your medicines?"),
            ("side_effects", "Have your medicines caused any side effects, like bruising or an upset stomach?"),
            ("new_mild", "Have you had any new symptoms that feel mild but are new for you?"),
        ],
        "caregiver": [
            ("fatigue_dizzy", "Has your family member seemed more tired or more dizzy than usual?"),
            ("missed_doses", "Has your family member missed any doses of their medicines?"),
            ("side_effects", "Have their medicines caused any side effects, like bruising or an upset stomach?"),
            ("new_mild", "Has your family member had any new symptoms that seem mild but are new?"),
        ],
    },
    "redflag": {
        "survivor": [
            ("be_fast_weak", "Right now, do you have sudden weakness or numbness on one side of your body?"),
            ("be_fast_speech", "Are you having any sudden trouble speaking or understanding speech?"),
            ("be_fast_other", "Have you had a sudden bad headache, sudden vision loss, or a sudden loss of balance?"),
        ],
        "caregiver": [
            ("be_fast_weak", "Right now, does your family member have sudden weakness or numbness on one side of their body?"),
            ("be_fast_speech", "Is your family member having sudden trouble speaking or understanding speech?"),
            ("be_fast_other", "Has your family member had a sudden bad headache, sudden vision loss, or a sudden loss of balance?"),
        ],
    },
}

URGENCY_PROMPT = "How urgent does this feel to you today? You can say routine, soon, or urgent."

WRAPUP = {
    "survivor": (
        "Thank you for talking with me today. I wrote down what you told me. A member of "
        "your care team will look at it and follow up if you need it. Remember: if you have "
        "new stroke signs, call 911 right away. These signs include sudden weakness, trouble "
        "talking, vision changes, a very bad headache, or loss of balance. Take care, and "
        "thank you again."
    ),
    "caregiver": (
        "Thank you for talking with me today. I wrote down what you told me. A member of the "
        "care team will look at it and follow up if needed. Remember: if your family member "
        "has new stroke signs, call 911 right away. These signs include sudden weakness, "
        "trouble talking, vision changes, a very bad headache, or loss of balance. Take care, "
        "and thank you again."
    ),
}

EMERGENCY_DISCLAIMER = (
    "Are you (or your family member) having an emergency or new stroke signs right now? "
    "Hang up and call 911 right away."
)


def yaml_block(text, indent="    "):
    """Render a folded scalar block for a long string."""
    return ">\n" + indent + text


def build_scenario(level, role):
    org = "PennState Health"
    site = "Hershey Medical Center"
    name = f"AI stroke navigator - {level} ({role})"
    lines = []
    lines.append("# Auto-generated micro-visit scenario (C.2). Regenerate with")
    lines.append("# scripts/generate_micro_scenarios.py")
    lines.append("# DRAFT CLINICAL TEXT — REQUIRES DR. ZAND SIGN-OFF — see CLINICAL_REVIEW_NEEDED.md")
    lines.append("#   (greeting human-oversight line, red-flag prompts, wrapup 911 guidance,")
    lines.append("#    and emergency_disclaimer are all DRAFT)")
    lines.append("meta:")
    lines.append(f'  organization: "{org}"')
    lines.append(f'  service_name: "{name}"')
    lines.append(f'  site: "{site}"')
    lines.append('  version: "1.0"')
    lines.append(f'  description: "{LEVEL_DESC[level]}"')
    lines.append('  mode: "guided"')
    lines.append(f'  level: "{level}"')
    lines.append(f'  role: "{role}"')
    lines.append("")
    lines.append("greeting:")
    lines.append("  template: " + yaml_block(GREETING[role], indent="    "))
    lines.append("  variables:")
    for v in ("timeofday", "honorific", "patient_name", "organization", "site"):
        lines.append(f"    - {v}")
    lines.append("")
    lines.append("flow:")
    # consent first
    lines.append("  - key: consent")
    lines.append("    type: confirm")
    lines.append('    prompt: "Is it okay to record this check-in so we can document your care?"')
    lines.append("    on_deny: " + yaml_block(
        "That's okay. We won't record today. If this is an emergency, call 911. "
        "You can call your care team any time. Take care.", indent="      "))
    for key, prompt in QUESTIONS[level][role]:
        lines.append(f"  - key: {key}")
        lines.append("    type: free")
        lines.append(f'    prompt: "{prompt}"')
    # urgency self-report (A.3)
    lines.append("  - key: urgency_self_report")
    lines.append("    type: free")
    lines.append("    # A.3 — advisory only; never changes automatic red-flag routing")
    lines.append(f'    prompt: "{URGENCY_PROMPT}"')
    lines.append("")
    lines.append("wrapup:")
    lines.append("  message: " + yaml_block(WRAPUP[role], indent="    "))
    lines.append("")
    lines.append("emergency_disclaimer: " + yaml_block(EMERGENCY_DISCLAIMER, indent="  "))
    lines.append("")
    lines.append("stroke_warning_signs:")
    for s in (
        "Sudden weakness or numbness in the face, arm, or leg, especially on one side",
        "Sudden trouble speaking or understanding speech",
        "Sudden trouble seeing in one or both eyes",
        "Sudden very bad headache with no known cause",
        "Sudden trouble walking, dizziness, or loss of balance",
    ):
        lines.append(f'  - "{s}"')
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for level in LEVELS:
        for role in ROLES:
            fn = OUT_DIR / f"micro_{level}_{role}.yml"
            fn.write_text(build_scenario(level, role))
            written.append(fn.name)
    print("wrote:", ", ".join(written))


if __name__ == "__main__":
    main()
