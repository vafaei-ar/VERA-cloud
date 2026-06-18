#!/usr/bin/env python3
"""
End-to-end demo (C.3) on SYNTHETIC data — no Azure required.

For each micro-visit scenario (3 levels x 2 roles) this:
  1. starts a dialog with a synthetic patient's PatientContext (B.2/B.3),
  2. feeds representative patient responses and accumulates flags (B.4/C.3),
  3. persists flags and builds the prioritized clinician summary (A.11),
  4. asserts the tier behaves per Part B.

NOTE: the live Azure conversational layer (speech, GPT, search) is NOT exercised
here — this sandbox has no Azure. The demo drives the full safety pipeline that
does NOT depend on Azure: CDM -> PatientContext -> flagging -> outcomes -> summary.
Run inside the repo:  python scripts/generate_synthetic_cdm.py && python scripts/demo_synthetic_session.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Keep demo outcomes out of the real outcomes dir.
os.environ.setdefault("OUTCOMES_PATH", tempfile.mkdtemp())

from api.services.enhanced_dialog import EnhancedDialog
from api.services.patient_data import load_patient_context
from api.services import outcomes

# scenario file, patient, simulated responses, expected overall tier
CASES = [
    ("scenarios/micro_routine_survivor.yml", "SYN0005", [
        "I have been feeling pretty good", "yes I am taking my medicines",
        "I am doing okay with walking and bathing", "no questions today"], 3),
    ("scenarios/micro_routine_caregiver.yml", "SYN0002", [
        "she has been doing well", "yes she takes them every day",
        "she manages daily tasks fine", "no questions"], 3),
    ("scenarios/micro_worsening_survivor.yml", "SYN0001", [   # on warfarin
        "I have felt more tired than usual", "I missed a couple doses of my medicine",
        "I noticed some unusual bruising", "nothing else new"], 2),
    ("scenarios/micro_worsening_caregiver.yml", "SYN0003", [  # poorly controlled BP
        "he seems a little more dizzy", "no missed doses",
        "no side effects", "nothing new really"], 2),
    ("scenarios/micro_redflag_survivor.yml", "SYN0006", [     # red-flag patient
        "yes my left side suddenly went weak", "yes I have trouble speaking",
        "no headache"], 1),
    ("scenarios/micro_redflag_caregiver.yml", "SYN0006", [
        "yes her face is drooping on one side", "no",
        "no other symptoms"], 1),
]


def run_case(scenario, patid, responses, expected_tier, session_id):
    ctx = load_patient_context(patid)
    dialog = EnhancedDialog(None, None, None, str(REPO / scenario),
                            patient_name="Demo", role="survivor", patient_context=ctx)
    # Drive the flag pipeline exactly as process_user_response does (minus Azure gen).
    for r in responses:
        dialog._accumulate_flags(r)
    outcomes.record_session_flags(session_id, dialog.flags)
    summary = outcomes.build_clinician_summary(outcomes.load_outcome(session_id))
    tier = dialog.overall_tier()
    route = summary["suggested_route"]
    ok = (tier == expected_tier)
    print(f"[{'PASS' if ok else 'FAIL'}] {Path(scenario).name:32s} patid={patid} "
          f"context={'yes' if ctx else 'no':3s} tier={tier} (exp {expected_tier}) "
          f"route={route['role']}/{route['priority']}")
    if dialog.flags:
        for f in dialog.flags:
            print(f"         - T{f.get('tier')} {f.get('rule_id')}: {f.get('reason')}")
    return ok


def main():
    print("=== AI-SoNar synthetic end-to-end demo (no Azure) ===\n")
    all_ok = True
    for i, (scn, patid, resp, exp) in enumerate(CASES, 1):
        all_ok &= run_case(scn, patid, resp, exp, session_id=f"demo-{i}")
        print()
    # Extra invariant: red-flag fires even with NO patient context and user says routine.
    d = EnhancedDialog(None, None, None, str(REPO / "scenarios/micro_redflag_survivor.yml"),
                       patient_context=None)
    d._accumulate_flags("sudden weakness on one side")
    inv_ok = d.overall_tier() == 1
    print(f"[{'PASS' if inv_ok else 'FAIL'}] invariant: Tier-1 fires in generic mode (no context)")
    all_ok &= inv_ok
    print("\n" + ("ALL CASES PASSED" if all_ok else "SOME CASES FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
