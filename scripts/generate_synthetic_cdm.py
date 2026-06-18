#!/usr/bin/env python3
"""
Generate a small SYNTHETIC PCORnet CDM (v6.0) dataset for AI-SoNar testing.

Writes the stroke-relevant subset of CDM tables as Parquet to data/synthetic_cdm/:
DEMOGRAPHIC, DIAGNOSIS, PRESCRIBING, VITAL, LAB_RESULT_CM — all joinable on PATID.

ALL DATA IS FICTIONAL. No real patients, no PHI. Identifiers (SYN0001..) and dates
are invented. The real CDM database cannot be shared; this stand-in lets the data
layer (B.2), flagging (B.4), and scenarios (Part C) be built and tested safely.

Run:  python scripts/generate_synthetic_cdm.py
Output is deterministic (fixed values), so tests can rely on it.
"""

import os
from datetime import date
from pathlib import Path

import pandas as pd

OUT_DIR = Path(os.environ.get(
    "CDM_DATA_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "synthetic_cdm"),
))

# ---- Reference codes (real coding systems, fictional usage) -------------------
DX_TYPE_ICD10 = "10"          # PCORnet DX_TYPE: 10 = ICD-10-CM
ICD = {
    "ischemic": "I63.9",      # Cerebral infarction, unspecified
    "hemorrhagic": "I61.9",   # Nontraumatic intracerebral hemorrhage, unspecified
    "htn": "I10",             # Essential hypertension
    "dm2": "E11.9",           # Type 2 diabetes mellitus
    "afib": "I48.91",         # Atrial fibrillation, unspecified
}
RXNORM = {  # RXNORM_CUI : RAW_RX_MED_NAME
    "warfarin": ("11289", "warfarin sodium 5 mg tablet"),
    "apixaban": ("1364430", "apixaban 5 mg tablet"),
    "rivaroxaban": ("1114195", "rivaroxaban 20 mg tablet"),
    "aspirin": ("1191", "aspirin 81 mg tablet"),
    "clopidogrel": ("32968", "clopidogrel 75 mg tablet"),
    "atorvastatin": ("83367", "atorvastatin 40 mg tablet"),
    "lisinopril": ("29046", "lisinopril 10 mg tablet"),
    "amlodipine": ("17767", "amlodipine 5 mg tablet"),
    "metformin": ("6809", "metformin 500 mg tablet"),
}
LOINC = {
    "inr": ("6301-6", ""),            # INR
    "glucose": ("2345-7", "mg/dL"),   # Glucose, serum/plasma
    "hba1c": ("4548-4", "%"),         # Hemoglobin A1c
}

# ---- Patient definitions (the brief's 6 profiles) ----------------------------
# Each: demographics, diagnoses, meds, BP readings (sys/dia), labs.
PATIENTS = [
    {
        "patid": "SYN0001",
        "desc": "Ischemic stroke, on warfarin, INR slightly high, well-controlled BP.",
        "birth_date": "1953-04-12", "sex": "M", "race": "05", "hispanic": "N",
        "dx": [("ischemic", "2026-02-10"), ("htn", "2019-06-01")],
        "meds": ["warfarin", "lisinopril", "atorvastatin"],
        "bp": [("2026-05-20", 128, 78), ("2026-05-27", 132, 80), ("2026-06-03", 126, 76)],
        "labs": [("inr", 3.6, "2026-06-05"), ("inr", 3.4, "2026-05-22")],
    },
    {
        "patid": "SYN0002",
        "desc": "Ischemic stroke, on apixaban + statin, well-controlled.",
        "birth_date": "1960-09-30", "sex": "F", "race": "03", "hispanic": "N",
        "dx": [("ischemic", "2026-01-18"), ("htn", "2021-03-15")],
        "meds": ["apixaban", "atorvastatin", "amlodipine"],
        "bp": [("2026-05-19", 124, 74), ("2026-05-26", 122, 72), ("2026-06-02", 126, 78)],
        "labs": [],
    },
    {
        "patid": "SYN0003",
        "desc": "Hemorrhagic stroke, NOT on anticoagulant, poorly controlled BP (several high readings).",
        "birth_date": "1948-12-02", "sex": "M", "race": "05", "hispanic": "Y",
        "dx": [("hemorrhagic", "2026-03-05"), ("htn", "2015-08-20")],
        "meds": ["lisinopril", "amlodipine", "atorvastatin"],
        "bp": [("2026-05-18", 188, 112), ("2026-05-25", 192, 118), ("2026-06-01", 185, 120)],
        "labs": [],
    },
    {
        "patid": "SYN0004",
        "desc": "Ischemic stroke + atrial fibrillation + diabetes, on multiple meds.",
        "birth_date": "1956-07-21", "sex": "F", "race": "06", "hispanic": "N",
        "dx": [("ischemic", "2026-02-25"), ("afib", "2024-11-10"),
               ("dm2", "2018-01-05"), ("htn", "2017-05-12")],
        "meds": ["apixaban", "metformin", "lisinopril", "atorvastatin"],
        "bp": [("2026-05-21", 138, 84), ("2026-05-28", 142, 86), ("2026-06-04", 136, 82)],
        "labs": [("hba1c", 7.8, "2026-05-15"), ("glucose", 168, "2026-06-04")],
    },
    {
        "patid": "SYN0005",
        "desc": "Ischemic stroke, on aspirin only, stable.",
        "birth_date": "1965-02-14", "sex": "M", "race": "05", "hispanic": "N",
        "dx": [("ischemic", "2026-04-01")],
        "meds": ["aspirin"],
        "bp": [("2026-05-22", 130, 80), ("2026-05-29", 128, 78), ("2026-06-05", 132, 79)],
        "labs": [],
    },
    {
        "patid": "SYN0006",
        "desc": ("Red-flag scenario patient: recent ischemic stroke on anticoagulant; "
                 "context makes NEW one-sided weakness clearly urgent."),
        "birth_date": "1951-10-08", "sex": "F", "race": "03", "hispanic": "N",
        "dx": [("ischemic", "2026-05-30"), ("afib", "2025-09-01"), ("htn", "2016-02-01")],
        "meds": ["rivaroxaban", "atorvastatin", "lisinopril"],
        "bp": [("2026-06-06", 150, 92), ("2026-06-10", 158, 96), ("2026-06-13", 154, 90)],
        "labs": [("inr", 1.1, "2026-06-08")],
    },
]


def build_frames():
    demo, diag, presc, vital, lab = [], [], [], [], []
    for p in PATIENTS:
        pid = p["patid"]
        demo.append({
            "PATID": pid, "BIRTH_DATE": p["birth_date"], "SEX": p["sex"],
            "RACE": p["race"], "HISPANIC": p["hispanic"],
        })
        for i, (dxkey, dxdate) in enumerate(p["dx"], 1):
            diag.append({
                "DIAGNOSISID": f"{pid}-DX{i:02d}", "PATID": pid,
                "DX": ICD[dxkey], "DX_TYPE": DX_TYPE_ICD10,
                "DX_DATE": dxdate, "ADMIT_DATE": dxdate,
            })
        for i, medkey in enumerate(p["meds"], 1):
            cui, name = RXNORM[medkey]
            start = p["dx"][0][1]  # start near index stroke date
            presc.append({
                "PRESCRIBINGID": f"{pid}-RX{i:02d}", "PATID": pid,
                "RXNORM_CUI": cui, "RAW_RX_MED_NAME": name,
                "RX_ORDER_DATE": start, "RX_START_DATE": start, "RX_END_DATE": "",
            })
        for i, (mdate, sys, dia) in enumerate(p["bp"], 1):
            vital.append({
                "VITALID": f"{pid}-VT{i:02d}", "PATID": pid,
                "MEASURE_DATE": mdate, "SYSTOLIC": sys, "DIASTOLIC": dia,
            })
        for i, (labkey, val, ldate) in enumerate(p["labs"], 1):
            loinc, unit = LOINC[labkey]
            lab.append({
                "LAB_RESULT_CM_ID": f"{pid}-LB{i:02d}", "PATID": pid,
                "LAB_LOINC": loinc, "RESULT_NUM": val, "RESULT_UNIT": unit,
                "RESULT_DATE": ldate, "SPECIMEN_DATE": ldate,
            })
    return {
        "DEMOGRAPHIC": pd.DataFrame(demo),
        "DIAGNOSIS": pd.DataFrame(diag),
        "PRESCRIBING": pd.DataFrame(presc),
        "VITAL": pd.DataFrame(vital),
        "LAB_RESULT_CM": pd.DataFrame(lab),
    }


README = """# Synthetic PCORnet CDM dataset (FICTIONAL — no PHI)

Generated by `scripts/generate_synthetic_cdm.py`. All identifiers and dates are
invented. This stands in for the real, unshareable CDM database so the patient-data
layer, flagging rules, and scenarios can be built and tested safely.

Tables (Parquet, joined on `PATID`): DEMOGRAPHIC, DIAGNOSIS, PRESCRIBING, VITAL,
LAB_RESULT_CM. Codes use real systems (ICD-10-CM, RxNorm, LOINC) with fictional usage.

## Patients

| PATID | Summary |
|-------|---------|
| SYN0001 | Ischemic stroke, on **warfarin**, INR slightly high (3.6), well-controlled BP. |
| SYN0002 | Ischemic stroke, on **apixaban** + statin, well-controlled. |
| SYN0003 | **Hemorrhagic** stroke, **not** on anticoagulant, **poorly controlled BP** (≈188–192/112–120). |
| SYN0004 | Ischemic stroke + **atrial fibrillation** + **diabetes**, on multiple meds (HbA1c 7.8). |
| SYN0005 | Ischemic stroke, on **aspirin only**, stable. |
| SYN0006 | **Red-flag scenario** patient: recent ischemic stroke on an anticoagulant; context makes new one-sided weakness clearly urgent. |

These double as test fixtures: e.g. SYN0001/SYN0006 exercise anticoagulant +
context-raised flags, SYN0003 exercises hypertensive-range BP flags.
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = build_frames()
    for name, df in frames.items():
        df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        print(f"wrote {name}.parquet ({len(df)} rows)")
    (OUT_DIR / "README.md").write_text(README)
    print(f"wrote README.md")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
