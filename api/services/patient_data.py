"""
Patient-data access layer (B.2) — load a patient's history from a PCORnet CDM
source and expose it as a structured PatientContext.

Reads Parquet from CDM_DATA_PATH (default: <repo>/data/synthetic_cdm). The real
CDM database is wired in later by changing CDM_DATA_PATH only — no code change.

This layer provides BACKGROUND CONTEXT for better questions and personalized plain
-language information and for the flagging rules (B.4). It is NOT a diagnostic tool
and must never be read back to the patient as their medical record verbatim.

Reads only; never writes. Missing PATID returns None (never raises). No Azure deps.
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# ICD-10 prefixes for stroke types / comorbidities.
ICD_ISCHEMIC_PREFIX = "I63"
ICD_HEMORRHAGIC_PREFIX = "I61"
ICD_HTN_PREFIXES = ("I10",)
ICD_DM_PREFIXES = ("E11", "E10")
ICD_AFIB_PREFIXES = ("I48",)

# RxNorm CUIs by class (matches the synthetic generator; extend for real data).
ANTICOAGULANT_CUIS = {"11289", "1364430", "1114195"}   # warfarin, apixaban, rivaroxaban
ANTIPLATELET_CUIS = {"1191", "32968"}                  # aspirin, clopidogrel
# Name-substring fallback when a CUI is unknown.
ANTICOAGULANT_NAMES = ("warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban")
ANTIPLATELET_NAMES = ("aspirin", "clopidogrel", "ticagrelor", "prasugrel")

LOINC_INR = "6301-6"

_CDM_TABLES = ("DEMOGRAPHIC", "DIAGNOSIS", "PRESCRIBING", "VITAL", "LAB_RESULT_CM")


def cdm_data_path() -> Path:
    default = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic_cdm"
    return Path(os.environ.get("CDM_DATA_PATH", str(default)))


@dataclass
class PatientContext:
    """Structured, non-diagnostic snapshot of a patient's stroke-relevant history."""
    patid: str
    age: Optional[int] = None
    sex: Optional[str] = None
    stroke_types: List[Dict[str, str]] = field(default_factory=list)   # [{type,date}]
    comorbidities: List[str] = field(default_factory=list)             # htn/diabetes/afib
    medications: List[Dict[str, str]] = field(default_factory=list)    # [{cui,name}]
    on_anticoagulant: bool = False
    on_antiplatelet: bool = False
    recent_bp: List[Dict[str, Any]] = field(default_factory=list)      # [{date,systolic,diastolic}]
    bp_trend: Optional[str] = None                                     # rising|falling|stable
    max_systolic: Optional[int] = None
    max_diastolic: Optional[int] = None
    latest_inr: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_background_summary(self) -> str:
        """Short, neutral background line for prompt injection (B.3).

        Plain facts only — no advice, no diagnosis. Used as model background, never
        recited to the patient verbatim.
        """
        parts = []
        if self.age is not None:
            parts.append(f"age {self.age}")
        if self.stroke_types:
            parts.append("stroke history: " + ", ".join(
                f"{s['type']}" for s in self.stroke_types))
        if self.comorbidities:
            parts.append("conditions: " + ", ".join(self.comorbidities))
        if self.on_anticoagulant:
            parts.append("on an anticoagulant (blood thinner)")
        if self.on_antiplatelet:
            parts.append("on an antiplatelet")
        if self.max_systolic is not None:
            parts.append(f"recent BP up to {self.max_systolic}/{self.max_diastolic}")
        if self.latest_inr is not None:
            parts.append(f"latest INR {self.latest_inr}")
        return "; ".join(parts)


def _calc_age(birth_date: str) -> Optional[int]:
    try:
        b = datetime.strptime(str(birth_date)[:10], "%Y-%m-%d").date()
        today = date.today()
        return today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    except Exception:
        return None


def _read_table(name: str) -> Optional[pd.DataFrame]:
    path = cdm_data_path() / f"{name}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Failed to read {path}: {e}")
        return None


def _classify_meds(rows: pd.DataFrame):
    meds, anticoag, antiplatelet = [], False, False
    for _, r in rows.iterrows():
        cui = str(r.get("RXNORM_CUI", "")).strip()
        name = str(r.get("RAW_RX_MED_NAME", "")).strip()
        meds.append({"cui": cui, "name": name})
        lname = name.lower()
        if cui in ANTICOAGULANT_CUIS or any(k in lname for k in ANTICOAGULANT_NAMES):
            anticoag = True
        if cui in ANTIPLATELET_CUIS or any(k in lname for k in ANTIPLATELET_NAMES):
            antiplatelet = True
    return meds, anticoag, antiplatelet


def _stroke_and_comorbidities(rows: pd.DataFrame):
    stroke_types, comorbid = [], set()
    for _, r in rows.iterrows():
        dx = str(r.get("DX", "")).upper().replace(".", "")
        dxdate = str(r.get("DX_DATE", ""))
        if dx.startswith(ICD_ISCHEMIC_PREFIX):
            stroke_types.append({"type": "ischemic", "date": dxdate})
        elif dx.startswith(ICD_HEMORRHAGIC_PREFIX):
            stroke_types.append({"type": "hemorrhagic", "date": dxdate})
        elif any(dx.startswith(p) for p in ICD_HTN_PREFIXES):
            comorbid.add("hypertension")
        elif any(dx.startswith(p) for p in ICD_DM_PREFIXES):
            comorbid.add("diabetes")
        elif any(dx.startswith(p) for p in ICD_AFIB_PREFIXES):
            comorbid.add("atrial_fibrillation")
    return stroke_types, sorted(comorbid)


def _bp_summary(rows: pd.DataFrame):
    if rows is None or rows.empty:
        return [], None, None, None
    rows = rows.sort_values("MEASURE_DATE")
    readings = [
        {"date": str(r["MEASURE_DATE"]), "systolic": int(r["SYSTOLIC"]),
         "diastolic": int(r["DIASTOLIC"])}
        for _, r in rows.iterrows()
    ]
    systolics = [x["systolic"] for x in readings]
    max_sys = max(systolics)
    max_dia = max(x["diastolic"] for x in readings)
    trend = "stable"
    if len(systolics) >= 2:
        delta = systolics[-1] - systolics[0]
        trend = "rising" if delta >= 10 else "falling" if delta <= -10 else "stable"
    return readings, trend, max_sys, max_dia


def load_patient_context(patid: Optional[str]) -> Optional[PatientContext]:
    """Build a PatientContext for a PATID from the CDM source.

    Returns None if patid is empty or not found. Never raises on missing data.
    """
    if not patid:
        return None
    demo = _read_table("DEMOGRAPHIC")
    if demo is None or "PATID" not in demo.columns:
        logger.warning("DEMOGRAPHIC table missing; cannot load patient context.")
        return None
    drow = demo[demo["PATID"].astype(str) == str(patid)]
    if drow.empty:
        return None  # PATID not found — graceful

    ctx = PatientContext(patid=str(patid))
    d0 = drow.iloc[0]
    ctx.age = _calc_age(d0.get("BIRTH_DATE"))
    ctx.sex = str(d0.get("SEX", "")) or None

    diag = _read_table("DIAGNOSIS")
    if diag is not None and not diag.empty:
        rows = diag[diag["PATID"].astype(str) == str(patid)]
        ctx.stroke_types, ctx.comorbidities = _stroke_and_comorbidities(rows)

    presc = _read_table("PRESCRIBING")
    if presc is not None and not presc.empty:
        rows = presc[presc["PATID"].astype(str) == str(patid)]
        ctx.medications, ctx.on_anticoagulant, ctx.on_antiplatelet = _classify_meds(rows)

    vital = _read_table("VITAL")
    if vital is not None and not vital.empty:
        rows = vital[vital["PATID"].astype(str) == str(patid)]
        ctx.recent_bp, ctx.bp_trend, ctx.max_systolic, ctx.max_diastolic = _bp_summary(rows)

    lab = _read_table("LAB_RESULT_CM")
    if lab is not None and not lab.empty:
        inr = lab[(lab["PATID"].astype(str) == str(patid)) &
                  (lab["LAB_LOINC"] == LOINC_INR)]
        if not inr.empty:
            inr = inr.sort_values("RESULT_DATE")
            try:
                ctx.latest_inr = float(inr.iloc[-1]["RESULT_NUM"])
            except Exception:
                ctx.latest_inr = None

    return ctx
