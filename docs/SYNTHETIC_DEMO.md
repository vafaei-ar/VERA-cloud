# Running the synthetic demo (no Azure needed)

This exercises the safety pipeline added in the focus-group work — synthetic CDM →
`PatientContext` → tiered flagging → clinician summary — entirely on fictional data.
It does **not** require Azure, Redis, or any secrets.

```bash
# from the repo root
pip install pandas pyarrow pyyaml        # if not already installed

# 1) generate the fictional CDM dataset (data/synthetic_cdm/)
python scripts/generate_synthetic_cdm.py

# 2) generate the 6 micro-visit scenarios (scenarios/micro_*.yml)
python scripts/generate_micro_scenarios.py

# 3) run the end-to-end demo (asserts flag tiers per scenario/patient)
python scripts/demo_synthetic_session.py
```

Expected: routine cases → Tier 3, worsening → Tier 2 (e.g. warfarin + bleeding,
hypertensive-range BP), red-flag → Tier 1, plus the invariant that Tier-1 fires
even with no patient context.

## Running the full app (requires Azure)

The web app additionally needs Azure OpenAI / Speech / Search and Redis configured
(see `env.example` and `config/azure.yaml`). Those services are not available in the
build sandbox, so the live voice flow was verified by module import + unit/e2e tests,
not a live boot. To run locally with Azure configured:

```bash
uvicorn api.main:app --port 8000
# open http://localhost:8000
```

To point the data layer at the real CDM instead of the synthetic set, set
`CDM_DATA_PATH` — no code change required.

## Safety reminder

All clinical logic in this branch is **DRAFT** pending sign-off by Dr. Ramin Zand.
See `CLINICAL_REVIEW_NEEDED.md`.
