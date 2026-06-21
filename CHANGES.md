# AI-SoNar / VERA-cloud — Focus-Group Refinements: Change Log

**Branch:** `focus-group-refinements` (off tag `pre-cowork-baseline` @ `fe8a678`)
**Scope:** 22 commits · 38 files · +2,899 / −85 · `main` untouched, not merged
**Safety:** all clinical logic is **DRAFT** pending sign-off by Dr. Ramin Zand — see `CLINICAL_REVIEW_NEEDED.md` (15 pending items)

---

## 1. New files (25)

**New backend modules (`api/services/`)**

| File | Purpose |
|------|---------|
| `prompt_guidelines.py` | Single source of truth for shared system-prompt rules (human oversight, plain language, specificity) + `is_boilerplate_only()` detector. |
| `roles.py` | Survivor/caregiver/clinician role normalization + greeting/prompt framing (wording only). |
| `outcomes.py` | JSON-per-session outcome store: user urgency, reminders, clinician summary, role routing, field-usage, flag persistence. |
| `patient_data.py` | Loads `PatientContext` from the PCORnet CDM (`CDM_DATA_PATH`). |
| `flagging.py` | DRAFT tiered flagging engine (Tier-1 BE-FAST / Tier-2 context-raised / Tier-3 routine). |
| `resources.py` | Geographic + need local-resource lookup (info-only). |
| `audit.py` | Append-only access audit log (no clinical content). |

**New data / scenarios / scripts**

| File | Purpose |
|------|---------|
| `scripts/generate_synthetic_cdm.py` | Generates the fictional CDM dataset. |
| `scripts/generate_micro_scenarios.py` | Generates the 6 micro-visit scenarios. |
| `scripts/demo_synthetic_session.py` | End-to-end demo (no Azure) asserting flag tiers. |
| `data/synthetic_cdm/*.parquet` (5) + `README.md` | Fictional CDM tables + patient documentation. |
| `scenarios/micro_{routine,worsening,redflag}_{survivor,caregiver}.yml` (6) | Micro-visit scenarios (3 levels × 2 roles). |
| `config/resources.yaml` | Curated region→need resource directory (sample data). |
| `CLINICAL_REVIEW_NEEDED.md` | Dr. Zand's running review checklist. |
| `docs/TRANSLATION_ROADMAP.md` | Disabled translation scaffold + roadmap. |
| `docs/SYNTHETIC_DEMO.md` | How to run the synthetic demo. |

## 2. Modified files (12)

`api/main.py` · `api/services/azure_openai.py` · `api/services/enhanced_dialog.py` · `config/azure.yaml` · `frontend/static/app.js` · `frontend/static/index.html` · `frontend/static/styles.css` · `requirements.txt` · `scenarios/guided.yml` · `scenarios/rag_enhanced.yml` · `websocket/services/streaming_asr.py` · `.gitignore`

---

## 3. Part A — focus-group UX / wording / behavior (12 changes)

**A.1 — Human oversight visible.** Added one concrete "a member of your care team reviews this… a real person follows up" line to both scenario greetings; consolidated redundant disclaimers; created `prompt_guidelines.py` (`HUMAN_OVERSIGHT`) and wired it into both system-prompt locations (`azure_openai.py`, `enhanced_dialog.py`). *Files:* `scenarios/guided.yml`, `scenarios/rag_enhanced.yml`, `api/services/prompt_guidelines.py`, `api/services/azure_openai.py`, `api/services/enhanced_dialog.py`.

**A.2 — Response-time expectations.** New `response_expectations` config (per-deployment, not hard-coded); `build_response_time_message()` appends it to the spoken wrapup and sends a field the UI shows via `showResponseTimeConfirmation()`. *Files:* `config/azure.yaml`, `api/main.py`, `frontend/static/app.js`.

**A.3 — User self-urgency.** Routine/Soon/Urgent large buttons + a scenario prompt; stored as its own `user_reported_urgency` field via `POST /api/session/{id}/urgency`; **never suppresses** automatic red-flag routing. *Files:* `api/services/outcomes.py`, `api/main.py`, `scenarios/*.yml`, `frontend/static/index.html`, `frontend/static/app.js`.

**A.4 — Specific, not boilerplate.** `SPECIFICITY` prompt rule (hedge once, then concrete; "general information, not advice"); `is_boilerplate_only()` flags content-free hedges; KB stroke-type content gap logged for clinical authoring. *Files:* `api/services/prompt_guidelines.py`, `api/services/enhanced_dialog.py`, `CLINICAL_REVIEW_NEEDED.md`.

**A.5 — Bursty transcript fixed.** Interim ASR now updates one debounced bubble in place and finalizes on the final transcript (was appending fragments); switched message text to `textContent`. *Files:* `frontend/static/app.js`.

**A.6 — Role tracks.** Survivor/caregiver/clinician selector; `roles.py` framing applied to greeting + system prompt; role persisted on session and in the summary. **Clinical content identical across tracks.** *Files:* `api/services/roles.py`, `api/services/enhanced_dialog.py`, `api/main.py`, `config/azure.yaml`, `frontend/static/index.html`, `frontend/static/app.js`.

**A.7 — Voice-first + accessibility.** Raised ASR silence tolerance (config + `ASR_*_SILENCE_MS` env); large push-to-talk, 60px tap targets, high-contrast focus, reduced-motion, typing marked optional; minimal voice-driven reminder (`POST /api/session/{id}/reminder`). *Files:* `websocket/services/streaming_asr.py`, `config/azure.yaml`, `frontend/static/styles.css`, `frontend/static/index.html`, `frontend/static/app.js`, `api/services/outcomes.py`, `api/main.py`.

**A.8 — Geographic resources.** `resources.py` + `config/resources.yaml` county/region + need lookup; `GET /api/resources` and `/api/resource-regions`; opt-in, info-only, no insurance-coverage promise. *Files:* `api/services/resources.py`, `config/resources.yaml`, `api/main.py`.

**A.9 — Plain language.** Rewrote patient-facing scenario strings to short, one-idea sentences (overall Flesch-Kincaid 5.3/6.6 → ~3.6–3.7); `PLAIN_LANGUAGE` prompt rule; simplified wrapup + emergency disclaimer (DRAFT). *Files:* `scenarios/guided.yml`, `scenarios/rag_enhanced.yml`, `api/services/prompt_guidelines.py`.

**A.10 — Translation scaffold (disabled).** `translation.enabled=false` config (env `TRANSLATION_ENABLED`), no code path; `docs/TRANSLATION_ROADMAP.md` requiring professional translation + clinical review. *Files:* `config/azure.yaml`, `docs/TRANSLATION_ROADMAP.md`.

**A.11 — Clinician summary.** `build_clinician_summary()` leads with flags + urgency, separates routine items (collapsible); DRAFT `suggest_route()` (Tier-1→physician/emergency; Tier-2 med→nurse/pharmacist; rehab→therapist; else navigator); `record_field_action()` workflow-fit metric; `GET /api/session/{id}/clinician-summary`, `POST …/field-action`. *Files:* `api/services/outcomes.py`, `api/main.py`.

**A.12 — Naming clarity.** Spoken self-id changed to "automated voice assistant… not a person on your care team" (drops human-navigator implication); visible AI-identity banner + standing 911/who-to-call notice on the setup screen. *Files:* `scenarios/guided.yml`, `scenarios/rag_enhanced.yml`, `frontend/static/index.html`.

---

## 4. Part B — PCORnet CDM patient-data layer

**B.1 — Synthetic CDM.** `generate_synthetic_cdm.py` writes 5 CDM v6 tables (DEMOGRAPHIC, DIAGNOSIS, PRESCRIBING, VITAL, LAB_RESULT_CM) joined on `PATID` for 6 fictional patients (warfarin/INR, apixaban, hemorrhagic+high BP, AFib+diabetes, aspirin-only, red-flag). Real code systems (ICD-10/RxNorm/LOINC), no PHI. *Files:* `scripts/generate_synthetic_cdm.py`, `data/synthetic_cdm/*`.

**B.2 — Data-access layer.** `load_patient_context(PATID)` → `PatientContext` (age, stroke type(s), comorbidities, meds + anticoag/antiplatelet flags, BP trend/max, latest INR). `CDM_DATA_PATH` swaps to the real source with no code change; missing PATID → `None`. *Files:* `api/services/patient_data.py`, `requirements.txt` (pandas, pyarrow).

**B.4 — DRAFT flagging.** `flagging.py` — data-driven Tier-1 (BE-FAST + chest pain / LOC / fall-injury), Tier-2 (symptom + context-raised: anticoagulant+bleed/fall/missed-dose, BP ≥180/120, diabetic glucose), Tier-3 routine. Each flag cites the rule that fired. **Tier-1 fires from the patient's words alone**, independent of context and user urgency; user urgency never lowers a tier; light negation guard. Entire module DRAFT-marked + logged. *Files:* `api/services/flagging.py`, `CLINICAL_REVIEW_NEEDED.md`.

**B.3 — Context injection.** `PatientContext` passed into the dialog as **background only** (ask better questions / personalize; never diagnose or recite the record verbatim); `evaluate_flags()` uses context; **generic mode** (no PATID) has no history-dependent flags but Tier-1 still works. *Files:* `api/services/enhanced_dialog.py`, `api/main.py`.

**B.5 — Audit line.** `audit.py` writes one access-event record per history-load attempt (timestamp, PATID, role, consent, context_loaded, note) with **no clinical content**; `audit/` gitignored. *Files:* `api/services/audit.py`, `api/main.py`, `.gitignore`.

---

## 5. Part C — session initiation & scenarios

**C.1 — Session initiation.** `POST /session/start` (`SessionStartRequest`: patient_name, role, patient_id?, caregiver_consent) with history-loading + consent gating (no PATID → generic; caregiver without consent → generic + audit note; survivor/clinician/caregiver-with-consent → load). Start-screen adds Patient ID field + caregiver-consent checkbox (shown only for caregiver). *Files:* `api/main.py`, `frontend/static/index.html`, `frontend/static/app.js`.

**C.2 — Six micro-visit scenarios.** `generate_micro_scenarios.py` emits `micro_{routine,worsening,redflag}_{survivor,caregiver}.yml` — plain language, human oversight, consent, level-specific questions, urgency self-flag; red-flag asks BE-FAST (→Tier-1), worsening asks fatigue/missed-dose/side-effects (→Tier-2 context), routine→routine. Escalation content DRAFT-marked. *Files:* `scripts/generate_micro_scenarios.py`, `scenarios/micro_*.yml`.

**C.3 — Wire together + demo.** Dialog accumulates flags per response (`self.flags`, `overall_tier()`), includes them in the summary, and persists them at completion via `record_session_flags()` so the clinician summary reflects them. `demo_synthetic_session.py` runs all 6 scenarios × synthetic patients and asserts tiers. *Files:* `api/services/enhanced_dialog.py`, `api/services/outcomes.py`, `api/main.py`, `scripts/demo_synthetic_session.py`.

---

## 6. New API endpoints

| Method & path | Purpose |
|---------------|---------|
| `POST /session/start` | Canonical session initiation (role + PATID + consent gating). |
| `POST /api/session/{id}/urgency` | Record user-reported urgency (A.3). |
| `POST /api/session/{id}/reminder` | Save a voice reminder (A.7). |
| `GET /api/session/{id}/clinician-summary` | Prioritized clinician summary (A.11). |
| `POST /api/session/{id}/field-action` | Field-usage metric (A.11). |
| `GET /api/resources` · `GET /api/resource-regions` | Local resource lookup (A.8). |

---

## 7. Verification

Each commit carries a `verified:` note. Final sweep: all 8 scenarios + 2 configs parse; all Python `py_compile`s; `api.main` imports with all routes registered (Azure/OpenAI SDKs stubbed — not available in the build sandbox); `node --check` passes; unit tests pass for `PatientContext`, flagging (incl. Tier-1 independence + negation), outcomes/routing, resources; the **end-to-end synthetic demo passes all 6 cases** plus the Tier-1-in-generic invariant. A live app boot was **not** possible (needs Azure + Redis) and was substituted with import + unit/e2e tests.

## 8. Not completed / deferred

- **Live voice end-to-end** — requires your Azure environment.
- **A.4 stroke-type KB content** — needs clinical authoring before specificity is fully realized (logged).
- **A.11 clinician dashboard UI** — backend summary/routing exist; no clinician-facing screen (current app is patient-facing).
- **Detector edge cases** (e.g., some missed-dose phrasings) are intentionally conservative DRAFT rules for Dr. Zand to refine.

## 9. Commits (newest first)

```
646cfdc docs: synthetic demo run guide
b6b0d72 Part C.3: wire flags through dialog + end-to-end synthetic demo
a058725 Part C.2: six micro-visit scenarios (3 levels x 2 roles)
0641be8 Part C.1: session initiation endpoint + start-screen fields
364fb3d Part B.5: access audit line on patient-history load
a5962df Part B.3: inject PatientContext into dialog + flagging (background only)
1e2896f Part B.4: DRAFT tiered flagging engine (BE-FAST Tier-1 + context-raised Tier-2)
2091a63 Part B.2: patient-data access layer (PatientContext from CDM)
6ebe92b Part B.1: synthetic PCORnet CDM dataset (fictional) + generator
d37c1e7 Part A.7: voice-first + low-dexterity accessibility
b557a10 Part A.5: fix bursty transcript — single in-place bubble + debounced interim
1de17ea Part A.11: prioritized clinician summary + role routing + field-usage metric
b86c616 Part A.8: geographic local-resource layer (opt-in, info-only)
1cc16c7 Part A.10: translation scaffold only (disabled) + roadmap doc
977ab1b Part A.4: enforce specific-not-boilerplate responses + boilerplate detector
21af58b Part A.6: survivor/caregiver/clinician tracks (framing only)
cedb42f Part A.3: user self-reported urgency as a separate, advisory field
d36ad2c Part A.2: configurable response-time expectations (spoken + on-screen)
cc8f4f8 Part A.12: resolve 'stroke navigator' naming + add standing who-to-call notice
2ef2eec Part A.9: plain-language rewrite of patient-facing scenario text + prompt enforcement
795fcfd Part A.1: make human oversight visible in intro + system prompts
b074b5c Scaffolding: CLINICAL_REVIEW_NEEDED.md + DRAFT-marker convention
```
