# CLINICAL_REVIEW_NEEDED — Dr. Ramin Zand sign-off checklist

This file lists **every clinical rule, threshold, escalation message, red-flag definition, and
response-time promise** introduced by the Cowork focus-group refinement work. **None of it is
clinically validated.** All items are DRAFT and require sign-off by **Dr. Ramin Zand (clinical lead)**
before any clinical use.

How to read this file: each entry names the **file**, the **rule/text**, a one-line **rationale**,
its **source**, and a **status**. Status is `PENDING ZAND REVIEW` until cleared.

In code, every draft clinical item is wrapped in:

```
# ===== DRAFT CLINICAL LOGIC — REQUIRES DR. ZAND SIGN-OFF =====
# Rationale: <one line>
# Source: focus group / standard stroke guidance (BE-FAST etc.)
# DO NOT treat as clinically validated.
...
# ===== END DRAFT CLINICAL LOGIC =====
```

For YAML/Markdown patient-facing text, the equivalent marker comment is:

```
# DRAFT CLINICAL TEXT — REQUIRES DR. ZAND SIGN-OFF — see CLINICAL_REVIEW_NEEDED.md
```

---

## Review log

_(Entries appended as work proceeds. Newest at the bottom of each Part.)_

### Part A

**A.1 — Human oversight wording**

- `scenarios/guided.yml` (greeting): line *"A member of your care team reviews what you share. If something looks urgent, a real person follows up."* — Rationale: focus group (Phil, Margie) wanted visible human oversight stated once. Source: focus group June 2026. **PENDING ZAND REVIEW** — confirm this accurately describes the real review/follow-up pathway.
- `scenarios/rag_enhanced.yml` (greeting): same line as above. **PENDING ZAND REVIEW.**
- `api/services/prompt_guidelines.py` (`HUMAN_OVERSIGHT`): instructs the model to point worried patients to the human-review pathway and to advise calling care team or 911 for anything urgent. Rationale: avoid over-disclaiming; route worry to humans. Source: focus group + BE-FAST escalation norms. **PENDING ZAND REVIEW** — confirm the "call care team or 911" framing and that pointing to human review is appropriate.

**A.9 — Plain-language rewrites of escalation text**

- `scenarios/guided.yml` and `scenarios/rag_enhanced.yml` (`wrapup`): simplified 911 guidance and the stroke-sign list ("sudden weakness, trouble talking, vision changes, a very bad headache, or loss of balance"). Rationale: plain language for low-literacy / aphasia patients. Source: focus group (Lisa) + BE-FAST. **PENDING ZAND REVIEW** — confirm the simplified stroke-sign wording is clinically adequate and not under-inclusive.
- `scenarios/guided.yml` and `scenarios/rag_enhanced.yml` (`emergency_disclaimer`): simplified to *"Are you having an emergency or new stroke signs right now? Hang up and call 911 right away."* **PENDING ZAND REVIEW.**
- Note: `api/services/prompt_guidelines.py` `PLAIN_LANGUAGE` is a wording/pacing constraint (not itself clinical), but it governs how generated clinical-adjacent responses are phrased.

**A.12 — AI identity + standing who-to-call instruction**

- `scenarios/guided.yml` and `scenarios/rag_enhanced.yml` (greeting): spoken self-identification changed to *"I'm an automated voice assistant ... a computer program, not a person on your care team"* (removes the human "stroke navigator" role implication per Margie). Identity wording — confirm it is acceptable phrasing. **PENDING ZAND REVIEW.**
- `frontend/static/index.html` (setup screen banner): standing instruction *"If you have new stroke signs or an emergency, call 911 right away. For urgent questions, call your care team at the number they gave you."* Escalation/contact guidance. **PENDING ZAND REVIEW** — confirm wording and whether a specific care-team number/route should be shown.

**A.2 — Response-time expectations**

- `config/azure.yaml` (`response_expectations`): default `routine_response_business_days: 2`, `monitored_real_time: false`, and urgent instructions text. Built into the end-of-call message by `build_response_time_message()` in `api/main.py` and shown on-screen by `showResponseTimeConfirmation()` in `app.js`. Rationale: Laura feared an unmonitored inbox; set expectations and route urgent issues to a human/911. Source: focus group June 2026. **PENDING ZAND REVIEW** — confirm the 2-business-day default (per site), the "not watched in real time" statement, and the urgent/911 wording.

**A.3 — User self-reported urgency (design invariant to confirm)**

- `api/services/outcomes.py` + `api/main.py` (`POST /api/session/{id}/urgency`): patient urgency (`routine`/`soon`/`urgent`) is stored as its OWN field (`user_reported_urgency`), separate from model/rule `flags`. **Design invariant:** user urgency is advisory and must NEVER suppress an automatic Tier-1 red-flag escalation (enforced when Part B.4 flagging lands). Not a clinical threshold itself, but **PENDING ZAND CONFIRMATION** that this separation/independence is the intended behavior.

**A.6 — Role tracks (invariant to confirm)**

- `api/services/roles.py`: survivor/caregiver/clinician change only *framing/wording* (greeting + system-prompt addressing). **Invariant:** clinical content, questions, and thresholds are identical across all three tracks. Not clinical itself, but **PENDING ZAND CONFIRMATION** that role-based framing is acceptable and that no clinical content should differ by track.

**A.4 — Specific-not-boilerplate + knowledge-base content gap**

- `api/services/prompt_guidelines.py` (`SPECIFICITY`): instructs the model to add concrete, stroke-type-relevant information rather than only a variability hedge, framed as "general information, not personal medical advice." **PENDING ZAND REVIEW** — confirm the "general information, not advice" framing and that encouraging specificity is safe.
- **CONTENT GAP (needs clinical authoring):** the RAG knowledge base (`stroke-care-knowledge` index) is not known to be tagged by stroke type / affected region, so the system cannot yet reliably retrieve stroke-type-specific content (it is wired to use `PatientContext` stroke type once Part B lands). **PENDING ZAND / care team** — stroke-type-specific educational content must be clinically authored and indexed before A.4's specificity can be fully realized.
- `is_boilerplate_only()` detector flags content-free hedges for QA only (does not block or alter clinical routing).

### Part B

<!-- entries added per task -->

### Part C

<!-- entries added per task -->

---

## Summary table

| # | File | Item | Rationale | Source | Status |
|---|------|------|-----------|--------|--------|
| | | | | | |
