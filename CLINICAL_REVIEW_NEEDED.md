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

### Part B

<!-- entries added per task -->

### Part C

<!-- entries added per task -->

---

## Summary table

| # | File | Item | Rationale | Source | Status |
|---|------|------|-----------|--------|--------|
| | | | | | |
