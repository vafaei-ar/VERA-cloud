# Translation / Limited-English Support — Roadmap (SCAFFOLD ONLY)

**Status: NOT IMPLEMENTED. Disabled by default.** This document is a placeholder
roadmap requested in the focus group (A.10). No live translation behavior exists
in the codebase, and none should be enabled without the work below.

## Why this is gated

AI-SoNar is safety-sensitive. Mistranslated stroke guidance, escalation wording,
or medication questions could cause real harm. **Machine translation is not
acceptable** for this content. A patient (Lisa) compared the app to a medical
interpreter — and medical interpretation has professional standards for a reason.

## What "enabling translation" would require (not done here)

1. **Professional human translation** of all patient-facing scenario text
   (`scenarios/*.yml`) into each target language, with clinical review per language.
2. **Translated, clinically-reviewed knowledge base** content for the RAG path —
   the English `stroke-care-knowledge` index is not sufficient.
3. **Localized escalation / 911-equivalent guidance** appropriate to the patient's
   region and language (emergency numbers and care pathways differ).
4. **Speech (STT/TTS) language configuration.** Azure Speech (`api/services/azure_speech.py`)
   supports multilingual STT/TTS, so the plumbing exists, but it must be wired to a
   per-session language only after 1–3 are complete.
5. **Sign-off from the clinical lead (Dr. Zand)** and the translation vendor before
   any language goes live.

## Current scaffold

- Config flag `translation.enabled: false` in `config/azure.yaml`
  (env override: `TRANSLATION_ENABLED`). It currently does nothing — there is no
  translation code path. It exists only to mark the roadmap and make the intended
  default explicit.

## Explicitly out of scope right now

- Any automatic / machine translation of prompts, responses, or knowledge content.
- Any language selection UI.
- Any non-English TTS/STT wiring.
