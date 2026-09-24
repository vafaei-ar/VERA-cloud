# AI-SoNar publication traceability record

This record supports the formative co-design manuscript for AI-SoNar / VERA. It maps stakeholder-derived requirements from the June 2026 preliminary session to concrete code changes made before the revised-prototype session.

## Provenance limitation

Contemporaneous study records differ by one day on the exact preliminary-session date (June 15 versus June 16, 2026), so the manuscript reports the session as occurring in June 2026 unless the date is confirmed from the authoritative study record. An annotated repository tag named `pre-cowork-baseline` was created on June 17, 2026 at 01:18 UTC. The tag points to commit `fe8a678e9632cd8cb012573a6be31e68b109f97a`, which was authored in November 2025. The tag therefore documents an earlier committed baseline selected before the mapped June 17 implementation sequence; it is **not** evidence that this commit exactly reproduces the live June 2026 demonstration or that it was the immediately preceding repository commit.

The June and August demonstration builds were not tagged as immutable releases at the time of the sessions. Therefore, the manuscript should not claim that the exact demonstrated builds were archived as fixed releases. The commit-level records below are the verifiable implementation evidence for the changes committed after the preliminary June session. They do not reconstruct uncommitted local state that may have existed during a live demonstration.

All mapped A.1-A.12 and C.1 commits were authored in Git as `Cowork (AI-SoNar)` on June 17, 2026 under investigator direction. Cowork was an AI coding workflow. Investigators defined the stakeholder-derived requirements, reviewed the implementation, and retained responsibility for clinical interpretation and safety boundaries.

## Session-1 requirement to implementation mapping

| ID | Stakeholder-derived requirement | Implemented response | Commit date (UTC) | Verifiable commit |
|---|---|---|---|---|
| A.1 | Make human oversight visible and concrete. | Added human-review language to greetings and shared system-prompt guidance. | 2026-06-17 | [795fcfd](https://github.com/vafaei-ar/VERA-cloud/commit/795fcfd5b3393eceff2165a282ce5d4e7a33f06b) |
| A.2 | Make expected response timing explicit without implying real-time monitoring. | Added configurable spoken/on-screen response-time expectations; clinical wording remains draft. | 2026-06-17 | [d36ad2c](https://github.com/vafaei-ar/VERA-cloud/commit/d36ad2c4d13cc2d98e07296bf10d2160b99170ca) |
| A.3 | Let users indicate their own sense of urgency without allowing it to override automatic safety flags. | Added a separate advisory urgency field and UI/API support. | 2026-06-17 | [cedb42f](https://github.com/vafaei-ar/VERA-cloud/commit/cedb42fd41db198ad877cd3a9a1caf771171f097) |
| A.4 | Replace generic boilerplate with more specific, useful responses. | Added specificity guidance and a non-blocking boilerplate detector; stroke-type knowledge remains a clinical-authoring dependency. | 2026-06-17 | [977ab1b](https://github.com/vafaei-ar/VERA-cloud/commit/977ab1b4fc65f972611afa8c52421b5846aa1d1f) |
| A.5 | Avoid bursty interim transcripts that increase cognitive load. | Changed interim ASR display to a single in-place, debounced transcript bubble. | 2026-06-17 | [b557a10](https://github.com/vafaei-ar/VERA-cloud/commit/b557a10195d522d203c4016df0eb56216c12881f) |
| A.6 | Support distinct survivor, caregiver, and clinician roles. | Added role tracks that alter framing/wording while keeping clinical content identical. | 2026-06-17 | [21af58b](https://github.com/vafaei-ar/VERA-cloud/commit/21af58bcba13cf371ed6ce3c8338bc02f16c3583) |
| A.7 | Reduce typing/dexterity burden and better accommodate slow or effortful speech. | Increased ASR silence tolerance, enlarged tap targets, retained optional typing, and added voice-driven reminder capture. | 2026-06-17 | [d37c1e7](https://github.com/vafaei-ar/VERA-cloud/commit/d37c1e78ee25d2a1229530e2bf4a745fc1df86e4) |
| A.8 | Provide geographically relevant practical resources without presenting them as clinical recommendations. | Added an opt-in, information-only local-resource layer. | 2026-06-17 | [b86c616](https://github.com/vafaei-ar/VERA-cloud/commit/b86c6165ff67fd5733a58b094f3fc1a956abc036) |
| A.9 | Use shorter, plainer patient-facing language. | Rewrote scenario text and added a plain-language prompt rule. | 2026-06-17 | [2ef2eec](https://github.com/vafaei-ar/VERA-cloud/commit/2ef2eec31048588a99cecdecf445995527c5bdf7) |
| A.10 | Address language access, but do not deploy unreviewed machine translation. | Added a disabled translation scaffold and roadmap requiring professional translation and clinical review before activation. | 2026-06-17 | [1cc16c7](https://github.com/vafaei-ar/VERA-cloud/commit/1cc16c7e1cf7a651014da3010c7d16a4038bb791) |
| A.11 | Give clinicians concise, prioritized information rather than another undifferentiated inbox. | Added a prioritized clinician-summary endpoint and draft role-routing logic. | 2026-06-17 | [1de17ea](https://github.com/vafaei-ar/VERA-cloud/commit/1de17eaa0ba6750035b722f6a2caf1a0ca73fff8) |
| A.12 | Clarify that the assistant is automated and is not the human stroke navigator/care team. | Changed self-identification and added a standing who-to-call/emergency notice. | 2026-06-17 | [cc8f4f8](https://github.com/vafaei-ar/VERA-cloud/commit/cc8f4f8bc328a3e97284a998b39939eb856ecb64) |
| C.1 | Distinguish survivor/caregiver initiation and gate patient-context use on caregiver consent. | Added the session-initiation endpoint, role selection, patient-ID handling, and caregiver-consent gating. | 2026-06-17 | [0641be8](https://github.com/vafaei-ar/VERA-cloud/commit/0641be84c18fea7efd53a1785f1b39d36947b0a2) |

## Requirements that remained unresolved after revised-prototype and workflow review

These items were deliberately retained as pending requirements rather than described as completed features:

- communication-aware interaction for aphasia and word-finding difficulty, including slower pacing and longer response windows;
- adaptive questioning and a way to skip already-understood or repetitive content;
- an explicit patient-requested callback / human-contact pathway;
- clinician-approved escalation thresholds and after-hours ownership;
- safe mental-health escalation before any automated depression screening;
- clinician-reviewed multilingual content before translation is enabled;
- modality choice that does not depend on smartphone-app installation;
- governance for retaining or reviewing original voice recordings;
- clinically curated stroke-specific knowledge content and maintenance procedures.

## Repository roles and licensing

- **VERA-cloud** contains the conversational / clinical-logic research prototype and is licensed under GPL-3.0.
- **Kura** is a separate source-available iOS/broker delivery layer used for bounded beta development; it contains no independent clinical logic. At the time of this record, Kura has no open-source license and should not be described as open source.

No repository state or software feature should be interpreted as clinically validated or cleared for patient care solely because it is implemented or tested in code.
