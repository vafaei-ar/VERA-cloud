# Session 1/2 engine changes

Updated 2026-09-07. All clinical configuration remains DRAFT.
The coordinated engineering runbook is `Kura/docs/FEEDBACK_IMPLEMENTATION.md`
in the sibling repository. Full status/acceptance tracking is in the sibling
`VERA-Kura-Project-Docs/IMPLEMENTATION_STATUS_2026-09-07.md`.

## Clinical review surface

Edit `config/clinical_policy.yaml` (or `CLINICAL_POLICY_PATH`), increment its
version, and run the engine tests. Configure messages, stroke education, response
targets, routing, phrase overrides, BP thresholds, and bounded symptom pathways.
Do not duplicate clinical rules in Kura. Unknown baseline remains explicit;
existing red flags are never suppressed globally for chronic impairment.
The initial pathways cover headache, balance, and weakness/communication; they
are reasonable conservative research defaults, not validated triage rules.

Each session saves its entire policy, SHA-256 digest, detection/dialog-engine
digest, exact scenario, question index, accepted turns, and partial concerns.
Changing YAML affects new sessions; an existing session resumes with its saved
version. Changed detection/dialog source code blocks incompatible resume rather
than silently changing decisions mid-session. Keep outcome snapshots restricted
like clinical records, including any patient-history snapshot.

`routine_business_days: null` makes no invented service commitment. Optional
numeric targets are bounded and explicitly not guaranteed; flagged, unsure,
soon/urgent, and callback outcomes do not receive the routine target copy.

FAQ text/keywords/sources live in `config/faq.yaml`. Tied matches refuse rather
than choosing arbitrarily. General information is not individualized advice;
unsupported questions can use opt-in human follow-up. `ASK_ENABLED=false` is the
standalone default, but in-session FAQ detours also require content review.
Public resource entries in `config/resources.yaml` have source URLs and a check
date. Local capacity, eligibility, costs, and clinical suitability remain unverified.

## Persistence, delivery, and privacy

`OUTCOMES_PATH` must be persistent, access-restricted, and encrypted by the host.
Atomic JSON replacement with fsync and per-session locks also forms the durable
outbox. The dispatcher retries versions until Kura accepts them. It does not mean
a clinician has reviewed an outcome. Original WAV is opt-in, separate from
synthesized speech, capped at 10 MB/120 seconds/100 clips and expires by configured
retention. HTTP upload is bound to a recently accepted consented turn, allowing
emergency guidance to precede recording transfer. Clinician playback requires
service authorization and writes an access audit. Backup and non-audio outcome
retention still require the institutional data-owner policy.

Use one worker/replica. File-update locks are not a distributed WebSocket lease.
Credentials: VERA_SERVICE_KEY for service HTTP; limited one-hour hashed session
tokens for conversation WS; independent KURA_EVENT_KEY for outbound events.
Do not expose service keys to participants. The standalone web demo has no
production participant-login UI; use the authenticated Kura broker path.

Production mode requires recorded policy approval, approved FAQ status, strong
service auth, and authenticated HTTPS outcome delivery. These checks do not
replace security or clinical review. Example variable names are in `.env.example`.

## Verify

The communication-preferences increment accepts validated nonclinical interaction
settings on `/session/start`, retains them in the outcome, and reports actual TTS
rate in the WS greeting. Settings never change automatic safety evaluation or
recording permission. Kura owns participant preference editing and readiness/
handoff authorization. The latest suite has 118 passing VERA backend tests.
Current native build and expanded paired-process checks pass. The user also reported a successful iPhone build and
one working call; exact revisions and broader feature coverage were not captured.
Further phone acceptance tests are deferred at the user's request. The successful
agent-run build/integration covers the current recovery and preferences revision.

Identified WebSocket answers now bind `message_id` to a SHA-256 fingerprint of
the exact text, original transcription, and recording expectation. Matching retries
return `duplicate: true`, `saved: true`, the ID, current prompt, and consent without
another state write. Changed content returns `answer_id_conflict`; old receipts
without a fingerprint return `retry_identity_unavailable`. Neither error submits
the incoming answer. Clients must not automatically assign a new ID to such retries.
The fingerprint is protected session metadata, not anonymized clinical data.
Omitting an ID remains supported for legacy clients but gives no retry guarantee.
This server contract is used by the native durable recovery implementation below.
Ten synthetic tests cover replay, conflicts, restore, blank IDs, legacy receipts,
and expired/withdrawn access. Clinical rule configuration is unchanged.

**Latest increment supersedes the earlier verification gaps:** native pending-answer
recovery is implemented and the current simulator build, five Swift store tests,
and expanded paired-process harness pass. Broader phone tests remain deferred.
The server now advertises `answer_recovery: 1` and an opaque `answer_context` in
the greeting. Native requests carry `expected_context` and `request_receipt: true`;
the context is enforced for first submissions, while exact duplicate IDs still
reconcile without advancing. A durable write precedes the opt-in `answer_receipt`
event for nonterminal turns. Terminal receipt IDs and saved status travel in the
same message as emergency/stop/completion guidance, not in an earlier receipt
that could be delivered while terminal guidance is lost. Legacy clients without
the opt-in retain their existing message sequence.
`POST /api/session/{id}/answer-receipt` is service-authenticated and read-only,
returns no-store, verifies exact payload identity, and permits checking terminal
receipts without reopening a terminal socket. Missing answers only permit retry
for the same active, unexpired, restorable dialog context. Broker ownership and
revocation checks protect the participant-facing proxy. Six new engine tests and
two broker tests cover these contracts; no clinical thresholds changed.
Deploy both backends before the new native revision; no deployment occurred here.

Run `.venv/bin/python -m pytest -q` from this repository. Tests use synthetic
records, including consent, terminal events, policy snapshot recovery, uncertainty,
retention, outbox retry, protected HTTP/WS, and resource coverage. Kura's
`push-service/tests/integration_pair.py` supplies the real paired-process check.
The latest separated-upload revision passed the paired-process rerun and unsigned
native simulator rebuild on 2026-09-07. The harness verified emergency guidance
before optional audio upload, protected playback, automatic outbox delivery,
enrollment, callback routing, and participant revocation. Interactive device
testing is separate and remains outstanding.

Remaining clinical risks include phrase negation/temporal interpretation,
false alerts for chronic deficits, and hypothetical/general symptom questions.
Do not interpret a routine result as an all-clear. Device accessibility and real
APNs/Azure/SMTP/hosting behavior require hands-on validation before a pilot.
