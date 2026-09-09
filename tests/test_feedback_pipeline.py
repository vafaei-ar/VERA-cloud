import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from api.services.enhanced_dialog import EnhancedDialog
from api.services.clinical_policy import load_policy
from api.services import outcomes


def dialog():
    return EnhancedDialog(None, None, None, str(Path(__file__).resolve().parents[1] / "scenarios/guided.yml"))


def answer(d, text):
    return asyncio.run(d.process_user_response(text))


@pytest.mark.parametrize("text", ["not sure", "yes no", "yesterday", "I disagree", "I am not ready", "okay but no"])
def test_ambiguous_consent_never_advances(text):
    d = dialog()
    assert answer(d, text)["type"] == "clarification"
    assert d.current_index == 0 and not d.responses and d.consent == "pending"


def test_decline_is_terminal_and_no_clinical_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    d = dialog()
    assert answer(d, "no")["state"] == "declined"
    assert answer(d, "yes")["state"] == "declined"
    record = outcomes.record_progress("consent", d, state="declined")
    assert not record.get("responses") and not record["flags"]


def test_safety_immediate_durable_without_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    d = dialog()
    answer(d, "yes")
    result = answer(d, "my face is drooping")
    assert result["type"] == "emergency_alert"
    assert result["next_action"] == "end_session"
    assert not d.is_complete()
    outcomes.record_progress("safety", d, state="escalated")
    record = outcomes.load_outcome("safety")
    assert outcomes.build_clinician_summary(record)["has_priority"]
    assert list(outcomes.pending_outcomes())[0]["state"] == "escalated"
    assert record["policy"]["digest"] == d.policy["digest"]


def test_unknown_stroke_education_and_question_detour():
    d = dialog()
    assert "ischemic" not in d.build_greeting()
    answer(d, "yes")
    before = d.current_index
    answer(d, "Can I drive?")
    assert d.current_index == before
    assert "Different types" in answer(d, "yes")["message"]


def test_callback_and_spoken_urgency():
    d = dialog()
    answer(d, "yes")
    before = d.current_index
    answer(d, "call me")
    assert d.callback_requested and d.current_index == before
    d.current_index = next(i for i, q in enumerate(d.scenario["flow"]) if q.get("key") == "urgency_self_report")
    answer(d, "not urgent")
    assert d.user_urgency is None
    answer(d, "soon")
    assert d.user_urgency == "soon"


def test_uncertain_urgency_is_preserved_and_visible_without_inventing_red_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    d = dialog()
    answer(d, "yes")
    d.current_index = next(i for i, q in enumerate(d.scenario["flow"]) if q.get("key") == "urgency_self_report")
    answer(d, "not sure")
    assert d.user_urgency == "unsure"
    summary = outcomes.build_clinician_summary(outcomes.record_progress("unsure", d, state="completed"))
    assert summary["has_priority"] and not summary["priority_items"]
    assert summary["user_reported_urgency"] == "unsure"


def test_retention_failure_does_not_block_outcome_dispatch(monkeypatch):
    from unittest.mock import AsyncMock
    from api.services import outbox, audio_store
    monkeypatch.setattr(audio_store, "purge_expired", lambda: (_ for _ in ()).throw(RuntimeError("bad retention record")))
    dispatch = AsyncMock(return_value=1)
    monkeypatch.setattr(outbox, "dispatch_once", dispatch)
    async def stop_after_pass(_):
        raise asyncio.CancelledError()
    monkeypatch.setattr(outbox.asyncio, "sleep", stop_after_pass)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(outbox.run_dispatcher())
    dispatch.assert_awaited_once()


def test_followups_bounded_and_never_delay_emergency():
    d = dialog()
    answer(d, "yes")
    d.current_index = next(i for i, q in enumerate(d.scenario["flow"]) if q.get("key") == "new_symptoms")
    before = d.current_index
    answer(d, "my sleep has changed")
    answer(d, "unchanged since my stroke")
    assert d.current_index == before
    answer(d, "unsure")
    assert d.current_index > before


def test_question_and_callback_detours_return_to_actual_followup():
    d = dialog()
    answer(d, "yes")
    d.current_index = next(i for i, q in enumerate(d.scenario["flow"]) if q.get("key") == "new_symptoms")
    answer(d, "my sleep has changed")
    for detour in ("can I drive?", "call me", "repeat"):
        assert d.current_prompt() in answer(d, detour)["message"]
        assert d.followup == 2


def test_parallel_updates_do_not_lose_notes_or_urgency(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda n: outcomes.record_reminder("parallel", str(n)), range(24)))
    outcomes.record_user_urgency("parallel", "soon")
    record = outcomes.load_outcome("parallel")
    assert len(record["reminders"]) == 24 and record["version"] == 25
    assert outcomes.build_clinician_summary(record)["has_priority"]
    outcomes.mark_delivered("parallel", 24)
    assert len(list(outcomes.pending_outcomes())) == 1
    outcomes.mark_delivered("parallel", 25)
    assert not list(outcomes.pending_outcomes())


def test_invalid_ids_and_corrupt_outcomes_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    with pytest.raises(ValueError):
        outcomes.load_outcome("../bad")
    (tmp_path / "bad.json").write_text("corrupt")
    with pytest.raises(RuntimeError):
        outcomes.record_user_urgency("bad", "routine")


def test_policy_invalid_missing_messages(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text("version: test\nstatus: draft\n")
    with pytest.raises(ValueError):
        load_policy(p)


@pytest.mark.parametrize("responses, expected", [
    (["no"], "declined"),
    (["yes", "my face is drooping"], "escalated"),
    (["yes", "yes", "stroke", "fine", "no", "routine"], "completed"),
    (["yes", "stop"], "withdrawn"),
])
def test_real_websocket_terminal_protocol(tmp_path, monkeypatch, responses, expected):
    from fastapi.testclient import TestClient
    from api import main
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setattr(main, "azure_speech", None)
    async def no_delay(*args):
        pass
    monkeypatch.setattr(main.asyncio, "sleep", no_delay)
    d = dialog()
    main.active_sessions["sockettest"] = {"dialog": d, "voice": None, "rate": 0.85}
    main.conversation_data["sockettest"] = {"transcript": [], "audio_segments": []}
    outcomes.record_progress("sockettest", d, state="awaiting_consent")
    client = TestClient(main.app)
    with client.websocket_connect("/ws/audio/sockettest") as ws:
        assert "Say yes or no" in ws.receive_json()["text"]  # introduction + consent
        for response in responses:
            ws.send_json({"type": "text_input", "text": response})
            result = ws.receive_json()
        assert result["type"] == {"completed": "completion", "escalated": "emergency_alert"}.get(expected, "session_ended")
    assert outcomes.load_outcome("sockettest")["state"] == expected
    if expected == "declined":
        assert not main.conversation_data["sockettest"]["transcript"]


def test_unknown_outcome_and_preconsent_urgency(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api import main
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    client = TestClient(main.app)
    assert client.get("/api/session/missing/clinician-summary").status_code == 404
    assert client.post("/api/session/missing/urgency", json={"urgency": "soon"}).status_code == 404
    outcomes.record_progress("pending", dialog(), state="awaiting_consent")
    assert client.post("/api/session/pending/urgency", json={"urgency": "soon"}).status_code == 409
    assert client.post("/api/session/pending/stop", json={}).json()["state"] == "declined"


def test_ask_requires_opt_in_and_does_not_overwrite_request(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api import main
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setenv("ASK_ENABLED", "true")
    client = TestClient(main.app)
    body = {"question": "my face is drooping", "session_id": "ask-test"}
    response = client.post("/api/ask", json=body).json()
    assert response["kind"] == "emergency" and not response["saved"]
    assert not outcomes.outcome_exists("ask-test")
    body["share_with_team"] = True
    assert client.post("/api/ask", json=body).json()["saved"]
    version = outcomes.load_outcome("ask-test")["version"]
    client.post("/api/ask", json=body)
    assert outcomes.load_outcome("ask-test")["version"] == version
    body["question"] = "can I drive?"
    assert not client.post("/api/ask", json=body).json()["saved"]
    assert outcomes.load_outcome("ask-test")["state"] == "escalated"


def test_policy_snapshot_and_configurable_rules():
    from api.services.flagging import evaluate
    d = dialog()
    d.policy["flagging"]["phrase_overrides"] = {"t1_face_droop": ["synthetic signal"]}
    assert evaluate("synthetic signal", policy=d.policy)["overall_tier"] == 1
    assert evaluate("synthetic signal")["overall_tier"] == 3
    d.policy["routing"]["routine"] = "review_team"
    summary = outcomes.build_clinician_summary({"policy": d.policy, "flags": []})
    assert summary["suggested_route"]["role"] == "review_team"


def test_outbox_retries_until_version_acknowledged(tmp_path, monkeypatch):
    from api.services import outbox
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setenv("KURA_EVENT_URL", "https://example.invalid/v1/vera/events")
    monkeypatch.setenv("KURA_EVENT_KEY", "synthetic-key")
    record = outcomes.record_progress("delivery", state="interrupted")
    class Client:
        accepted = False
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, headers, json):
            assert headers["X-Event-Key"] == "synthetic-key"
            if not self.accepted: raise RuntimeError("offline")
            return self
        def raise_for_status(self): pass
        def json(self): return {"accepted_version": record["version"]}
    monkeypatch.setattr(outbox.httpx, "AsyncClient", Client)
    assert asyncio.run(outbox.dispatch_once()) == 0
    assert len(list(outcomes.pending_outcomes())) == 1
    Client.accepted = True
    assert asyncio.run(outbox.dispatch_once()) == 1
    assert not list(outcomes.pending_outcomes())


def test_duplicate_turn_does_not_advance_and_corrections_are_preserved(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api import main
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setattr(main, "azure_speech", None)
    d = dialog()
    main.active_sessions["duplicate"] = {"dialog": d, "voice": None, "rate": 0.85}
    outcomes.record_progress("duplicate", d, state="awaiting_consent")
    with TestClient(main.app).websocket_connect("/ws/audio/duplicate") as ws:
        ws.receive_json()
        ws.send_json({"type": "text_input", "text": "yes", "message_id": "consent"})
        ws.receive_json()
        index = d.current_index
        ws.send_json({"type": "text_input", "text": "yes", "message_id": "consent"})
        assert ws.receive_json()["duplicate"]
        assert d.current_index == index
        ws.send_json({"type": "text_input", "text": "no", "original_transcript": "know", "message_id": "education"})
        # Educational denial currently emits two messages (brief text, next question).
        ws.receive_json()
    assert outcomes.load_outcome("duplicate")["transcription_reviews"][0]["original"] == "know"


def test_authored_pathway_initial_danger_never_waits_for_clarification():
    d = dialog()
    answer(d, "yes")
    d.current_index = next(i for i, q in enumerate(d.scenario["flow"]) if q.get("key") == "new_symptoms")
    assert answer(d, "I have new weakness")["type"] == "emergency_alert"


def test_headache_pathway_baseline_unknown_and_immediate_qualifier():
    d = dialog()
    answer(d, "yes")
    d.current_index = next(i for i, q in enumerate(d.scenario["flow"]) if q.get("key") == "new_symptoms")
    assert "headache" in answer(d, "my headache bothers me")["message"]
    assert d.active_pathway == "headache"
    answer(d, "not sure")
    assert d.overall_tier() == 2
    assert answer(d, "the worst ever")["type"] == "emergency_alert"


def test_unknown_response_target_has_no_invented_deadline():
    from api.services.clinical_policy import response_message
    policy = load_policy()
    assert "business days" not in response_message(policy)
    policy["response_windows"]["routine_business_days"] = 3
    assert "3 business days" in response_message(policy)
    assert "business days" not in response_message(policy, priority=True)
