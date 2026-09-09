from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import pytest

from api import main
from api.services import session_auth, outcomes
from tests.test_feedback_pipeline import dialog, answer


def test_service_key_is_required_for_clinical_http(monkeypatch):
    monkeypatch.setenv("VERA_SERVICE_KEY", "synthetic-service-key-long-enough-for-test")
    client = TestClient(main.app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/session/unknown/clinician-summary").status_code == 401


def test_session_tokens_do_not_grant_service_access_and_expire():
    token, metadata = session_auth.issue_session_token()
    assert session_auth.session_authorized(token, metadata)
    assert not session_auth.session_authorized("wrong", metadata)
    assert not session_auth.session_authorized(token, {**metadata, "state": "withdrawn"})
    assert not session_auth.session_authorized(token, {**metadata, "session_token_expires": 0})


def test_real_websocket_requires_correct_session_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setenv("VERA_SERVICE_KEY", "synthetic-service-key-long-enough-for-test")
    monkeypatch.setattr(main, "azure_speech", None)
    token, metadata = session_auth.issue_session_token()
    d = dialog()
    outcomes.record_progress("secure", d, state="awaiting_consent", **metadata)
    main.active_sessions["secure"] = {"dialog": d, "voice": None, "rate": 0.85}
    client = TestClient(main.app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/audio/secure"):
            pass
    with client.websocket_connect("/ws/audio/secure", headers={"Authorization": "Bearer " + token}) as ws:
        assert "Say yes or no" in ws.receive_json()["text"]
        ws.send_json({"type": "text_input", "text": "no"})
        assert ws.receive_json()["state"] == "declined"


def test_restart_resumes_pending_clarification_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setenv("VERA_SERVICE_KEY", "synthetic-service-key-long-enough-for-test")
    monkeypatch.setattr(main, "azure_speech", None)
    d = dialog()
    answer(d, "yes")
    d.current_index = next(i for i, q in enumerate(d.scenario["flow"]) if q.get("key") == "new_symptoms")
    receipt = answer(d, "my sleep has changed")
    d.turn_receipts["accepted-turn"] = {**receipt, "_request_digest": main._answer_request_digest(
        {"text": "my sleep has changed"})}
    token, metadata = session_auth.issue_session_token()
    outcomes.record_progress("restart", d, state="interrupted", **metadata)
    client = TestClient(main.app)
    with client.websocket_connect("/ws/audio/restart", headers={"Authorization": "Bearer " + token}) as ws:
        greeting = ws.receive_json()
        assert greeting["resumed"] and greeting["consent"] == "accepted"
        assert d.current_prompt() in greeting["text"]
        ws.send_json({"type": "text_input", "text": "my sleep has changed", "message_id": "accepted-turn"})
        assert ws.receive_json()["duplicate"]
        assert main.active_sessions["restart"]["dialog"].followup == 2
        ws.send_json({"type": "text_input", "text": "stop"})
        assert ws.receive_json()["state"] == "withdrawn"


def test_snapshot_retains_policy_but_rejects_changed_detection_engine():
    d = dialog()
    snapshot = d.snapshot()
    snapshot["state"]["policy"]["messages"]["consent"] = "Saved draft wording"
    restored = main.EnhancedDialog.restore(snapshot)
    assert restored.policy["messages"]["consent"] == "Saved draft wording"
    snapshot["state"]["policy"]["engine_digest"] = "changed"
    with pytest.raises(ValueError, match="engine changed"):
        main.EnhancedDialog.restore(snapshot)


def test_production_refuses_unreviewed_policy_before_external_startup(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_MODE", "production")
    monkeypatch.setenv("VERA_SERVICE_KEY", "synthetic-service-key-long-enough-for-test")
    monkeypatch.setenv("AUDIO_RECORDING_ENABLED", "false")
    with pytest.raises(RuntimeError, match="clinical policy approval"):
        with TestClient(main.app):
            pass


def test_restart_after_last_answer_finishes_without_extra_question(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setenv("VERA_SERVICE_KEY", "synthetic-service-key-long-enough-for-test")
    monkeypatch.setattr(main, "azure_speech", None)
    d = dialog()
    answer(d, "yes")
    d.current_index = len(d.scenario["flow"])
    token, metadata = session_auth.issue_session_token()
    outcomes.record_progress("lastanswer", d, state="interrupted", **metadata)
    with TestClient(main.app).websocket_connect("/ws/audio/lastanswer", headers={"Authorization": "Bearer " + token}) as ws:
        assert ws.receive_json()["resumed"]
        assert ws.receive_json()["type"] == "completion"
    assert outcomes.load_outcome("lastanswer")["state"] == "completed"
