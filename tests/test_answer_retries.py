"""Synthetic WebSocket retry contracts; no cloud services or notifications."""
import copy

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from api import main
from api.services import outcomes, session_auth
from tests.test_feedback_pipeline import dialog


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setenv("DEPLOYMENT_MODE", "development")
    monkeypatch.setenv("VERA_SERVICE_KEY", "synthetic-retry-service-key-long-enough")
    monkeypatch.setattr(main, "azure_speech", None)
    monkeypatch.setattr(main, "active_sessions", {})
    monkeypatch.setattr(main, "websocket_connections", {})
    monkeypatch.setattr(main, "conversation_data", {})
    d = dialog()
    token, metadata = session_auth.issue_session_token()
    outcomes.record_progress("retry", d, state="awaiting_consent", **metadata)
    main.active_sessions["retry"] = {"dialog": d, "voice": None, "rate": 0.85}
    return TestClient(main.app), {"Authorization": "Bearer " + token}


def test_old_answer_retry_returns_current_prompt_and_does_not_write(session):
    client, headers = session
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        ws.receive_json()
        ws.send_json({"type": "text_input", "text": "yes", "message_id": "consent"})
        ws.receive_json()
        ws.send_json({"type": "text_input", "text": "yes", "message_id": "education"})
        ws.receive_json()
        d = main.active_sessions["retry"]["dialog"]
        before = copy.deepcopy(outcomes.load_outcome("retry"))
        ws.send_json({"type": "text_input", "text": "yes", "message_id": "consent"})
        replay = ws.receive_json()
        assert replay["duplicate"] and replay["saved"]
        assert replay["message_id"] == "consent"
        assert replay["text"] == d.current_prompt()
        assert replay["consent"] == "accepted"
        assert outcomes.load_outcome("retry") == before


@pytest.mark.parametrize("changed", [
    {"text": "call me"},
    {"original_transcript": "different recognized words"},
    {"expects_audio": True},
])
def test_reused_identifier_with_changed_payload_is_rejected(session, changed):
    client, headers = session
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        ws.receive_json()
        payload = {"type": "text_input", "text": "yes", "message_id": "consent"}
        ws.send_json(payload)
        ws.receive_json()
        before = copy.deepcopy(outcomes.load_outcome("retry"))
        ws.send_json({**payload, **changed})
        error = ws.receive_json()
        assert error["type"] == "error" and error["code"] == "answer_id_conflict"
        assert error["message_id"] == "consent"
        assert outcomes.load_outcome("retry") == before


def test_retry_identity_survives_server_session_restore(session):
    client, headers = session
    payload = {"type": "text_input", "text": "yes", "message_id": "consent"}
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        ws.receive_json()
        ws.send_json(payload)
        ws.receive_json()
    assert "retry" not in main.active_sessions
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        assert ws.receive_json()["resumed"]
        before = copy.deepcopy(outcomes.load_outcome("retry"))
        ws.send_json(payload)
        assert ws.receive_json()["duplicate"]
        ws.send_json({**payload, "text": "call me"})
        assert ws.receive_json()["code"] == "answer_id_conflict"
        assert outcomes.load_outcome("retry") == before


@pytest.mark.parametrize("identifier", ["", "   "])
def test_explicit_empty_identifier_is_not_silently_non_idempotent(session, identifier):
    client, headers = session
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        ws.receive_json()
        ws.send_json({"type": "text_input", "text": "yes", "message_id": identifier})
        assert ws.receive_json()["type"] == "error"
        assert outcomes.load_outcome("retry")["consent"] == "pending"


def test_legacy_receipt_without_identity_is_not_blindly_replayed(session):
    client, headers = session
    d = main.active_sessions["retry"]["dialog"]
    d.turn_receipts["legacy"] = {"message": "Old response"}
    outcomes.record_progress("retry", d)
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        ws.receive_json()
        before = copy.deepcopy(outcomes.load_outcome("retry"))
        ws.send_json({"type": "text_input", "text": "yes", "message_id": "legacy"})
        assert ws.receive_json()["code"] == "retry_identity_unavailable"
        assert outcomes.load_outcome("retry") == before


@pytest.mark.parametrize("invalidation", [{"session_token_expires": 0}, {"state": "withdrawn"}])
def test_retry_rechecks_access_before_replaying_saved_content(session, invalidation):
    client, headers = session
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        ws.receive_json()
        payload = {"type": "text_input", "text": "yes", "message_id": "consent"}
        ws.send_json(payload)
        ws.receive_json()
        outcomes.record_progress("retry", **invalidation)
        ws.send_json(payload)
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1008
