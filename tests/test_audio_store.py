import io
import wave
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from api.services import audio_store, outcomes
from api import main


def wav(frames=160):
    output = io.BytesIO()
    with wave.open(output, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)
    return output.getvalue()


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setenv("AUDIO_RECORDING_ENABLED", "true")
    outcomes.record_progress("audio", state="in_progress", consent="accepted")
    return "audio"


def test_audio_requires_independent_consent_and_is_idempotent(session):
    with pytest.raises(PermissionError):
        audio_store.store_clip(session, "turn1", wav())
    audio_store.set_consent(session, True)
    clip = audio_store.store_clip(session, "turn1", wav())
    assert clip["source"] == "original_respondent_microphone"
    assert audio_store.clip_path(session, "turn1").read_bytes() == wav()
    assert audio_store.store_clip(session, "turn1", wav()) == clip
    assert len(outcomes.load_outcome(session)["original_audio"]) == 1
    with pytest.raises(ValueError):
        audio_store.store_clip(session, "turn1", wav(161))


def test_recording_does_not_begin_before_answer_consent(session):
    outcomes.record_progress(session, state="awaiting_consent", consent="pending")
    audio_store.set_consent(session, True)
    with pytest.raises(PermissionError):
        audio_store.store_clip(session, "turn1", wav())


def test_disabled_invalid_and_ended_recordings_rejected(session, monkeypatch):
    audio_store.set_consent(session, True)
    with pytest.raises((wave.Error, EOFError)):
        audio_store.store_clip(session, "turn1", b"x" * 64)
    with pytest.raises(ValueError):
        audio_store.store_clip(session, "../escape", wav())
    monkeypatch.setenv("AUDIO_RECORDING_ENABLED", "false")
    with pytest.raises(ValueError):
        audio_store.store_clip(session, "turn1", wav())
    monkeypatch.setenv("AUDIO_RECORDING_ENABLED", "true")
    outcomes.record_progress(session, state="withdrawn")
    with pytest.raises(PermissionError):
        audio_store.store_clip(session, "turn1", wav())


def test_authorized_access_audited_and_retention_enforced(session, monkeypatch):
    audio_store.set_consent(session, True)
    audio_store.store_clip(session, "turn1", wav(), partial=True)
    monkeypatch.setenv("VERA_SERVICE_KEY", "test-audio-service-key-32-characters")
    client = TestClient(main.app)
    assert client.get("/api/session/audio/audio/turn1").status_code == 401
    response = client.get("/api/session/audio/audio/turn1", headers={"Authorization": "Bearer test-audio-service-key-32-characters", "X-Reviewer": "reviewer1"})
    assert response.content == wav()
    assert response.headers["cache-control"] == "no-store"
    record = outcomes.load_outcome(session)
    assert record["audio_access_log"][0]["viewer"] == "reviewer1"
    record["original_audio"][0]["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    outcomes.save_outcome(record)
    with pytest.raises(FileNotFoundError):
        audio_store.authorized_playback(session, "turn1", "reviewer")
    audio_store.purge_expired()
    assert not audio_store.clip_path(session, "turn1").exists()
    assert outcomes.load_outcome(session)["original_audio"][0]["deleted_at"]


def test_post_emergency_upload_requires_recent_acknowledged_turn(session):
    audio_store.set_consent(session, True)
    outcomes.record_progress(session, state="escalated", audio_turns={"turn1": datetime.now(timezone.utc).isoformat()})
    with pytest.raises(ValueError):
        audio_store.store_clip(session, "unaccepted", wav(), acknowledged_turn=True)
    assert audio_store.store_clip(session, "turn1", wav(), acknowledged_turn=True)["id"] == "turn1"
    outcomes.record_progress(session, audio_turns={"turn2": (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()})
    with pytest.raises(PermissionError):
        audio_store.store_clip(session, "turn2", wav(), acknowledged_turn=True)
    outcomes.record_progress(session, state="withdrawn", audio_turns={"turn3": datetime.now(timezone.utc).isoformat()})
    with pytest.raises(PermissionError):
        audio_store.store_clip(session, "turn3", wav(), acknowledged_turn=True)


def test_separate_http_upload_is_service_protected_and_turn_bound(session, monkeypatch):
    key = "synthetic-audio-service-key-long-enough"
    monkeypatch.setenv("VERA_SERVICE_KEY", key)
    audio_store.set_consent(session, True)
    outcomes.record_progress(session, state="escalated", audio_turns={"urgent": datetime.now(timezone.utc).isoformat()})
    client = TestClient(main.app)
    path = "/api/session/audio/audio/urgent"
    assert client.post(path, content=wav()).status_code == 401
    headers = {"Authorization": "Bearer " + key}
    assert client.post(path, headers=headers, content=b"x" * 64).status_code == 409
    response = client.post(path, headers=headers, content=wav())
    assert response.status_code == 200 and response.json()["stored"]
    assert outcomes.load_outcome(session)["state"] == "escalated"
