from fastapi.testclient import TestClient
from api import main
from api.services.outcomes import load_outcome


def test_profile_snapshot_and_actual_tts_rate_do_not_change_safety(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTCOMES_PATH", str(tmp_path))
    monkeypatch.setenv("DEPLOYMENT_MODE", "development")
    key = "synthetic-profile-service-key-long-enough"
    monkeypatch.setenv("VERA_SERVICE_KEY", key)
    monkeypatch.setattr(main, "azure_openai", object())
    monkeypatch.setattr(main, "azure_search", object())
    class Speech:
        async def synthesize_text(self, *args, **kwargs):
            return b""
    monkeypatch.setattr(main, "azure_speech", Speech())
    client = TestClient(main.app)
    preferences = {"text_only": True, "speech_rate": 0.7, "communication_difficulty": "yes",
                   "support_preference": "independent", "manual_finish": True,
                   "review_before_sending": True, "silence_seconds": 15}
    response = client.post("/session/start", headers={"Authorization": "Bearer " + key},
        json={"patient_name": "Synthetic", "role": "survivor", "rate": 0.7,
              "communication_preferences": preferences})
    assert response.status_code == 200
    sid, token = response.json()["session_id"], response.json()["session_token"]
    assert load_outcome(sid)["communication_preferences"] == preferences
    with client.websocket_connect(f"/ws/audio/{sid}", headers={"Authorization": "Bearer " + token}) as ws:
        assert ws.receive_json()["speech_rate"] == 0.7
        ws.send_json({"type": "text_input", "text": "yes"})
        assert ws.receive_json()["consent"] == "accepted"
        ws.send_json({"type": "text_input", "text": "my face is drooping"})
        assert ws.receive_json()["type"] == "emergency_alert"
    assert load_outcome(sid)["communication_preferences"] == preferences


def test_preferences_contract_rejects_invalid_rate_or_permission_fields():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        main.CommunicationPreferences(speech_rate=4)
    with pytest.raises(ValidationError):
        main.CommunicationPreferences(audio_consent=True)
