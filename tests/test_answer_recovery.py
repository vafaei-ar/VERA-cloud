from api import main
from api.services import outcomes
from tests.test_answer_retries import session  # synthetic authenticated fixture


SERVICE = {"Authorization": "Bearer synthetic-retry-service-key-long-enough"}


def payload(context, text="yes", identifier="recover"):
    return {"type": "text_input", "message_id": identifier, "text": text,
            "expected_context": context, "expects_audio": False, "request_receipt": True}


def status(client, body):
    response = client.post("/api/session/retry/answer-receipt", json=body, headers=SERVICE)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    return response.json()


def test_lost_ack_reconciles_from_disk_without_resubmitting(session):
    client, headers = session
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        greeting = ws.receive_json()
        body = payload(greeting["answer_context"])
        assert greeting["answer_recovery"] == 1
        assert status(client, body)["can_retry"]
        ws.send_json(body)
        receipt = ws.receive_json()
        assert receipt["type"] == "answer_receipt" and receipt["saved"]
        assert "recover" in outcomes.load_outcome("retry")["dialog_snapshot"]["state"]["turn_receipts"]
        ws.receive_json()
    before = outcomes.load_outcome("retry")
    assert status(client, body)["status"] == "accepted"
    assert not status(client, body)["can_retry"]
    assert outcomes.load_outcome("retry") == before
    assert status(client, {**body, "text": "call me"})["status"] == "conflict"


def test_unsent_answer_keeps_same_question_context_after_reconnect(session):
    client, headers = session
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        body = payload(ws.receive_json()["answer_context"])
    assert status(client, body)["can_retry"]
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        assert ws.receive_json()["answer_context"] == body["expected_context"]
        ws.send_json(body)
        assert ws.receive_json()["saved"]
        ws.receive_json()
        ws.send_json(body)
        assert ws.receive_json()["duplicate"]


def test_old_question_cannot_receive_an_unconfirmed_answer(session):
    client, headers = session
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        old = payload(ws.receive_json()["answer_context"], identifier="not-sent")
        ws.send_json({**old, "message_id": "other-device"})
        ws.receive_json(); ws.receive_json()
        assert status(client, old)["status"] == "missing"
        assert not status(client, old)["can_retry"]
        before = outcomes.load_outcome("retry")
        ws.send_json(old)
        assert ws.receive_json()["code"] == "question_changed"
        assert outcomes.load_outcome("retry") == before


def test_terminal_emergency_receipt_can_be_reconciled_without_new_socket(session):
    client, headers = session
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        ws.send_json(payload(ws.receive_json()["answer_context"], identifier="consent"))
        context = ws.receive_json()["answer_context"]
        ws.receive_json()
        body = payload(context, "my face is drooping")
        ws.send_json(body)
        terminal = ws.receive_json()
        assert terminal["type"] == "emergency_alert"
        assert terminal["saved"] and terminal["message_id"] == body["message_id"]
    result = status(client, body)
    assert result["saved"] and result["state"] == "escalated" and result["message"]
    assert not result["can_retry"]


def test_no_success_receipt_when_durable_write_fails(session, monkeypatch):
    client, headers = session
    with client.websocket_connect("/ws/audio/retry", headers=headers) as ws:
        body = payload(ws.receive_json()["answer_context"])
        def fail(*args, **kwargs):
            raise OSError("synthetic full disk")
        monkeypatch.setattr(main, "record_progress", fail)
        ws.send_json(body)
        assert ws.receive_json()["type"] == "error"
    assert status(client, body)["status"] == "missing"


def test_receipt_requires_service_auth_and_expired_missing_turn_is_not_retried(session):
    client, headers = session
    body = payload(main._answer_context(main.active_sessions["retry"]["dialog"]))
    assert client.post("/api/session/retry/answer-receipt", json=body, headers=headers).status_code == 401
    outcomes.record_progress("retry", session_token_expires=0)
    assert not status(client, body)["can_retry"]
