"""Separate service credentials from limited, expiring conversation credentials."""
import hashlib
import os
import secrets
import time


def secured():
    return bool(os.getenv("VERA_SERVICE_KEY")) or os.getenv("DEPLOYMENT_MODE", "development") != "development"


def service_authorized(authorization):
    expected = os.getenv("VERA_SERVICE_KEY", "")
    return bool(expected) and secrets.compare_digest(authorization or "", "Bearer " + expected)


def issue_session_token():
    token = secrets.token_urlsafe(32)
    return token, {"session_token_hash": hashlib.sha256(token.encode()).hexdigest(),
                   "session_token_expires": time.time() + 3600}


def session_authorized(token, record):
    if not token or record.get("session_token_expires", 0) <= time.time():
        return False
    if record.get("state") in {"completed", "declined", "withdrawn", "escalated"}:
        return False
    return secrets.compare_digest(hashlib.sha256(token.encode()).hexdigest(), record.get("session_token_hash", ""))
