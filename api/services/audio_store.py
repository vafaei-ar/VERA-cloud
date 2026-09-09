"""Opt-in original PCM WAV evidence with bounded retention and access audit.

The storage volume must be encrypted and backed up by the deployment owner.
Disabled by default; never treat a transcript or synthesized voice as original.
"""
import hashlib
import io
import os
import wave
from datetime import datetime, timedelta, timezone

from . import outcomes

MAX_BYTES = 10_000_000


def enabled():
    return os.getenv("AUDIO_RECORDING_ENABLED", "false").lower() == "true"


def clip_path(session_id, clip_id):
    # Use the same strict ID validation as the outcome store for both levels.
    outcomes._session_file(session_id)
    outcomes._session_file(clip_id)
    return outcomes._outcomes_dir() / "audio" / session_id / (clip_id + ".wav")


@outcomes.atomic_update
def set_consent(session_id, consent):
    if consent and not enabled():
        raise ValueError("Original audio recording is not enabled for this deployment")
    if not outcomes.outcome_exists(session_id):
        raise ValueError("Unknown session")
    record = outcomes.load_outcome(session_id)
    record["audio_consent"] = bool(consent)
    record.setdefault("audio_consent_history", []).append({"accepted": bool(consent), "at": datetime.now(timezone.utc).isoformat()})
    outcomes.save_outcome(record)
    return record


@outcomes.atomic_update
def store_clip(session_id, clip_id, content, partial=False, acknowledged_turn=False):
    if not enabled():
        raise ValueError("Original audio recording is disabled")
    record = outcomes.load_outcome(session_id)
    if record.get("consent") != "accepted" or not record.get("audio_consent"):
        raise PermissionError("Answer consent and separate audio consent are required")
    if acknowledged_turn:
        accepted_at = record.get("audio_turns", {}).get(clip_id)
        if not accepted_at:
            raise ValueError("Transcript has not been accepted for this recording")
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(accepted_at)).total_seconds()
        if not 0 <= elapsed <= 300:
            raise PermissionError("Recording upload window expired")
    if record.get("state") in {"declined", "withdrawn"} or (not acknowledged_turn and record.get("state") in {"completed", "escalated"}):
        raise PermissionError("This session has ended")
    if not 44 <= len(content) <= MAX_BYTES:
        raise ValueError("Recording exceeds the supported size")
    with wave.open(io.BytesIO(content), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or not 8000 <= source.getframerate() <= 96000:
            raise ValueError("Expected mono 16-bit PCM WAV")
        duration = source.getnframes() / source.getframerate()
        if duration > 120:
            raise ValueError("Recording exceeds 120 seconds")
        if len(source.readframes(source.getnframes())) != source.getnframes() * 2:
            raise ValueError("Recording is incomplete")
    digest = hashlib.sha256(content).hexdigest()
    clips = record.setdefault("original_audio", [])
    existing = next((clip for clip in clips if clip["id"] == clip_id), None)
    if existing:
        if existing["sha256"] != digest:
            raise ValueError("Recording ID already used for different audio")
        return existing
    if len(clips) >= 100:
        raise ValueError("Recording limit reached for this session")
    path = clip_path(session_id, clip_id)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix(".tmp")
    with open(temp, "wb") as target:
        os.chmod(temp, 0o600)
        target.write(content)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temp, path)
    days = min(30, max(1, int(os.getenv("AUDIO_RETENTION_DAYS", "7"))))
    now = datetime.now(timezone.utc)
    clip = {"id": clip_id, "sha256": digest, "duration_seconds": duration,
            "bytes": len(content), "partial": bool(partial), "source": "original_respondent_microphone",
            "respondent_role": record.get("respondent_role", "unknown"),
            "created_at": now.isoformat(), "expires_at": (now + timedelta(days=days)).isoformat()}
    clips.append(clip)
    outcomes.save_outcome(record)
    return clip


@outcomes.atomic_update
def authorized_playback(session_id, clip_id, viewer):
    record = outcomes.load_outcome(session_id)
    clip = next((c for c in record.get("original_audio", []) if c["id"] == clip_id), None)
    if not clip or datetime.fromisoformat(clip["expires_at"]) <= datetime.now(timezone.utc):
        raise FileNotFoundError("Recording unavailable or expired")
    path = clip_path(session_id, clip_id)
    if not path.is_file():
        raise FileNotFoundError("Recording unavailable")
    record.setdefault("audio_access_log", []).append({"clip_id": clip_id, "viewer": viewer,
        "at": datetime.now(timezone.utc).isoformat()})
    outcomes.save_outcome(record)
    return path


def purge_expired():
    # Scope deletion to validated clip IDs whose recorded retention has elapsed.
    now = datetime.now(timezone.utc)
    for path in outcomes._outcomes_dir().glob("*.json"):
        if path.name.startswith("_"):
            continue
        with outcomes.session_lock(path.stem):
            record = outcomes.load_outcome(path.stem)
            changed = False
            for clip in record.get("original_audio", []):
                if not clip.get("deleted_at") and datetime.fromisoformat(clip["expires_at"]) <= now:
                    clip_path(path.stem, clip["id"]).unlink(missing_ok=True)
                    clip["deleted_at"] = now.isoformat()
                    changed = True
            if changed:
                outcomes.save_outcome(record)
