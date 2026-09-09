"""At-least-once, versioned server-to-server delivery; never client-directed URLs."""
import asyncio
import logging
import os

import httpx

from .outcomes import pending_outcomes, build_clinician_summary, mark_delivered

logger = logging.getLogger(__name__)


async def dispatch_once():
    url, key = os.getenv("KURA_EVENT_URL"), os.getenv("KURA_EVENT_KEY")
    if not url or not key:
        return 0
    count = 0
    async with httpx.AsyncClient(timeout=10) as client:
        for record in pending_outcomes():
            try:
                response = await client.post(url, headers={"X-Event-Key": key}, json={
                    "session_id": record["session_id"], "version": record["version"],
                    "summary": build_clinician_summary(record),
                })
                response.raise_for_status()
                receipt = response.json()
                if receipt.get("accepted_version", 0) < record["version"]:
                    raise ValueError("Receiver did not acknowledge this version")
                mark_delivered(record["session_id"], record["version"])
                count += 1
            except Exception:
                logger.exception("Outcome delivery failed for %s; retained for retry", record["session_id"])
    return count


async def run_dispatcher():
    last_purge = 0
    while True:
        try:
            import time
            from .audio_store import purge_expired
            if time.time() - last_purge >= 3600:
                try:
                    purge_expired()
                except Exception:
                    logger.exception("Recording retention pass failed; outcome delivery continues")
                last_purge = time.time()
            await dispatch_once()
        except Exception:
            logger.exception("Outbox pass failed; will retry")
        await asyncio.sleep(15)
