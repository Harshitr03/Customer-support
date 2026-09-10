"""End-to-end: message -> (intent, reply, escalation decision)."""
import logging
import time

from .classify import llm_classify
from .retrieve import retrieve
from .draft_reply import grounded_reply
from .escalate import decide

logger = logging.getLogger(__name__)


def handle(message: str, turns: list[dict] | None = None) -> dict:
    turns = turns or []

    t0 = time.monotonic()
    intent, confidence = llm_classify(message)
    t1 = time.monotonic()
    evidence = retrieve(message, k=4)
    t2 = time.monotonic()
    reply = grounded_reply(message, intent, examples=evidence)
    t3 = time.monotonic()
    escalate, reason = decide(intent, confidence, turns, message)
    t4 = time.monotonic()

    logger.debug(
        "handle: intent=%s confidence=%.2f escalate=%s n_evidence=%d "
        "classify_ms=%.1f retrieve_ms=%.1f draft_ms=%.1f decide_ms=%.1f",
        intent, confidence, escalate, len(evidence),
        (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000,
    )

    return {"intent": intent, "confidence": confidence, "reply": reply,
            "escalate": escalate, "reason": reason, "evidence": evidence}
