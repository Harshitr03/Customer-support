"""Auditable rule-based escalation policy. Every path returns a reason string.

This is a deliberately small, transparent rule layer (not an LLM call) so the
decision to escalate a message to a human is easy to audit and explain.
"""
import logging
import re

logger = logging.getLogger(__name__)

# Renamed from the original taxonomy names (playback_bug -> technical_bug,
# account_login -> account_access) after Task 3 reviewed real data.
ESCALATE_INTENTS = {"account_access", "billing_subscription", "cancellation_refund"}
CONF_THRESHOLD = 0.55
MAX_UNRESOLVED_TURNS = 6  # 3+ back-and-forth rounds

# Word boundaries wrap each alternative so short words (e.g. "sue", "legal")
# don't match inside unrelated words. Every alternative here is a complete
# word/phrase, not a partial stem, EXCEPT the fixed case below: the original
# reference regex used the bare stem "unauthoriz" with a trailing \b, which
# requires a word boundary immediately after "...unauthoriz" -- but the
# customer phrase is "unauthorized"/"unauthorised", where letters ("ed"/"ised")
# immediately follow the stem, so no boundary exists there and the pattern
# could never match. Fixed by spelling out the full word with both spellings.
_RISK = re.compile(r"\b(sue|lawsuit|legal|fraud|scam|gdpr|police|hack(ed|ing)?|"
                   r"stolen|unauthori[sz]ed|dispute|chargeback)\b", re.I)


def decide(intent: str, confidence: float, turns: list[dict], message: str) -> tuple[bool, str]:
    """Decide whether to escalate a message to a human, with a human-readable reason."""
    if _RISK.search(message or ""):
        escalate, reason = True, "Escalate: message contains legal/security/risk language."
        rule = "risk_language"
    elif intent in ESCALATE_INTENTS:
        escalate, reason = True, f"Escalate: '{intent}' is a sensitive intent (account/billing/cancellation)."
        rule = "sensitive_intent"
    elif confidence < CONF_THRESHOLD:
        escalate, reason = True, f"Escalate: classifier confidence {confidence:.2f} below {CONF_THRESHOLD}."
        rule = "low_confidence"
    elif turns and len(turns) >= MAX_UNRESOLVED_TURNS:
        escalate, reason = True, "Escalate: long unresolved back-and-forth thread (3+ rounds)."
        rule = "long_thread"
    else:
        escalate, reason = False, f"Auto-handle: '{intent}' with confidence {confidence:.2f}, no risk signals."
        rule = "auto_handle"

    logger.debug("decide: intent=%s confidence=%.2f escalate=%s rule=%s",
                 intent, confidence, escalate, rule)
    return escalate, reason
