"""Auditable rule-based escalation policy. Every path returns a reason string.

This is a deliberately small, transparent rule layer (not an LLM call) so the
decision to escalate a message to a human is easy to audit and explain.
"""
import logging
import re

logger = logging.getLogger(__name__)

# Renamed from the original taxonomy names (playback_bug -> technical_bug,
# account_login -> account_access) after reviewing real data.
#
# technical_bug escalates by default too: a support agent can rarely resolve
# a bug report from the first reply alone, so routing it straight to a human
# beats drafting a reply likely to need a follow-up anyway.
ESCALATE_INTENTS = {"technical_bug", "account_access", "billing_subscription", "cancellation_refund"}
CONF_THRESHOLD = 0.55
MAX_UNRESOLVED_TURNS = 6  # 3+ back-and-forth rounds

# Word boundaries wrap each alternative so short words (e.g. "sue", "legal")
# don't match inside unrelated words. That trailing \b is exactly what makes a
# *bare stem* alternative fail to match its own inflected forms: \bfraud\b
# demands a word boundary immediately after "...fraud", but "fraudulent"
# continues with more letters right there, so no boundary exists and the
# alternative can never match "fraudulent" -- same mechanism as the original
# "unauthoriz" bug (fixed previously by spelling out "unauthori[sz]ed").
#
# Round-1 fix only patched "unauthoriz"; this pass audited every remaining
# alternative for the same stem/word-boundary shape and found it recurring in:
#   - fraud       -> fraud(ulent)?                  ("a fraudulent charge")
#   - scam        -> scam(s|med|ming|mer|mers)?     ("they scammed me")
#   - sue         -> su(e|ed|es|ing)                ("I am suing you")
#   - dispute     -> disput(e|es|ed|ing)             ("I disputed the charge")
#   - chargeback  -> charge[ds]?\s*back              ("I charged back the payment")
#   - lawsuit     -> lawsuit(s)?                     (plural "lawsuits")
# lawsuit, legal, gdpr, police, stolen, and hack(ed|ing)? were re-checked and
# either already enumerate their suffix forms (hack) or have no plausible
# customer inflection beyond the bare word (legal, gdpr, police, stolen).
_RISK = re.compile(r"\b(su(e|ed|es|ing)|lawsuit(s)?|legal|fraud(ulent)?|"
                   r"scam(s|med|ming|mer|mers)?|gdpr|police|hack(ed|ing)?|"
                   r"stolen|unauthori[sz]ed|disput(e|es|ed|ing)|"
                   r"charge[ds]?\s*back)\b", re.I)

# Mined from the 5,400-thread history (corpus_pool), not the golden set.
# On non-billing, non-account messages, Spotify's actual reply asked for a DM
# or the account email 64% of the time when the customer signaled exhaustion
# ("tried everything", "already tried", "still happening") vs. 29% baseline
# and 26% for a bare single-fix mention ("I reinstalled"). So this rule fires
# on exhaustion language only, not on the bare mention of one fix.
_TRIED_FIXES = re.compile(
    r"\b(tried everything|already tried|i['’]?ve tried|have tried|tried all|nothing works|"
    r"no success|no luck|still (happening|occurring|not working|doesn['’]?t work|"
    r"isn['’]?t working|won['’]?t \w+|the same|getting)|same (issue|problem|error))\b",
    re.I,
)


def decide(intent: str, confidence: float, turns: list[dict], message: str) -> tuple[bool, str]:
    """Decide whether to escalate a message to a human, with a human-readable reason."""
    if _RISK.search(message or ""):
        escalate, reason = True, "Escalate: message contains legal/security/risk language."
        rule = "risk_language"
    elif intent in ESCALATE_INTENTS:
        escalate, reason = True, f"Escalate: '{intent}' is a sensitive intent (technical bug/account/billing/cancellation)."
        rule = "sensitive_intent"
    elif _TRIED_FIXES.search(message or ""):
        escalate, reason = True, "Escalate: customer reports standard troubleshooting already failed."
        rule = "tried_fixes"
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
