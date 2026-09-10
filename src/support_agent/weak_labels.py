"""Heuristic keyword weak-labeler to create training data for the simple baseline."""
import re

from .taxonomy import OTHER

KEYWORDS = {
    "cancellation_refund": ["cancel", "refund", "money back", "unsubscribe"],
    "billing_subscription": ["charged", "charge", "payment", "bill", "invoice",
                              "double charge", "paid", "declined", "currency"],
    "account_access": ["log in", "login", "log-in", "password", "hacked",
                        "can't access", "cant access", "locked out", "sign in",
                        "change my email", "change my username", "display name"],
    "technical_bug": ["crash", "crashing", "won't play", "wont play", "buffering",
                       "skip", "skipping", "not working", "keeps stopping", "offline",
                       "black screen", "battery"],
    "content_catalog": ["song", "album", "podcast", "missing", "removed",
                         "not available", "disappear", "playlist gone"],
    "feature_complaint": ["shuffle", "new update", "recommendation", "bring back",
                           "hate the", "ui", "design", "terrible"],
}
# Order matters: more specific intents first (cancel before generic billing).
_ORDER = ["cancellation_refund", "account_access", "billing_subscription",
          "technical_bug", "content_catalog", "feature_complaint"]


def weak_label(text: str) -> str:
    t = text.lower()
    for intent in _ORDER:
        for kw in KEYWORDS[intent]:
            if re.search(r"\b" + re.escape(kw), t):
                return intent
    return OTHER
