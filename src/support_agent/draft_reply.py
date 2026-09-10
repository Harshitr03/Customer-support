"""Three reply-drafting systems: canned, nearest-neighbor, grounded generation."""
import logging
import re

from . import llm_client
from .retrieve import retrieve
from .taxonomy import INTENT_NAMES  # noqa: F401 (part of the documented interface)

logger = logging.getLogger(__name__)

CANNED = {
    "technical_bug": "Sorry for the trouble! Please DM us your device, OS, and app version so we can dig in.",
    "account_access": "We can help you get back in. Please DM us the email on the account (never your password).",
    "billing_subscription": "Thanks for flagging — please DM us the email on the account and we'll review the charge.",
    "content_catalog": "Availability can vary by region/licensing. DM us the track/podcast name and your country.",
    "cancellation_refund": "You can manage or cancel Premium under Account > Subscription. DM us if you need a hand.",
    "feature_complaint": "Thanks for the feedback — we've shared it with the team. DM us any details you'd like us to pass on.",
    "other": "Thanks for reaching out! DM us the details and we'll take a look.",
}

# Historical Spotify replies are anonymized customer support tweets: they
# start with the original customer's @handle (e.g. "@115887"), sometimes
# mention @SpotifyCares mid-text, may contain dead t.co short links, and
# commonly end with a two-or-three-letter agent-initials sign-off such as
# "/LS" or "/CH" (occasionally a single letter, e.g. "/K"). None of that
# belongs in a reply we show or feed back into a prompt.
_HANDLE_RE = re.compile(r"@\w+")
_TCO_RE = re.compile(r"https?://t\.co/\S+")
_WS_RE = re.compile(r"\s+")
# A trailing "/XX" sign-off: a slash plus 1-3 letters, where everything that
# follows it is at most a t.co link and/or whitespace up to the end of the
# string. The lookahead is what keeps this from matching a slash used
# mid-sentence, e.g. "24/7" or "and/or restart, then ..." — those have real
# content after the slash-word, not just an optional trailing link.
_SIGNOFF_RE = re.compile(r"/[A-Za-z]{1,3}(?=\s*(?:https?://t\.co/\S+)?\s*$)")


def clean_reply(text: str) -> str:
    """Strip @handles, a trailing agent-initials sign-off, and t.co links
    from a historical reply/message, and collapse the whitespace left
    behind."""
    text = _HANDLE_RE.sub("", text)
    text = _SIGNOFF_RE.sub("", text)
    text = _TCO_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def trivial_reply(intent: str) -> str:
    logger.debug("trivial_reply: intent=%s", intent)
    return CANNED.get(intent, CANNED["other"])


def nearest_reply(message: str) -> str:
    hits = retrieve(message, k=1)
    top_score = hits[0]["score"] if hits else float("nan")
    logger.debug("nearest_reply: n_examples=%d top_score=%.4f", len(hits), top_score)
    if not hits:
        return CANNED["other"]
    return clean_reply(hits[0]["spotify_reply"])


_GEN_PROMPT = """You are a Spotify customer-support agent on Twitter. Write ONE short,
empathetic reply (<280 chars) to the customer message below. Ground your reply in how
Spotify has historically handled similar issues (examples provided). Do not invent
account-specific facts. Do not include URLs or @handles in your reply.

Predicted intent: {intent}

Historical examples (customer -> Spotify reply):
{examples}

Customer message: {message}

Reply:"""


def grounded_reply(message: str, intent: str, examples: list[dict] | None = None) -> str:
    if examples is None:
        examples = retrieve(message, k=4)
    top_score = examples[0].get("score", float("nan")) if examples else float("nan")
    logger.debug("grounded_reply: intent=%s n_examples=%d top_score=%.4f",
                 intent, len(examples), top_score)
    ex_block = "\n".join(
        f"- {clean_reply(e['customer_open'])} -> {clean_reply(e['spotify_reply'])}"
        for e in examples) or "(none)"
    prompt = _GEN_PROMPT.format(intent=intent, examples=ex_block, message=message)
    return clean_reply(llm_client.generate(prompt, temperature=0.3).strip())
