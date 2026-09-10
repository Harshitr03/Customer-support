"""Three reply-drafting systems: canned, nearest-neighbor, grounded generation."""
import logging
import re

from . import llm_client
from .retrieve import retrieve
from .taxonomy import INTENT_NAMES  # noqa: F401 (part of the documented interface)

logger = logging.getLogger(__name__)

CANNED = {
    "technical_bug": "Sorry for the trouble! Please DM us your device, OS, and app version so we can dig in. ^S",
    "account_access": "We can help you get back in. Please DM us the email on the account (never your password). ^S",
    "billing_subscription": "Thanks for flagging — please DM us the email on the account and we'll review the charge. ^S",
    "content_catalog": "Availability can vary by region/licensing. DM us the track/podcast name and your country. ^S",
    "cancellation_refund": "You can manage or cancel Premium under Account > Subscription. DM us if you need a hand. ^S",
    "feature_complaint": "Thanks for the feedback — we've shared it with the team. DM us any details you'd like us to pass on. ^S",
    "other": "Thanks for reaching out! DM us the details and we'll take a look. ^S",
}

# Historical Spotify replies are anonymized customer support tweets: they
# start with the original customer's @handle (e.g. "@115887"), sometimes
# mention @SpotifyCares mid-text, and may contain dead t.co short links.
# None of that belongs in a reply we show or feed back into a prompt.
_HANDLE_RE = re.compile(r"@\w+")
_TCO_RE = re.compile(r"https?://t\.co/\S+")
_WS_RE = re.compile(r"\s+")


def clean_reply(text: str) -> str:
    """Strip @handles and t.co links from a historical reply/message, and
    collapse the whitespace left behind."""
    text = _HANDLE_RE.sub("", text)
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
account-specific facts. Do not include URLs or @handles in your reply. End with the
agent signature " ^S".

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
    return llm_client.generate(prompt, temperature=0.3).strip()
