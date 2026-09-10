"""Intent taxonomy for SpotifyCares — single source of truth."""
from dataclasses import dataclass

OTHER = "other"


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str
    examples: tuple[str, ...]


INTENTS: list[Intent] = [
    Intent("technical_bug",
        "App or website not working as intended: crashes, songs won't play, skipping/buffering, "
        "offline errors, excessive battery/memory/CPU use, broken links/buttons, broken "
        "integrations (e.g. Hulu linking, sharing to social media).",
        ("the app keeps crashing when I hit play",
         "songs stop after 10 seconds on my phone",
         "my app's memory usage keeps growing the longer I use it",
         "the app shows a black screen and an internet error even though my internet is fine")),
    Intent("account_access",
        "Cannot access or manage account: login failures, password reset, hacked/compromised "
        "account, and account-detail changes such as email, username, or display name.",
        ("I can't log into my account",
         "someone hacked my Spotify account",
         "I need to change the name on my profile",
         "someone else is logging into my account and playing music")),
    Intent("billing_subscription",
        "Charges, payment methods, Premium not active after paying, unexpected/duplicate charges, "
        "plan changes (monthly/yearly/family), regional pricing/currency confusion. Classify by "
        "the customer's primary ask: if they ask to cancel, get a refund, or get their money "
        "back, use cancellation_refund even if a charge is mentioned; use billing_subscription "
        "only when no cancellation or refund is requested.",
        ("paid for Premium but still seeing ads",
         "why was I charged twice, I didn't request anything",
         "I'm being charged in the wrong currency for my country",
         "my card got declined but I have funds")),
    Intent("content_catalog",
        "Missing/removed songs, albums, podcasts, wrong metadata, regional availability of content.",
        ("why did my favorite album disappear",
         "this podcast isn't available in my country",
         "a song's title is showing up wrong in the app")),
    Intent("cancellation_refund",
        "Wants to cancel Premium or get a refund/money back for a charge. Applies whenever "
        "cancelling or refunding is the customer's primary ask, even if a charge or billing "
        "issue is mentioned as the reason (e.g. 'I was charged twice, refund me').",
        ("how do I cancel my subscription",
         "I want a refund for last month",
         "I was charged twice for Premium, refund me")),
    Intent("feature_complaint",
        "Dislikes a feature/UX change, shuffle behavior, recommendations, or requests a new "
        "feature; general feedback/complaint about the product. Distinguish from technical_bug "
        "by whether the app is malfunctioning (technical_bug) versus working as designed but "
        "disliked, or missing a capability the customer wants added (feature_complaint).",
        ("the new shuffle is terrible",
         "bring back the old playlist UI",
         "please add an alarm clock feature to the app")),
    Intent(OTHER,
        "Anything that does not fit the above, including greetings, praise, off-topic messages, "
        "meta-complaints about support responsiveness, or artist/creator-side issues.",
        ("you guys are awesome",
         "why aren't you responding to my question",
         "how do I get hired by Spotify")),
]

INTENT_NAMES = [i.name for i in INTENTS]


def describe() -> str:
    lines = []
    for it in INTENTS:
        ex = " | ".join(it.examples)
        lines.append(f"- {it.name}: {it.definition} Examples: {ex}")
    return "\n".join(lines)
