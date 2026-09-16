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
        "integrations (e.g. Hulu linking, sharing to social media). Distinguish from "
        "billing_subscription: an error, stuck screen, or failed action occurring specifically "
        "during a payment, signup, or plan-change flow (e.g. a premium purchase that won't "
        "activate, a family-invite form rejecting a valid email) is billing_subscription, not "
        "technical_bug, even though it looks like a malfunction; use technical_bug only for "
        "problems in ordinary app usage unrelated to a billing or account-plan action.",
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
        "plan changes (monthly/yearly/family), regional pricing/currency confusion, and any error "
        "or failed action that occurs specifically while paying, signing up, or changing/joining a "
        "plan (e.g. a broken family-invite form, a premium signup that won't complete) -- classify "
        "these as billing_subscription even though they look like a bug. Classify by "
        "the customer's primary ask: if they ask to cancel, get a refund, or get their money "
        "back, use cancellation_refund even if a charge is mentioned; use billing_subscription "
        "only when no cancellation or refund is requested.",
        ("paid for Premium but still seeing ads",
         "why was I charged twice, I didn't request anything",
         "I'm being charged in the wrong currency for my country",
         "my card got declined but I have funds",
         "I get an error trying to switch from monthly to yearly premium")),
    Intent("content_catalog",
        "Missing/removed songs, albums, podcasts, wrong metadata, or a specific song/album/podcast "
        "blocked or unavailable in the customer's region while they otherwise have Spotify access. "
        "Does not include Spotify (the service) not being launched or available in a country at "
        "all -- that is other.",
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
        "meta-complaints about support responsiveness, artist/creator-side issues, or requests/"
        "complaints that Spotify itself isn't available yet in the customer's country.",
        ("you guys are awesome",
         "why aren't you responding to my question",
         "how do I get hired by Spotify",
         "when will Spotify finally launch in my country")),
]

INTENT_NAMES = [i.name for i in INTENTS]


def describe() -> str:
    lines = []
    for it in INTENTS:
        ex = " | ".join(it.examples)
        lines.append(f"- {it.name}: {it.definition} Examples: {ex}")
    return "\n".join(lines)
