from support_agent import draft_reply as dr


def test_canned_covers_all_intents():
    from support_agent.taxonomy import INTENT_NAMES
    for n in INTENT_NAMES:
        assert n in dr.CANNED and dr.CANNED[n].strip()


def test_nearest_returns_retrieved_reply(monkeypatch):
    monkeypatch.setattr(dr, "retrieve",
        lambda m, k=1: [{"customer_open": "x", "spotify_reply": "Try reinstalling.", "score": 0.9}])
    assert dr.nearest_reply("app broken") == "Try reinstalling."


def test_grounded_uses_examples(monkeypatch):
    captured = {}

    def fake_gen(prompt, **k):
        captured["prompt"] = prompt
        return "Sorry to hear that — please try X."

    monkeypatch.setattr(dr.llm_client, "generate", fake_gen)
    out = dr.grounded_reply("app crashes", "technical_bug",
                             examples=[{"customer_open": "crash", "spotify_reply": "reinstall", "score": 1.0}])
    assert "reinstall" in captured["prompt"]   # grounding passed into prompt
    assert out.strip()


# --- clean_reply -----------------------------------------------------------

def test_clean_reply_strips_leading_handle():
    text = "@115887 Hmm. Can you try restarting your device? Keep us posted /LS"
    cleaned = dr.clean_reply(text)
    assert "@115887" not in cleaned
    assert cleaned.startswith("Hmm.")


def test_clean_reply_strips_mid_text_handle():
    text = "Please DM @SpotifyCares your account email so we can help."
    cleaned = dr.clean_reply(text)
    assert "@SpotifyCares" not in cleaned
    assert "your account email" in cleaned


def test_clean_reply_strips_tco_link():
    text = "We'll take a look backstage /CH https://t.co/ldFdZRiNAt"
    cleaned = dr.clean_reply(text)
    assert "t.co" not in cleaned
    # the trailing "/CH" sign-off (now right at the end, after the link is
    # stripped) is also removed -- see test_clean_reply_strips_signoff below.
    assert cleaned == "We'll take a look backstage"


def test_clean_reply_leaves_plain_text_unchanged():
    text = "Thanks for reaching out, we'll take a look."
    assert dr.clean_reply(text) == text


def test_nearest_reply_cleans_handles_and_links(monkeypatch):
    monkeypatch.setattr(dr, "retrieve", lambda m, k=1: [{
        "customer_open": "x",
        "spotify_reply": "@115887 Could you DM us? /CH https://t.co/ldFdZRiNAt",
        "score": 0.9,
    }])
    out = dr.nearest_reply("app broken")
    assert "@115887" not in out
    assert "t.co" not in out


def test_grounded_prompt_has_no_tco_link_from_examples(monkeypatch):
    captured = {}

    def fake_gen(prompt, **k):
        captured["prompt"] = prompt
        return "reply"

    monkeypatch.setattr(dr.llm_client, "generate", fake_gen)
    dr.grounded_reply("app crashes", "technical_bug", examples=[
        {"customer_open": "@SpotifyCares my app crashed",
         "spotify_reply": "@115887 Try reinstalling /LS https://t.co/abc123",
         "score": 1.0},
    ])
    assert "t.co" not in captured["prompt"]
    assert "@115887" not in captured["prompt"]
    assert "@SpotifyCares" not in captured["prompt"]


def test_grounded_prompt_instructs_no_urls_or_handles(monkeypatch):
    captured = {}

    def fake_gen(prompt, **k):
        captured["prompt"] = prompt
        return "reply"

    monkeypatch.setattr(dr.llm_client, "generate", fake_gen)
    dr.grounded_reply("app crashes", "technical_bug", examples=[
        {"customer_open": "crash", "spotify_reply": "reinstall", "score": 1.0}])
    assert "URLs" in captured["prompt"] and "@handles" in captured["prompt"]


# --- no synthetic "^S" signature; strip real agent-initials sign-offs --

def test_canned_replies_have_no_synthetic_signature():
    for intent, text in dr.CANNED.items():
        assert "^S" not in text, f"{intent} still has the synthetic ^S signature"


def test_grounded_prompt_does_not_ask_for_a_signature():
    assert "signature" not in dr._GEN_PROMPT.lower()
    assert "^S" not in dr._GEN_PROMPT


def test_clean_reply_strips_signoff():
    text = "Thanks for reaching out, we'll take a look /LS"
    assert dr.clean_reply(text) == "Thanks for reaching out, we'll take a look"


def test_clean_reply_strips_single_letter_signoff():
    text = "Give it a try and let us know /K"
    assert dr.clean_reply(text) == "Give it a try and let us know"


def test_clean_reply_preserves_24_7():
    text = "We're here for you 24/7, just ask."
    assert dr.clean_reply(text) == text


def test_clean_reply_preserves_and_or():
    text = "Try disabling and/or reinstalling the app."
    assert dr.clean_reply(text) == text


def test_clean_reply_preserves_mid_text_slash_not_at_end():
    text = "Log out and/or restart, then check for an update /RS"
    cleaned = dr.clean_reply(text)
    assert "and/or restart" in cleaned
    assert "/RS" not in cleaned


def test_grounded_reply_output_is_cleaned(monkeypatch):
    monkeypatch.setattr(dr.llm_client, "generate",
        lambda prompt, **k: "@115887 Please try reinstalling /LS https://t.co/abc123")
    out = dr.grounded_reply("app crashes", "technical_bug",
                             examples=[{"customer_open": "x", "spotify_reply": "y", "score": 1.0}])
    assert out == "Please try reinstalling"
