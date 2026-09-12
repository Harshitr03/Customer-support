"""Smoke tests for app/streamlit_app.py using streamlit.testing.v1.AppTest.

All hermetic: pipeline.handle is mocked or made to raise (no network), and
cache/results directories are pointed at empty tmp dirs so behavior never
depends on the real repo's data/cache (a live eval run may be writing to it
in the background) or results/ (not written yet in this worktree).
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from eval import run_eval
from support_agent import config, llm_client, pipeline
from support_agent.llm_client import OfflineModeError, QuotaExhaustedError

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

FIXED_RESULT = {
    # feature_complaint, not technical_bug: technical_bug now escalates by
    # default (sensitive_intent rule), so it can no longer stand in for an
    # auto-handled example -- this test's point is checking the UI renders
    # the verdict/reason pipeline.handle() returns, not exercising a
    # specific intent, so any genuinely auto-handling intent preserves that.
    "intent": "feature_complaint",
    "confidence": 0.82,
    "reply": "Thanks for the feedback -- we've shared it with the team. DM us any details you'd like us to pass on.",
    "escalate": False,
    "reason": "Auto-handle: 'feature_complaint' with confidence 0.82, no risk signals.",
    "evidence": [
        {"customer_open": "the new shuffle is terrible", "spotify_reply": "thanks, we've shared it with the team",
         "score": 0.83},
        {"customer_open": "bring back the old playlist UI", "spotify_reply": "noted, passing this along",
         "score": 0.71},
    ],
}


@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    """Empty cache/replay/results dirs so the app's empty states and
    cached-example count are deterministic regardless of the real repo's
    current (possibly in-progress) eval run."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "REPLAY_CACHE_DIR", tmp_path / "replay")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")


def _make_one_cached_example(tmp_path, monkeypatch, message: str, root_id: int = 1,
                              intent: str = "technical_bug"):
    """Write a one-row golden CSV plus a fully cached response for
    `message` (classify + embed + grounded reply, via the same key/prompt
    helpers eval/run_eval.py's --estimate and test_ui_data.py use), so the
    "Choose a cached example" picker has exactly one real, selectable
    option instead of just "(type your own)"."""
    golden_path = tmp_path / "golden_eval.csv"
    golden_path.write_text(f'root_id,message\n{root_id},"{message}"\n')
    monkeypatch.setattr(config, "GOLDEN_DIR", tmp_path)

    fake_examples = [{"customer_open": "cust text", "spotify_reply": "Spotify reply text", "score": 0.5}]
    fake_retrieve = lambda msg, k=4: fake_examples[:k]
    monkeypatch.setattr(run_eval.retrieve, "retrieve", fake_retrieve)
    monkeypatch.setattr(run_eval.draft_reply, "retrieve", fake_retrieve)

    ckey = run_eval._classify_cache_key(message)
    llm_client._cache_path("gen", ckey).write_text(
        json.dumps({"text": json.dumps({"intent": intent, "confidence": 0.9})}))
    ekey = llm_client._embed_cache_key(message)
    llm_client._cache_path("emb", ekey).write_text("[]")
    prompt = run_eval._grounded_prompt(message, intent, fake_examples)
    gkey = llm_client.gen_cache_key(config.GEN_MODEL, prompt, 0.3, False)
    llm_client._cache_path("gen", gkey).write_text(
        json.dumps({"text": "Thanks for reaching out, we'll look into it."}))


def _all_text(at) -> str:
    parts = []
    for coll in (at.markdown, at.caption, at.text, at.info, at.error, at.warning, at.title,
                 at.header, at.subheader):
        parts.extend(el.value for el in coll)
    return "\n".join(parts)


def test_app_renders_with_empty_cache_and_no_results_and_shows_empty_state():
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert at.exception == []
    text = _all_text(at)
    assert "No results yet" in text
    assert "run_demo.py" in text
    # Not-affiliated note is always visible.
    assert "Not affiliated with Spotify" in text


def test_run_agent_on_custom_message_shows_verdict_and_reason(monkeypatch):
    monkeypatch.setattr(pipeline, "handle", lambda message, turns=None: FIXED_RESULT)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_area(key="custom_message_input").set_value("the app keeps crashing on my phone")
    at.button(key="run_agent_button").click().run()

    assert at.exception == []
    text = _all_text(at)
    assert "Auto-reply" in text
    assert FIXED_RESULT["reason"] in text


def test_offline_mode_error_shows_friendly_message_not_traceback(monkeypatch):
    def _raise(message, turns=None):
        raise OfflineModeError("offline mode: no cached response for this call — run with --live")

    monkeypatch.setattr(pipeline, "handle", _raise)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_area(key="custom_message_input").set_value("a brand new message not in any cache")
    at.button(key="run_agent_button").click().run()

    assert at.exception == []
    text = _all_text(at)
    assert "offline cache" in text
    assert "--live" in text


def test_quota_exhausted_error_shows_its_message_not_traceback(monkeypatch):
    def _raise(message, turns=None):
        raise QuotaExhaustedError(
            "daily quota exhausted (GenerateContentPerDay). It resets at midnight Pacific time.")

    monkeypatch.setattr(pipeline, "handle", _raise)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_area(key="custom_message_input").set_value("another new message")
    at.button(key="run_agent_button").click().run()

    assert at.exception == []
    text = _all_text(at)
    assert "daily quota exhausted" in text


# ---------------------------------------------------------------------------
# Picked example vs. typed free text -- precedence (Issue 1)
#
# These drive the actual widgets through AppTest (streamlit.testing.v1),
# which runs the real script + real on_change callbacks in-process, not a
# simulated keyboard. That distinction matters here: cmd+a/Delete in a real
# browser does NOT clear Streamlit's textarea *widget state* (only what's
# on screen), so a browser-keyboard "verification" of this fix would look
# like it worked while leaving the stale session_state value in place.
# AppTest instead calls .set_value(...).run(), which goes through the same
# WidgetStateManager machinery a real interaction does (on_change fires),
# so it's a faithful check of the fix rather than a false positive.
# ---------------------------------------------------------------------------

def test_picking_a_cached_example_runs_it_when_textarea_is_empty(monkeypatch, tmp_path):
    message = "the app keeps crashing when I hit play"
    _make_one_cached_example(tmp_path, monkeypatch, message, root_id=42)
    calls = []
    monkeypatch.setattr(pipeline, "handle", lambda msg, turns=None: (calls.append(msg), FIXED_RESULT)[1])

    at = AppTest.from_file(APP_PATH)
    at.run()
    example_label = at.selectbox(key="example_picker").options[1]
    assert "42" in example_label
    at.selectbox(key="example_picker").select(example_label).run()
    # The picked example's text is now the only thing in play -- the
    # precedence rule is visible, not silent: the free-text box reads
    # empty in the widget tree, exactly as a user watching the screen
    # would see it clear on selection.
    assert at.text_area(key="custom_message_input").value == ""

    at.button(key="run_agent_button").click().run()

    assert at.exception == []
    assert calls == [message]


def test_typing_only_runs_the_typed_text(monkeypatch, tmp_path):
    _make_one_cached_example(tmp_path, monkeypatch, "some cached example message", root_id=7)
    calls = []
    monkeypatch.setattr(pipeline, "handle", lambda msg, turns=None: (calls.append(msg), FIXED_RESULT)[1])

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_area(key="custom_message_input").set_value("a message I typed myself").run()
    at.button(key="run_agent_button").click().run()

    assert at.exception == []
    assert calls == ["a message I typed myself"]


def test_typing_after_picking_overrides_the_stale_pick_most_recent_wins(monkeypatch, tmp_path):
    """Issue 1's exact repro: type something, then pick a cached example
    (which clears the box and should win), then keep typing again without
    re-touching the selectbox -- the freshest interaction (the new typed
    text) must win, not a stale value from whichever widget was touched
    first."""
    picked_message = "cached example about crashing"
    _make_one_cached_example(tmp_path, monkeypatch, picked_message, root_id=99)
    calls = []
    monkeypatch.setattr(pipeline, "handle", lambda msg, turns=None: (calls.append(msg), FIXED_RESULT)[1])

    at = AppTest.from_file(APP_PATH)
    at.run()

    # 1. Type something first (the old bug's setup).
    at.text_area(key="custom_message_input").set_value("stale typed text").run()

    # 2. Pick the cached example -- this must win over the stale typed text,
    #    and must visibly clear the textarea (not just win silently).
    example_label = at.selectbox(key="example_picker").options[1]
    at.selectbox(key="example_picker").select(example_label).run()
    assert at.text_area(key="custom_message_input").value == ""
    at.button(key="run_agent_button").click().run()
    assert at.exception == []
    assert calls == [picked_message]

    # 3. Type again without touching the selectbox again -- the fresh
    #    typed text must now win over the still-selected example.
    at.text_area(key="custom_message_input").set_value("freshly typed override").run()
    at.button(key="run_agent_button").click().run()
    assert at.exception == []
    assert calls == [picked_message, "freshly typed override"]


def test_neither_picked_nor_typed_shows_friendly_prompt_and_does_not_run(monkeypatch, tmp_path):
    _make_one_cached_example(tmp_path, monkeypatch, "some cached example message", root_id=3)
    calls = []
    monkeypatch.setattr(pipeline, "handle", lambda msg, turns=None: (calls.append(msg), FIXED_RESULT)[1])

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.button(key="run_agent_button").click().run()

    assert at.exception == []
    assert calls == []
    text = _all_text(at)
    assert "Type a message or pick a cached example first." in text
