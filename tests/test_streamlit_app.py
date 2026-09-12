"""Smoke tests for app/streamlit_app.py using streamlit.testing.v1.AppTest.

All hermetic: pipeline.handle is mocked or made to raise (no network), and
cache/results directories are pointed at empty tmp dirs so behavior never
depends on the real repo's data/cache (a live eval run may be writing to it
in the background) or results/ (not written yet in this worktree).
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from support_agent import config, pipeline
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
