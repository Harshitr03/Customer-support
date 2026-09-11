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
    "intent": "technical_bug",
    "confidence": 0.82,
    "reply": "Sorry about that! Please DM us your device, OS, and app version so we can dig in.",
    "escalate": False,
    "reason": "Auto-handle: 'technical_bug' with confidence 0.82, no risk signals.",
    "evidence": [
        {"customer_open": "the app keeps crashing", "spotify_reply": "please DM us your device info",
         "score": 0.83},
        {"customer_open": "songs stop after a few seconds", "spotify_reply": "try reinstalling the app",
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
