"""Session-wide test hermeticity (A7).

support_agent.config calls load_dotenv() at import time, and this worktree
sits inside a checkout that has a real GEMINI_API_KEY in the parent
checkout's .env -- config.load_dotenv() picks it up before pytest ever
gets a chance to intervene. Without this fixture, a test that forgets to
mock the network wouldn't fail loudly; it would just quietly spend real
quota. This fixture makes every test start from the same hermetic baseline
regardless of what's in the environment or .env: no real key, offline by
default (any accidental real call raises OfflineModeError instead of
reaching the network).

Test modules that need to exercise the with-key / online paths explicitly
override this per test via monkeypatch.setenv (e.g. tests/test_llm_client.py
resets SUPPORT_AGENT_OFFLINE for its own replay/offline-guard tests, and
tests/test_run_demo.py's --live tests set a fake GEMINI_API_KEY) -- those
overrides run after this fixture within the same test and are unaffected by
it.
"""
import pytest


@pytest.fixture(autouse=True)
def _hermetic_llm_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SUPPORT_AGENT_OFFLINE", "1")
