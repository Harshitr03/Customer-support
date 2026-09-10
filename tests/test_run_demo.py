"""Tests for scripts/run_demo.py (Task 13). Every stage is mocked -- this
never touches the network, the real data/ directories, or results/."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_demo  # noqa: E402


@pytest.fixture(autouse=True)
def _offline_env_isolated():
    """run_demo.main() mutates os.environ["SUPPORT_AGENT_OFFLINE"] directly
    (that's the whole point of the flag), so plain monkeypatch.setenv/delenv
    can't be relied on to undo it -- monkeypatch only reverts changes made
    through its own API, not raw os.environ mutations performed by the code
    under test. Save/restore the real value by hand instead."""
    had = "SUPPORT_AGENT_OFFLINE" in os.environ
    original = os.environ.get("SUPPORT_AGENT_OFFLINE")
    yield
    if had:
        os.environ["SUPPORT_AGENT_OFFLINE"] = original
    else:
        os.environ.pop("SUPPORT_AGENT_OFFLINE", None)


def _fake_handle_result():
    return {"intent": "technical_bug", "confidence": 0.91, "reply": "Try reinstalling. ^S",
            "escalate": False, "reason": "Auto-handle: ok", "evidence": [{"score": 0.77}]}


def _mock_all_stages(monkeypatch, calls, golden_dir):
    monkeypatch.setattr(run_demo.config, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(run_demo.data_prep, "build_pool", lambda: calls.append("build_pool"))
    monkeypatch.setattr(run_demo.retrieve, "build_index", lambda: calls.append("build_index"))
    monkeypatch.setattr(run_demo.build_golden_set, "build", lambda: calls.append("build_golden_set"))
    monkeypatch.setattr(run_demo.run_eval, "main", lambda: calls.append("run_eval"))
    monkeypatch.setattr(run_demo.human_agreement, "main", lambda: calls.append("human_agreement"))

    def fake_handle(message, turns=None):
        calls.append("pipeline_handle")
        return _fake_handle_result()

    monkeypatch.setattr(run_demo.pipeline, "handle", fake_handle)


def test_stages_run_in_order_and_offline_is_default(tmp_path, monkeypatch):
    calls = []
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    (golden_dir / "human_scores.csv").write_text("pair_id\n1\n")
    _mock_all_stages(monkeypatch, calls, golden_dir)

    code = run_demo.main([])

    assert code == 0
    assert calls == ["build_pool", "build_index", "build_golden_set", "run_eval",
                      "human_agreement", "pipeline_handle"]
    assert os.environ.get("SUPPORT_AGENT_OFFLINE") == "1"


def test_live_flag_disables_offline(tmp_path, monkeypatch):
    calls = []
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    (golden_dir / "human_scores.csv").write_text("pair_id\n1\n")
    _mock_all_stages(monkeypatch, calls, golden_dir)

    code = run_demo.main(["--live"])

    assert code == 0
    assert os.environ.get("SUPPORT_AGENT_OFFLINE") != "1"


def test_missing_human_scores_csv_is_handled_without_calling_human_agreement(tmp_path, monkeypatch, capsys):
    calls = []
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()  # no human_scores.csv
    _mock_all_stages(monkeypatch, calls, golden_dir)

    code = run_demo.main([])

    assert code == 0
    assert "human_agreement" not in calls
    assert calls == ["build_pool", "build_index", "build_golden_set", "run_eval", "pipeline_handle"]
    out = capsys.readouterr().out
    assert "human_scores.csv" in out


def test_offline_missing_replay_entry_stops_cleanly_with_helpful_message(tmp_path, monkeypatch, capsys):
    calls = []
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    _mock_all_stages(monkeypatch, calls, golden_dir)

    def raise_offline():
        calls.append("run_eval")
        raise run_demo.OfflineModeError(
            "offline mode: no cached response for this call — run with --live")

    monkeypatch.setattr(run_demo.run_eval, "main", raise_offline)

    code = run_demo.main([])

    assert code == 1
    assert calls == ["build_pool", "build_index", "build_golden_set", "run_eval"]
    out = capsys.readouterr().out
    assert "--live" in out
    assert "eval harness" in out.lower() or "4/6" in out  # names the stage that needed it


def test_export_cache_flag_calls_export_after_live_run(tmp_path, monkeypatch):
    calls = []
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    (golden_dir / "human_scores.csv").write_text("pair_id\n1\n")
    _mock_all_stages(monkeypatch, calls, golden_dir)
    monkeypatch.setattr(run_demo, "export_touched_cache", lambda: (3, 12345))

    code = run_demo.main(["--live", "--export-cache"])

    assert code == 0


def test_export_cache_without_live_is_a_noop(tmp_path, monkeypatch, capsys):
    calls = []
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    (golden_dir / "human_scores.csv").write_text("pair_id\n1\n")
    _mock_all_stages(monkeypatch, calls, golden_dir)

    def fail_export():
        raise AssertionError("export_touched_cache must not run without --live")

    monkeypatch.setattr(run_demo, "export_touched_cache", fail_export)

    code = run_demo.main(["--export-cache"])
    assert code == 0
