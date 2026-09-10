import pytest

from eval import llm_judge


def test_judge_parses_and_clamps(monkeypatch):
    monkeypatch.setattr(llm_judge.llm_client, "generate",
        lambda *a, **k: '{"grounded":7,"factual":4,"tone":5,"actionable":3,"overall":9}')
    s = llm_judge.judge_reply("m", "r", "ref")
    assert s["grounded"] == 5 and s["overall"] == 5   # clamped to 1..5
    assert s["actionable"] == 3
    assert s["factual"] == 4
    assert s["tone"] == 5


def test_judge_handles_unparseable_json(monkeypatch, caplog):
    monkeypatch.setattr(llm_judge.llm_client, "generate", lambda *a, **k: "not json at all")
    with caplog.at_level("WARNING", logger=llm_judge.logger.name):
        s = llm_judge.judge_reply("m", "r", "ref")
    assert all(s[k] == 3 for k in llm_judge.JUDGE_KEYS)
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_judge_handles_non_dict_json(monkeypatch):
    # a JSON array is valid JSON but not a dict -- must fall back to defaults,
    # not raise (e.g. from obj.get on a list).
    monkeypatch.setattr(llm_judge.llm_client, "generate", lambda *a, **k: '[1,2,3]')
    s = llm_judge.judge_reply("m", "r", "ref")
    assert all(s[k] == 3 for k in llm_judge.JUDGE_KEYS)


def test_judge_handles_missing_keys(monkeypatch):
    monkeypatch.setattr(llm_judge.llm_client, "generate",
        lambda *a, **k: '{"overall": 4}')
    s = llm_judge.judge_reply("m", "r", "ref")
    assert s["overall"] == 4
    assert s["grounded"] == 3   # default fallback for missing key


def test_judge_handles_non_numeric_value(monkeypatch):
    monkeypatch.setattr(llm_judge.llm_client, "generate",
        lambda *a, **k: '{"grounded":"n/a","factual":4,"tone":5,"actionable":3,"overall":4}')
    s = llm_judge.judge_reply("m", "r", "ref")
    assert s["grounded"] == 3   # non-numeric falls back to default


# --- A6: parse_ok flag -----------------------------------------------------

def test_judge_reports_parse_ok_true_on_valid_json(monkeypatch):
    monkeypatch.setattr(llm_judge.llm_client, "generate",
        lambda *a, **k: '{"grounded":4,"factual":4,"tone":5,"actionable":3,"overall":4}')
    s = llm_judge.judge_reply("m", "r", "ref")
    assert s["parse_ok"] is True


def test_judge_reports_parse_ok_false_on_unparseable_json(monkeypatch):
    monkeypatch.setattr(llm_judge.llm_client, "generate", lambda *a, **k: "not json at all")
    s = llm_judge.judge_reply("m", "r", "ref")
    assert s["parse_ok"] is False
    assert all(s[k] == 3 for k in llm_judge.JUDGE_KEYS)  # fallback values still clamped


def test_judge_reports_parse_ok_false_on_non_dict_json(monkeypatch):
    monkeypatch.setattr(llm_judge.llm_client, "generate", lambda *a, **k: "[1,2,3]")
    s = llm_judge.judge_reply("m", "r", "ref")
    assert s["parse_ok"] is False


def test_judge_reports_parse_ok_false_on_missing_key(monkeypatch):
    monkeypatch.setattr(llm_judge.llm_client, "generate",
        lambda *a, **k: '{"overall": 4}')
    s = llm_judge.judge_reply("m", "r", "ref")
    assert s["parse_ok"] is False
    assert s["overall"] == 4       # present key is kept as-is
    assert s["grounded"] == 3      # missing key still falls back, clamped


def test_judge_reports_parse_ok_false_on_non_numeric_value(monkeypatch):
    monkeypatch.setattr(llm_judge.llm_client, "generate",
        lambda *a, **k: '{"grounded":"n/a","factual":4,"tone":5,"actionable":3,"overall":4}')
    s = llm_judge.judge_reply("m", "r", "ref")
    assert s["parse_ok"] is False
    assert s["grounded"] == 3  # fallback


def test_judge_prompt_describes_reference_as_guide_not_answer_key():
    """Controller ruling 2: the judge prompt must frame the reference as a
    real historical reply used as a guide, not an answer key to copy, and
    must never reveal which system produced the drafted reply."""
    prompt = llm_judge._RUBRIC.format(reference="REF", message="MSG", reply="REPLY")
    assert "REF" in prompt and "MSG" in prompt and "REPLY" in prompt
    assert "answer key" in llm_judge._RUBRIC.lower()
    for banned in ("trivial", "nearest", "grounded_reply", "system under", "baseline"):
        assert banned not in llm_judge._RUBRIC.lower()
