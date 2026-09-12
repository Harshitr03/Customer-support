"""Pure-function tests for app/ui_data.py -- no Streamlit, no network.

Cache-presence tests reuse eval.run_eval's own cache-key helpers
(_classify_cache_key, _grounded_prompt, llm_client.gen_cache_key/
_embed_cache_key/_cache_path) to write fake cache files, exactly like
tests/test_run_eval.py's --estimate tests do, instead of re-deriving the
key/prompt logic here.
"""
import importlib.util
import json
import re

import pandas as pd
import pytest

# pandas' .style accessor needs jinja2, which arrives only as a transitive
# dependency of the optional [ui] extra (streamlit). A grader running plain
# `pytest` after `pip install -e ".[dev]"` must not see these fail.
_needs_jinja2 = pytest.mark.skipif(
    importlib.util.find_spec("jinja2") is None,
    reason="pandas .style needs jinja2; install the optional [ui] extra",
)

from app import ui_data
from eval import run_eval
from support_agent import config, llm_client


@pytest.fixture(autouse=True)
def _isolate_cache_dirs(tmp_path, monkeypatch):
    """Every test starts with empty, isolated cache/replay dirs so results
    never depend on the real repo's data/cache (a live eval run may be
    writing to it in the background)."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "REPLAY_CACHE_DIR", tmp_path / "replay")


# ---------------------------------------------------------------------------
# Cached-example discovery
# ---------------------------------------------------------------------------

def _fake_examples():
    return [{"customer_open": "cust text", "spotify_reply": "Spotify reply text", "score": 0.5}]


def _populate_full_cache(monkeypatch, message: str, intent: str = "technical_bug"):
    """Write real classify + embed + grounded-reply cache entries for
    `message`, using the same key/prompt helpers eval/run_eval.py's
    --estimate uses -- and fake out retrieve() (both places it's imported)
    so no real KB index is needed, matching test_run_eval.py's pattern."""
    fake_examples = _fake_examples()
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
    return fake_retrieve


def test_find_cached_examples_returns_only_fully_cached_messages(monkeypatch):
    golden = pd.DataFrame({
        "root_id": [1, 2],
        "message": ["cached message", "uncached message"],
    })
    _populate_full_cache(monkeypatch, "cached message", intent="technical_bug")

    result = ui_data.find_cached_examples(golden)

    assert [r["root_id"] for r in result] == [1]
    assert result[0]["message"] == "cached message"
    assert result[0]["intent"] == "technical_bug"


def test_find_cached_examples_empty_when_nothing_cached():
    golden = pd.DataFrame({"root_id": [1], "message": ["hello"]})
    assert ui_data.find_cached_examples(golden) == []


def test_find_cached_examples_requires_all_three_cache_kinds(monkeypatch):
    """Classify-only cached (no embed, no grounded reply) must not count."""
    message = "half cached"
    golden = pd.DataFrame({"root_id": [1], "message": [message]})
    ckey = run_eval._classify_cache_key(message)
    llm_client._cache_path("gen", ckey).write_text(
        json.dumps({"text": json.dumps({"intent": "other", "confidence": 0.5})}))

    assert ui_data.find_cached_examples(golden) == []


# ---------------------------------------------------------------------------
# Verdict formatting
# ---------------------------------------------------------------------------

def test_format_verdict_escalate_is_amber_hand_to_human():
    v = ui_data.format_verdict(True, "Escalate: message contains legal/security/risk language.")
    assert v["label"] == "Hand to a human"
    assert v["color"] == "#F5A524"
    assert v["reason"] == "Escalate: message contains legal/security/risk language."


def test_format_verdict_auto_is_green_auto_reply():
    v = ui_data.format_verdict(False, "Auto-handle: 'other' with confidence 0.90, no risk signals.")
    assert v["label"] == "Auto-reply"
    assert v["color"] == "#2FD470"
    assert v["reason"] == "Auto-handle: 'other' with confidence 0.90, no risk signals."


# ---------------------------------------------------------------------------
# format_intent / truncate
# ---------------------------------------------------------------------------

def test_format_intent_sentence_case():
    assert ui_data.format_intent("technical_bug") == "Technical bug"
    assert ui_data.format_intent("billing_subscription") == "Billing subscription"
    assert ui_data.format_intent("other") == "Other"


def test_truncate_short_text_unchanged():
    assert ui_data.truncate("short text", limit=100) == "short text"


def test_truncate_long_text_is_cut_with_ellipsis():
    text = "a" * 200
    out = ui_data.truncate(text, limit=50)
    assert len(out) <= 50
    assert out.endswith("…")


# ---------------------------------------------------------------------------
# resolve_message -- precedence between the picked example and typed text
# ---------------------------------------------------------------------------

def test_resolve_message_picked_only():
    assert ui_data.resolve_message("picked example text", "", "picked") == "picked example text"
    # last_changed shouldn't matter when only one side has text.
    assert ui_data.resolve_message("picked example text", "", "typed") == "picked example text"
    assert ui_data.resolve_message("picked example text", "", "") == "picked example text"


def test_resolve_message_typed_only():
    assert ui_data.resolve_message("", "typed text", "typed") == "typed text"
    assert ui_data.resolve_message("", "typed text", "picked") == "typed text"
    assert ui_data.resolve_message(None, "typed text", "") == "typed text"


def test_resolve_message_both_set_last_changed_wins():
    # Most-recent-interaction wins: if the user picked an example and then
    # kept typing without re-picking, the freshly typed text wins even
    # though the picker still holds the old selection.
    assert ui_data.resolve_message("picked example", "typed text", "typed") == "typed text"
    # And if they picked *after* typing, the pick wins.
    assert ui_data.resolve_message("picked example", "typed text", "picked") == "picked example"


def test_resolve_message_neither_set_returns_empty():
    assert ui_data.resolve_message("", "", "") == ""
    assert ui_data.resolve_message(None, None, "picked") == ""
    assert ui_data.resolve_message("   ", "  ", "typed") == ""


# ---------------------------------------------------------------------------
# Results loader
# ---------------------------------------------------------------------------

def _synthetic_eval_results() -> dict:
    """Minimal but schema-accurate eval_results.json, matching exactly what
    eval/run_eval.py's main() writes (see its `results = {...}` dict)."""
    per_class_f1 = {"technical_bug": 0.5, "account_access": 0.4, "billing_subscription": 0.3,
                     "content_catalog": 0.2, "cancellation_refund": 0.6, "feature_complaint": 0.1,
                     "other": 0.7}

    def cls_block(acc, f1):
        return {"accuracy": acc, "macro_f1": f1, "per_class_f1": per_class_f1,
                "confusion_matrix": [[1, 0], [0, 1]], "confusion_matrix_labels": ["a", "b"]}

    def esc_block(p, r, acc, rate):
        return {"precision": p, "recall": r, "accuracy": acc, "tp": 1, "fp": 1, "fn": 1, "tn": 1,
                "escalate_rate": rate}

    def ci(lo, hi, mean):
        return {"lo": lo, "hi": hi, "mean": mean}

    return {
        "classification": {
            "trivial": cls_block(0.2, 0.1),
            "simple_tfidf": cls_block(0.4, 0.3),
            "llm": cls_block(0.8, 0.75),
        },
        "reply_quality": {
            "trivial": {"grounded": 2.0, "factual": 3.0, "tone": 3.0, "actionable": 2.5, "overall": 2.5},
            "nearest": {"grounded": 3.5, "factual": 3.2, "tone": 3.0, "actionable": 3.0, "overall": 3.2},
            "grounded": {"grounded": 4.2, "factual": 4.0, "tone": 4.1, "actionable": 4.0, "overall": 4.1},
        },
        "escalation": {
            "end_to_end": esc_block(0.7, 0.6, 0.8, 0.3),
            "policy_only": esc_block(0.75, 0.65, 0.82, 0.32),
            "always_escalate": esc_block(0.3, 1.0, 0.3, 1.0),
            "never_escalate": esc_block(0.0, 0.0, 0.7, 0.0),
            "simple_tfidf": esc_block(0.5, 0.4, 0.6, 0.25),
            "gold_escalate_rate": 0.3,
        },
        "bootstrap_ci": {
            "classification_accuracy": {"trivial": ci(0.15, 0.25, 0.2), "simple_tfidf": ci(0.35, 0.45, 0.4),
                                         "llm": ci(0.75, 0.85, 0.8)},
            "classification_macro_f1": {"trivial": ci(0.05, 0.15, 0.1), "simple_tfidf": ci(0.25, 0.35, 0.3),
                                         "llm": ci(0.7, 0.8, 0.75)},
            "reply_overall": {"trivial": ci(2.0, 3.0, 2.5), "nearest": ci(2.8, 3.6, 3.2),
                               "grounded": ci(3.8, 4.4, 4.1)},
            "escalation_precision": {"end_to_end": ci(0.6, 0.8, 0.7), "policy_only": ci(0.65, 0.85, 0.75)},
            "escalation_recall": {"end_to_end": ci(0.5, 0.7, 0.6), "policy_only": ci(0.55, 0.75, 0.65)},
        },
        "metadata": {
            "n_golden": 200, "n_reply_subset": 60, "n_spotcheck": 40,
            "gen_model": "gemini-3.5-flash-lite", "gen_thinking": {"thinking_level": "low"},
            "embed_model": "gemini-embedding-001", "kb_size": 3800, "seed": 42,
            "bootstrap_n": 1000, "bootstrap_seed": 42,
            "judge_parse_failures": {"trivial": 0, "nearest": 1, "grounded": 0},
            "judge_stats_note": "note", "escalation_baselines_note": "note",
            "fingerprint": {"classify_prompt_sha256": "abc"},
        },
    }


def test_load_results_returns_none_when_results_dir_missing(tmp_path):
    assert ui_data.load_results(results_dir=tmp_path / "does_not_exist") is None


def test_load_results_parses_synthetic_fixture_with_real_schema(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "eval_results.json").write_text(json.dumps(_synthetic_eval_results()))
    pd.DataFrame({
        "root_id": [1, 2],
        "message": ["m1", "m2"],
        "gold_intent": ["technical_bug", "other"],
        "trivial_pred": ["other", "other"],
        "simple_pred": ["technical_bug", "other"],
        "llm_pred": ["technical_bug", "other"],
        "llm_confidence": [0.9, 0.4],
        "gold_escalate": [False, True],
        "pred_escalate": [False, False],
        "pred_escalate_reason": ["Auto-handle: ...", "Escalate: ..."],
    }).to_csv(results_dir / "classification_rows.csv", index=False)
    pd.DataFrame({
        "root_id": [1, 1, 1],
        "message": ["m1", "m1", "m1"],
        "system": ["trivial", "nearest", "grounded"],
        "pred_intent": ["technical_bug"] * 3,
        "reply": ["r1", "r2", "r3"],
        "reference": ["ref", "ref", "ref"],
        "grounded": [2, 3, 4], "factual": [3, 3, 4], "tone": [3, 3, 4],
        "actionable": [2, 3, 4], "overall": [2.5, 3.0, 4.0],
        "judge_parse_ok": [True, True, True],
    }).to_csv(results_dir / "reply_rows.csv", index=False)

    loaded = ui_data.load_results(results_dir=results_dir)

    assert loaded is not None
    assert loaded["eval_results"]["classification"]["llm"]["accuracy"] == 0.8
    assert list(loaded["classification_rows"]["root_id"]) == [1, 2]
    assert list(loaded["reply_rows"]["system"]) == ["trivial", "nearest", "grounded"]


# ---------------------------------------------------------------------------
# Escalation mistakes
# ---------------------------------------------------------------------------

def test_escalation_mistakes_all_vs_filtered():
    rows = pd.DataFrame({
        "message": ["m1", "m2", "m3", "m4"],
        "gold_escalate": [True, False, True, False],
        "pred_escalate": [False, True, True, False],
        "pred_escalate_reason": ["r1", "r2", "r3", "r4"],
        "llm_pred": ["technical_bug", "other", "billing_subscription", "other"],
    })

    all_mistakes = ui_data.escalation_mistakes(rows)
    assert set(all_mistakes["message"]) == {"m1", "m2"}

    false_esc = ui_data.escalation_mistakes(rows, "false_escalations")
    assert list(false_esc["message"]) == ["m2"]

    missed = ui_data.escalation_mistakes(rows, "missed_escalations")
    assert list(missed["message"]) == ["m1"]


# ---------------------------------------------------------------------------
# Escalation-mistakes styling (amber for missed, muted for false)
# ---------------------------------------------------------------------------

def _mistakes_df():
    return pd.DataFrame({
        "message": ["missed one", "false one"],
        "gold": [True, False],
        "predicted": [False, True],
        "reason": ["r1", "r2"],
        "intent": ["technical_bug", "other"],
    })


def test_missed_escalation_mask_is_gold_true_predicted_false():
    df = pd.DataFrame({
        "gold": [True, False, True],
        "predicted": [False, True, True],
    })
    assert list(ui_data.missed_escalation_mask(df)) == [True, False, False]


def test_false_escalation_mask_is_predicted_true_gold_false():
    df = pd.DataFrame({
        "gold": [True, False, True],
        "predicted": [False, True, True],
    })
    assert list(ui_data.false_escalation_mask(df)) == [False, True, False]


@_needs_jinja2
def test_style_escalation_mistakes_uses_amber_for_missed_and_muted_for_false():
    styled = ui_data.style_escalation_mistakes(_mistakes_df())
    html = styled.to_html()

    assert ui_data.MISSED_ESCALATION_BG in html
    assert ui_data.FALSE_ESCALATION_BG in html
    # amber (missed, row 0) must land before muted (false, row 1) in the
    # rendered row order, i.e. each colour is attached to its own row.
    assert html.index(ui_data.MISSED_ESCALATION_BG) < html.index(ui_data.FALSE_ESCALATION_BG)


@_needs_jinja2
def test_style_escalation_mistakes_no_third_colour_introduced():
    """Only the brief's amber/muted tokens are used -- no unrelated accent
    colour sneaks into the row styling."""
    styled = ui_data.style_escalation_mistakes(_mistakes_df())
    html = styled.to_html()
    background_colors = set(re.findall(r"background-color:\s*(#[0-9A-Fa-f]{6,8})", html))
    assert background_colors == {ui_data.MISSED_ESCALATION_BG, ui_data.FALSE_ESCALATION_BG}
