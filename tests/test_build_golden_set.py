from pathlib import Path

import pandas as pd
import pytest

from eval import build_golden_set as bg
from support_agent import config


def test_stratified_respects_min(monkeypatch):
    df = pd.DataFrame({"root_id": range(100),
                       "customer_open": [f"msg {i}" for i in range(100)],
                       "spotify_reply": ["r"]*100})
    # fake weak labeler: alternate two real intent names
    seq = iter(["technical_bug" if i % 2 else "billing_subscription"
                for i in range(100)])
    monkeypatch.setattr(bg, "weak_label", lambda m: next(seq))
    out = bg.stratified_sample(df, size=20, min_per_intent=5)
    counts = out["pre_intent"].value_counts()
    assert len(out) == 20
    assert counts.min() >= 5


def test_stratified_raises_when_min_exceeds_size(monkeypatch):
    # 3 intents x min_per_intent=10 = 30 guaranteed rows, but size=20: the
    # guaranteed per-intent block alone can't fit inside the requested size,
    # so the final .sample() would silently drop some intent below its
    # minimum. That must raise instead of silently under-filling.
    df = pd.DataFrame({"root_id": range(90),
                       "customer_open": [f"msg {i}" for i in range(90)],
                       "spotify_reply": ["r"]*90})
    intents = ["technical_bug", "billing_subscription", "account_access"]
    seq = iter([intents[i % 3] for i in range(90)])
    monkeypatch.setattr(bg, "weak_label", lambda m: next(seq))
    with pytest.raises(ValueError, match="exceeds"):
        bg.stratified_sample(df, size=20, min_per_intent=10)


def test_in_spotcheck_is_a_simple_random_sample_not_stratified():
    """eval/golden_labeling_notes.md documents in_spotcheck as a seeded
    simple random sample of 40, not stratified by intent. Guard that claim
    directly against build()'s actual sampling call -- pandas' plain
    DataFrame.sample() (no `weights`/grouping) is simple random sampling."""
    import inspect
    normalized = " ".join(inspect.getsource(bg.build).split())
    assert "g.sample(min(40, len(g)), random_state=config.SEED)" in normalized


# ---------------------------------------------------------------------------
# eval/golden_labeling_notes.md's "Labeling results" numbers must match
# the actual (frozen, committed) golden set -- a regression guard against
# the doc silently drifting from the data.
# ---------------------------------------------------------------------------

def test_golden_labeling_notes_numbers_match_the_committed_golden_set():
    golden = pd.read_csv(config.GOLDEN_DIR / "golden_eval.csv")
    notes = (Path(__file__).resolve().parents[1] / "eval" / "golden_labeling_notes.md").read_text()

    n_intent_diff = int((golden["gold_intent"] != golden["pre_intent"]).sum())
    n_escalate_diff = int((golden["gold_escalate"] != golden["pre_escalate"]).sum())
    n_escalate_rate = int(golden["gold_escalate"].sum())

    assert f"{n_intent_diff} of\n  200" in notes or f"{n_intent_diff} of 200" in notes.replace("\n", " ")
    assert f"{n_escalate_diff} of\n  200" in notes or f"{n_escalate_diff} of 200" in notes.replace("\n", " ")
    assert f"{n_escalate_rate} of 200" in notes.replace("\n", " ")

    counts = golden["gold_intent"].value_counts()
    for intent, count in counts.items():
        assert f"| {intent} | {count} |" in notes


def test_golden_labeling_notes_quotes_decision_log_item_7_not_escalate_py():
    notes = (Path(__file__).resolve().parents[1] / "eval" / "golden_labeling_notes.md").read_text()
    log = (Path(__file__).resolve().parents[1] / "report" / "DECISION_LOG.md").read_text()

    # item 7's own sentence, verbatim, must appear in the quoted rubric.
    assert 'What "escalate" means in the golden set.' in log
    assert 'What "escalate" means in the golden set.' in notes
    assert "cancelling an account the customer can't reach" in notes

    # explicitly disclaims the circular "just read escalate.py" shortcut
    assert "circular" in notes.lower()


def test_golden_labeling_notes_has_author_review_placeholder():
    notes = (Path(__file__).resolve().parents[1] / "eval" / "golden_labeling_notes.md").read_text()
    assert ("The author reviews every row before the evaluation is run; the "
            "number of labels changed will be recorded here.") in notes


# ---------------------------------------------------------------------------
# README.md, eval/golden_labeling_notes.md, and report/DECISION_LOG.md
# item 4 must describe the SAME labeling process, truthfully -- Claude
# drafts against the rubric, the author reviews before the eval runs, and
# the review has not (yet) happened.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path):
    return (_REPO_ROOT / rel_path).read_text()


def test_labeling_process_described_consistently_across_docs():
    readme = _read("README.md")
    notes = _read("eval/golden_labeling_notes.md")
    log = _read("report/DECISION_LOG.md")

    for doc in (readme, notes, log):
        assert "Claude" in doc
        assert "review" in doc.lower()

    # DECISION_LOG item 4 is the one the brief points at specifically.
    assert "4. **The golden set is sampled with free keyword labels" in log


def test_docs_do_not_claim_the_author_review_has_already_happened():
    readme = _read("README.md")
    notes = _read("eval/golden_labeling_notes.md")
    log = _read("report/DECISION_LOG.md")

    # past-tense claims that would assert the review is DONE
    banned_phrases = ("author-reviewed", "has been reviewed", "was reviewed",
                       "author has reviewed")
    for doc, name in ((readme, "README.md"), (notes, "golden_labeling_notes.md"),
                       (log, "DECISION_LOG.md")):
        lowered = doc.lower()
        for phrase in banned_phrases:
            assert phrase not in lowered, f"{name} claims the review already happened: {phrase!r}"

    # the notes file's own placeholder must still read as not-yet-done
    assert "will be recorded here" in notes


def test_readme_names_the_spotcheck_subset_before_defending_the_set():
    readme = _read("README.md")
    assert "65" in readme and "gold_escalate" in readme
    assert "101" in readme and "pre_intent" in readme
