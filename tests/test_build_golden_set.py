import re
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


def test_golden_labeling_notes_record_the_finished_author_review():
    """The author review happened on 2026-09-12, so the notes must record its
    outcome and no longer carry the pending-review placeholder."""
    notes = (Path(__file__).resolve().parents[1] / "eval" / "golden_labeling_notes.md").read_text()
    assert "will be recorded here" not in notes, "stale pending-review placeholder left in the notes"
    assert "## Author review" in notes
    # the stated change count must match the rows actually tabulated under it
    review = notes.split("## Author review", 1)[1]
    claimed = int(re.search(r"\*\*(\d+) rows were annotated and (\d+) gold labels changed", review).group(2))
    tabulated = len(re.findall(r"^\| \d+ \| ", review, flags=re.MULTILINE))
    assert claimed == tabulated, f"notes claim {claimed} label changes but tabulate {tabulated}"


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


def test_docs_do_not_overstate_the_author_review():
    """The review is done, but the docs must not inflate it: they may not claim
    an independent panel, and the row-43 exception must stay disclosed rather
    than smoothed away."""
    notes = _read("eval/golden_labeling_notes.md")
    log = _read("report/DECISION_LOG.md")

    # a single AI labeler plus one human reviewer is NOT an independent panel
    for doc, name in ((notes, "golden_labeling_notes.md"), (log, "DECISION_LOG.md")):
        lowered = doc.lower()
        for phrase in ("independently labelled", "independently labeled",
                        "two annotators", "panel of", "inter-annotator"):
            assert phrase not in lowered, f"{name} overstates the labeling process: {phrase!r}"

    # the deliberate exception to the bug-escalation rubric must remain documented
    assert "43" in notes and "exception" in notes.lower(), \
        "the row-43 exception to the bug-escalation rubric is no longer disclosed"


def test_readme_names_the_rows_most_likely_to_be_mislabeled():
    """The README must point a reader at the rows where the gold label disagrees
    with the cheap keyword baseline -- the ones most worth re-checking. The
    disagreement count is derived from the committed golden set, not
    hardcoded, so this doesn't go stale the next time a gold label changes."""
    readme = _read("README.md")
    golden = pd.read_csv(config.GOLDEN_DIR / "golden_eval.csv")
    n_intent_diff = int((golden["gold_intent"] != golden["pre_intent"]).sum())
    assert "gold_escalate" in readme and "pre_intent" in readme
    assert str(n_intent_diff) in readme, (
        f"README no longer names the current disagreement count ({n_intent_diff})"
    )
