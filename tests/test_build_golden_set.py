import pandas as pd
import pytest

from eval import build_golden_set as bg


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
