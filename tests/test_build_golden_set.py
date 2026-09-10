import pandas as pd
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
