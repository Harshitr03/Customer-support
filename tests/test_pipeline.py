from support_agent import pipeline


def test_handle_shape(monkeypatch):
    monkeypatch.setattr(pipeline, "llm_classify", lambda m: ("technical_bug", 0.9))
    monkeypatch.setattr(pipeline, "retrieve",
        lambda m, k=4: [{"customer_open": "crash", "spotify_reply": "reinstall", "score": 1.0}])
    monkeypatch.setattr(pipeline, "grounded_reply", lambda m, i, examples=None: "Try X. ^S")
    monkeypatch.setattr(pipeline, "decide", lambda intent, confidence, turns, message: (False, "Auto-handle: ok"))
    out = pipeline.handle("app keeps crashing")
    assert out["intent"] == "technical_bug"
    assert out["reply"].endswith("^S")
    assert out["escalate"] is False
    assert out["evidence"] and "reason" in out


def test_handle_escalate_path(monkeypatch):
    monkeypatch.setattr(pipeline, "llm_classify", lambda m: ("billing_subscription", 0.95))
    monkeypatch.setattr(pipeline, "retrieve",
        lambda m, k=4: [{"customer_open": "charge", "spotify_reply": "we'll look into it", "score": 0.8}])
    monkeypatch.setattr(pipeline, "grounded_reply", lambda m, i, examples=None: "We'll review. ^S")
    monkeypatch.setattr(pipeline, "decide",
        lambda intent, confidence, turns, message: (True, "Escalate: 'billing_subscription' is a sensitive intent."))
    out = pipeline.handle("why was I charged twice")
    assert out["escalate"] is True
    assert out["reason"]
