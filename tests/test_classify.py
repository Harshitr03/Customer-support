import logging

import pandas as pd
import pytest

from support_agent import classify


def test_trivial_predicts_majority():
    clf = classify.fit_trivial(["billing_subscription", "billing_subscription", "other"])
    assert clf.predict(["x", "y"]) == ["billing_subscription", "billing_subscription"]


def test_llm_classify_coerces_label(monkeypatch):
    monkeypatch.setattr(classify.llm_client, "generate",
        lambda *a, **k: '{"intent":"not_a_real_intent","confidence":0.9}')
    intent, conf = classify.llm_classify("hi")
    assert intent == "other"          # unknown label coerced to other
    assert 0.0 <= conf <= 1.0


def test_llm_classify_returns_valid_intent_and_confidence(monkeypatch):
    monkeypatch.setattr(classify.llm_client, "generate",
        lambda *a, **k: '{"intent":"billing_subscription","confidence":0.75}')
    intent, conf = classify.llm_classify("I was charged twice")
    assert intent == "billing_subscription"
    assert conf == pytest.approx(0.75)


def test_llm_classify_handles_unparseable_json(monkeypatch, caplog):
    monkeypatch.setattr(classify.llm_client, "generate", lambda *a, **k: "not json at all")
    with caplog.at_level("WARNING", logger=classify.logger.name):
        intent, conf = classify.llm_classify("hi")
    assert intent == "other"
    assert conf == 0.0
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_llm_classify_warns_on_unknown_intent(monkeypatch, caplog):
    monkeypatch.setattr(classify.llm_client, "generate",
        lambda *a, **k: '{"intent":"not_a_real_intent","confidence":0.9}')
    with caplog.at_level("WARNING", logger=classify.logger.name):
        classify.llm_classify("hi")
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_llm_classify_confidence_is_clamped(monkeypatch):
    monkeypatch.setattr(classify.llm_client, "generate",
        lambda *a, **k: '{"intent":"other","confidence":5.0}')
    intent, conf = classify.llm_classify("hi")
    assert conf == 1.0


def test_simple_classifier_from_weak_corpus(monkeypatch, caplog):
    fake_corpus = pd.DataFrame({
        "customer_open": [
            "I was charged twice for my subscription",
            "please refund my money and cancel my plan",
            "how do I cancel my subscription today",
            "the app keeps crashing every time I open it",
            "songs keep buffering and skipping on my phone",
            "random unrelated hello there",
        ] * 3,
    })
    monkeypatch.setattr(classify.data_prep, "load_pools",
        lambda: (fake_corpus, pd.DataFrame()))
    with caplog.at_level("INFO", logger=classify.logger.name):
        clf = classify.SimpleClassifier.from_weak_corpus()
    preds = clf.predict(["I want a refund", "random hello"])
    assert all(p in classify.INTENT_NAMES for p in preds)
    assert any(r.levelname == "INFO" for r in caplog.records)
