"""Trivial / simple (TF-IDF) / LLM classifiers."""
import json
import logging
import time
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import llm_client, data_prep, weak_labels, config
from .taxonomy import INTENT_NAMES, OTHER, describe

logger = logging.getLogger(__name__)


class TrivialClassifier:
    def __init__(self, majority: str):
        self.majority = majority

    def predict(self, msgs):
        return [self.majority] * len(msgs)


def fit_trivial(train_labels):
    majority = Counter(train_labels).most_common(1)[0][0]
    return TrivialClassifier(majority)


class SimpleClassifier:
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe

    def predict(self, msgs):
        return list(self.pipe.predict(msgs))

    @classmethod
    def from_weak_corpus(cls):
        start = time.monotonic()
        corpus, _ = data_prep.load_pools()
        X = corpus["customer_open"].tolist()
        y = [weak_labels.weak_label(t) for t in X]
        dist = Counter(y)
        logger.info("training SimpleClassifier: %d examples, label distribution=%s",
                    len(X), dict(dist))
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=config.SEED)),
        ])
        pipe.fit(X, y)
        elapsed = time.monotonic() - start
        logger.info("SimpleClassifier trained in %.2fs", elapsed)
        return cls(pipe)


_CLS_PROMPT = """You are an intent classifier for Spotify customer-support tweets.
Classify the message into exactly one intent from this taxonomy:
{taxonomy}

Respond ONLY with JSON: {{"intent": "<one intent name>", "confidence": <0.0-1.0>}}.

Message: {message}"""


def llm_classify(message: str):
    raw = llm_client.generate(
        _CLS_PROMPT.format(taxonomy=describe(), message=message), json_mode=True)
    try:
        obj = json.loads(raw)
        intent = str(obj.get("intent", OTHER))
        conf = float(obj.get("confidence", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("llm_classify: unparseable JSON response from LLM")
        intent, conf = OTHER, 0.0
    if intent not in INTENT_NAMES:
        logger.warning("llm_classify: unknown intent %r coerced to %r", intent, OTHER)
        intent = OTHER
    conf = max(0.0, min(1.0, conf))
    logger.debug("llm_classify: intent=%s confidence=%.2f", intent, conf)
    return intent, conf
