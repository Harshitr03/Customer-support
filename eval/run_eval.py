"""Automated eval harness: classification, escalation, and reply-quality
metrics over the hand-labeled golden set, plus a network-free --estimate
mode that reports exactly how many API calls a real run would make.

Controller rulings applied here (see task-11-brief.md for the full spec,
and the ruling list in the task prompt for what overrides it):

  1. The trivial classifier baseline is fit on corpus weak labels
     (`fit_trivial_on_corpus`), never on the golden test labels.
  2. The judge's reference reply is the REAL Spotify reply from the golden
     message's own thread (`build_reference_map`, joined by root_id) --
     not nearest-neighbor retrieval, which would make the nearest system's
     reply and its own judging reference identical.
  3. The reply-quality subset is the 40 in_spotcheck rows plus the first 20
     non-spotcheck rows in file order (`select_reply_subset`) = 60 rows.
  4. Reply systems use the LLM-PREDICTED intent end-to-end, not gold_intent.
  5. `llm_classify` is called once per golden message and reused for
     classification metrics, reply drafting, and escalation.
  6. Escalation is scored twice: end-to-end (predicted intent/confidence)
     and policy-only (gold intent, confidence=1.0) -- isolating rule-policy
     error from classifier error (`compute_escalation_variant`).
  7. Outputs go to the committed `results/` directory: eval_results.json,
     classification_rows.csv, reply_rows.csv, human_scoring_template.csv.
  8. `--estimate` (`main(estimate_only=True)`) prints planned API-call
     counts without ever calling the network -- see `estimate_calls`.
"""
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix as sk_confusion_matrix,
    f1_score, precision_score, recall_score,
)

from support_agent import classify, config, data_prep, draft_reply, llm_client, retrieve, weak_labels
from support_agent.escalate import decide
from support_agent.taxonomy import INTENT_NAMES
from eval.llm_judge import JUDGE_KEYS, judge_reply
from eval.llm_judge import _RUBRIC as _JUDGE_RUBRIC

logger = logging.getLogger(__name__)

N_EXTRA_NONSPOTCHECK = 20
REPLY_SYSTEMS = ("trivial", "nearest", "grounded")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
_BOOTSTRAP_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, labels=None) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {lbl: float(f) for lbl, f in zip(labels, per_class)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(labels),
    }


def escalation_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_ci(statistic_fn, n: int, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """95% CI for statistic_fn(idx) over `n_boot` resamples of `n` indices
    drawn with replacement. Deterministic for a fixed (n, n_boot, seed)."""
    rng = np.random.RandomState(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        vals[b] = statistic_fn(idx)
    lo, hi = np.percentile(vals, [100 * _BOOTSTRAP_ALPHA / 2, 100 * (1 - _BOOTSTRAP_ALPHA / 2)])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(vals))}


# ---------------------------------------------------------------------------
# Data prep helpers
# ---------------------------------------------------------------------------

def load_golden() -> pd.DataFrame:
    return pd.read_csv(config.GOLDEN_DIR / "golden_eval.csv")


def select_reply_subset(golden: pd.DataFrame, n_extra: int | None = None) -> pd.DataFrame:
    """Ruling 3: all in_spotcheck rows + the first `n_extra` non-spotcheck
    rows in original file order. Returned in original file order.

    n_extra defaults to the module-level N_EXTRA_NONSPOTCHECK, read at call
    time (not bound at def-time) so tests can monkeypatch the module
    constant and have main() pick it up without passing it explicitly."""
    if n_extra is None:
        n_extra = N_EXTRA_NONSPOTCHECK
    spot_idx = golden.index[golden["in_spotcheck"]]
    non_spot_idx = golden.index[~golden["in_spotcheck"]][:n_extra]
    idx = spot_idx.union(non_spot_idx).sort_values()
    return golden.loc[idx].reset_index(drop=True)


def build_reference_map(golden: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    """Ruling 2: the judge reference is the real Spotify reply from the
    golden message's own thread, joined by root_id -- never a
    nearest-neighbor retrieval result."""
    merged = golden[["root_id"]].merge(
        eval_df[["root_id", "spotify_reply"]], on="root_id", how="left")
    missing = merged.loc[merged["spotify_reply"].isna(), "root_id"].tolist()
    if missing:
        raise ValueError(
            f"eval_df has no spotify_reply for root_id(s) {missing}; "
            "golden root_ids must be a subset of eval_df.")
    return dict(zip(merged["root_id"], merged["spotify_reply"].map(draft_reply.clean_reply)))


def fit_trivial_on_corpus(corpus: pd.DataFrame):
    """Ruling 1: fit the trivial baseline on corpus weak labels, never on
    the golden test labels (fitting on test labels would leak the test set
    into the baseline)."""
    weak = [weak_labels.weak_label(t) for t in corpus["customer_open"]]
    return classify.fit_trivial(weak)


def compute_escalation_variant(golden: pd.DataFrame, intents: list, confidences: list):
    """decide() has no memory of prior turns for golden-set evaluation
    (turns=[] always), matching the brief's escalation eval."""
    preds, reasons = [], []
    for message, intent, conf in zip(golden["message"], intents, confidences):
        esc, reason = decide(intent, conf, [], message)
        preds.append(esc)
        reasons.append(reason)
    return preds, reasons


def build_human_scoring_template(spotcheck: pd.DataFrame, reply_rows: pd.DataFrame) -> pd.DataFrame:
    """Ruling 7d: the 40 in_spotcheck rows, one system per row in rotation
    (trivial, nearest, grounded, trivial, ...), for a human to score
    independently against judge_overall (Task 12: judge/human agreement)."""
    recs = []
    for i, row in enumerate(spotcheck.itertuples()):
        system = REPLY_SYSTEMS[i % len(REPLY_SYSTEMS)]
        match = reply_rows[(reply_rows["root_id"] == row.root_id) & (reply_rows["system"] == system)]
        if match.empty:
            raise ValueError(f"no reply_rows entry for root_id={row.root_id} system={system}")
        m = match.iloc[0]
        recs.append({
            "pair_id": i + 1,
            "root_id": row.root_id,
            "message": row.message,
            "system": system,
            "reply": m["reply"],
            "reference": m["reference"],
            "judge_overall": m["overall"],
            "human_overall": "",
        })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# --estimate mode: exact/upper-bound API-call counts, zero network calls
# ---------------------------------------------------------------------------

def _classify_cache_key(message: str) -> str:
    prompt = classify._CLS_PROMPT.format(taxonomy=classify.describe(), message=message)
    return json.dumps({"m": config.GEN_MODEL, "p": prompt, "t": 0.2, "j": True})


def _is_gen_cached(key: str) -> bool:
    return llm_client._cache_path("gen", key).exists()


def _is_embed_cached(text: str) -> bool:
    key = llm_client._embed_cache_key(text)
    return llm_client._cache_path("emb", key).exists()


def _cached_llm_classify(message: str):
    """Return (intent, confidence) if llm_classify's cache is already
    populated for this exact message, else None. Only ever reads the disk
    cache -- calling llm_classify here is safe because we've already proven
    the cache entry exists, so it cannot reach the network."""
    if not _is_gen_cached(_classify_cache_key(message)):
        return None
    return classify.llm_classify(message)


def _cached_retrieve(message: str, k: int):
    """Same idea as _cached_llm_classify: only calls retrieve() (which
    embeds the query) once the embed cache is proven to hold this message,
    so it never reaches the network."""
    if not _is_embed_cached(message):
        return None
    return retrieve.retrieve(message, k=k)


def _grounded_prompt(message: str, pred_intent: str, examples: list) -> str:
    ex_block = "\n".join(
        f"- {draft_reply.clean_reply(e['customer_open'])} -> {draft_reply.clean_reply(e['spotify_reply'])}"
        for e in examples) or "(none)"
    return draft_reply._GEN_PROMPT.format(intent=pred_intent, examples=ex_block, message=message)


def _cached_grounded_reply(message: str, pred_intent):
    """Return the grounded reply text if it's provably already cached
    (prediction known + query embedding cached + the exact generate() call
    for that prompt is cached), else None. Never touches the network."""
    if pred_intent is None:
        return None
    examples = _cached_retrieve(message, k=4)
    if examples is None:
        return None
    prompt = _grounded_prompt(message, pred_intent, examples)
    key = json.dumps({"m": config.GEN_MODEL, "p": prompt, "t": 0.3, "j": False})
    if not _is_gen_cached(key):
        return None
    return draft_reply.grounded_reply(message, pred_intent, examples=examples)


def _is_judge_cached(message: str, reply: str, reference: str) -> bool:
    prompt = _JUDGE_RUBRIC.format(reference=reference, message=message, reply=reply)
    key = json.dumps({"m": config.GEN_MODEL, "p": prompt, "t": 0.0, "j": True})
    return _is_gen_cached(key)


def estimate_calls(golden: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    """Plan the API calls a real run would make, counting exactly how many
    are already cached wherever that's cheaply provable, without ever
    calling the network. See module docstring, ruling 8."""
    messages = golden["message"].tolist()
    n_classify_cached = sum(1 for m in messages if _is_gen_cached(_classify_cache_key(m)))
    classification = {
        "total": len(messages),
        "cached": n_classify_cached,
        "uncached": len(messages) - n_classify_cached,
    }

    subset = select_reply_subset(golden)
    unique_subset_msgs = list(dict.fromkeys(subset["message"]))
    n_embed_cached = sum(1 for m in unique_subset_msgs if _is_embed_cached(m))
    embeddings = {
        "total_unique_messages": len(unique_subset_msgs),
        "cached": n_embed_cached,
        "uncached": len(unique_subset_msgs) - n_embed_cached,
    }

    reference_map = build_reference_map(subset, eval_df)

    n_grounded_cached = 0
    n_judge_cached = 0
    n_rows = len(subset)
    for row in subset.itertuples():
        pred = _cached_llm_classify(row.message)
        pred_intent = pred[0] if pred else None
        reference = reference_map[row.root_id]

        grounded_text = _cached_grounded_reply(row.message, pred_intent)
        if grounded_text is not None:
            n_grounded_cached += 1
            if _is_judge_cached(row.message, grounded_text, reference):
                n_judge_cached += 1

        if pred_intent is not None:
            trivial_text = draft_reply.trivial_reply(pred_intent)  # pure, no network
            if _is_judge_cached(row.message, trivial_text, reference):
                n_judge_cached += 1

        hits = _cached_retrieve(row.message, k=1)
        if hits is not None:
            nearest_text = draft_reply.nearest_reply(row.message)  # embed cache hit -> safe
            if _is_judge_cached(row.message, nearest_text, reference):
                n_judge_cached += 1

    grounded_replies = {
        "upper_bound": n_rows,
        "provably_cached": n_grounded_cached,
        "uncached_upper_bound": n_rows - n_grounded_cached,
    }
    judge_upper_bound = n_rows * len(REPLY_SYSTEMS)
    judge_calls = {
        "upper_bound": judge_upper_bound,
        "provably_cached": n_judge_cached,
        "uncached_upper_bound": judge_upper_bound - n_judge_cached,
    }

    return {
        "estimate": True,
        "classification": classification,
        "embeddings": embeddings,
        "grounded_replies": grounded_replies,
        "judge_calls": judge_calls,
    }


def _print_estimate(est: dict) -> None:
    print("\n=== --estimate: planned API calls (no network calls made) ===")
    c = est["classification"]
    print(f"  classification prompts : {c['uncached']:4d} uncached / {c['total']} total (LLM generate)")
    e = est["embeddings"]
    print(f"  query embeddings        : {e['uncached']:4d} uncached / {e['total_unique_messages']} total (embed)")
    g = est["grounded_replies"]
    print(f"  grounded replies        : <= {g['uncached_upper_bound']} uncached "
          f"(upper bound {g['upper_bound']}, {g['provably_cached']} provably cached)")
    j = est["judge_calls"]
    print(f"  judge calls             : <= {j['uncached_upper_bound']} uncached "
          f"(upper bound {j['upper_bound']}, {j['provably_cached']} provably cached)")


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def _write_results(results: dict, classification_rows: pd.DataFrame,
                    reply_rows: pd.DataFrame, human_template: pd.DataFrame) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
    classification_rows.to_csv(config.RESULTS_DIR / "classification_rows.csv", index=False)
    reply_rows.to_csv(config.RESULTS_DIR / "reply_rows.csv", index=False)
    human_template.to_csv(config.RESULTS_DIR / "human_scoring_template.csv", index=False)
    logger.info("wrote results to %s", config.RESULTS_DIR)


def _print_table(r: dict) -> None:
    print("\n=== Classification (accuracy / macro-F1) ===")
    for k, v in r["classification"].items():
        ci_acc = r["bootstrap_ci"]["classification_accuracy"][k]
        ci_f1 = r["bootstrap_ci"]["classification_macro_f1"][k]
        print(f"  {k:14s} acc={v['accuracy']:.3f} [{ci_acc['lo']:.3f},{ci_acc['hi']:.3f}]  "
              f"macroF1={v['macro_f1']:.3f} [{ci_f1['lo']:.3f},{ci_f1['hi']:.3f}]")
    print("\n=== Reply quality (judge overall, 1-5) ===")
    for k, v in r["reply_quality"].items():
        ci = r["bootstrap_ci"]["reply_overall"][k]
        print(f"  {k:10s} overall={v['overall']:.2f} [{ci['lo']:.2f},{ci['hi']:.2f}]")
    print("\n=== Escalation ===")
    for variant in ("end_to_end", "policy_only"):
        e = r["escalation"][variant]
        cip = r["bootstrap_ci"]["escalation_precision"][variant]
        cir = r["bootstrap_ci"]["escalation_recall"][variant]
        print(f"  {variant:12s} precision={e['precision']:.3f} [{cip['lo']:.3f},{cip['hi']:.3f}]  "
              f"recall={e['recall']:.3f} [{cir['lo']:.3f},{cir['hi']:.3f}]  "
              f"acc={e['accuracy']:.3f}  tp={e['tp']} fp={e['fp']} fn={e['fn']} tn={e['tn']}")


def main(estimate_only: bool = False) -> dict:
    t_start = time.monotonic()
    golden = load_golden()
    corpus, eval_df = data_prep.load_pools()
    logger.info("loaded golden set: %d rows (%d in_spotcheck)",
                len(golden), int(golden["in_spotcheck"].sum()))

    if estimate_only:
        est = estimate_calls(golden, eval_df)
        _print_estimate(est)
        logger.info("estimate complete in %.2fs, no network calls made", time.monotonic() - t_start)
        return est

    # --- classify once, reuse everywhere (ruling 5) ---------------------
    t0 = time.monotonic()
    messages = golden["message"].tolist()
    root_ids = golden["root_id"].tolist()
    llm_preds = []
    for root_id, m in zip(root_ids, messages):
        pred = classify.llm_classify(m)
        logger.debug("classify: root_id=%s intent=%s confidence=%.2f", root_id, pred[0], pred[1])
        llm_preds.append(pred)
    pred_intents = [p[0] for p in llm_preds]
    pred_confs = [p[1] for p in llm_preds]
    logger.info("classified %d golden messages via llm_classify in %.1fs",
                len(messages), time.monotonic() - t0)

    y_true = golden["gold_intent"].tolist()
    trivial_clf = fit_trivial_on_corpus(corpus)
    simple_clf = classify.SimpleClassifier.from_weak_corpus()
    trivial_preds = trivial_clf.predict(messages)
    simple_preds = simple_clf.predict(messages)

    classification = {
        "trivial": classification_metrics(y_true, trivial_preds, labels=INTENT_NAMES),
        "simple_tfidf": classification_metrics(y_true, simple_preds, labels=INTENT_NAMES),
        "llm": classification_metrics(y_true, pred_intents, labels=INTENT_NAMES),
    }
    logger.info("classification metrics: trivial acc=%.3f simple acc=%.3f llm acc=%.3f",
                classification["trivial"]["accuracy"], classification["simple_tfidf"]["accuracy"],
                classification["llm"]["accuracy"])

    # --- escalation, twice (ruling 6) ------------------------------------
    gold_escalate = golden["gold_escalate"].astype(bool).tolist()
    e2e_pred, e2e_reason = compute_escalation_variant(golden, pred_intents, pred_confs)
    policy_pred, policy_reason = compute_escalation_variant(
        golden, y_true, [1.0] * len(golden))
    escalation = {
        "end_to_end": escalation_metrics(gold_escalate, e2e_pred),
        "policy_only": escalation_metrics(gold_escalate, policy_pred),
    }
    logger.info("escalation: end_to_end precision=%.3f recall=%.3f | policy_only precision=%.3f recall=%.3f",
                escalation["end_to_end"]["precision"], escalation["end_to_end"]["recall"],
                escalation["policy_only"]["precision"], escalation["policy_only"]["recall"])

    # --- reply-quality subset (ruling 3, 4) -------------------------------
    subset = select_reply_subset(golden)
    root_to_pred = dict(zip(golden["root_id"], pred_intents))
    sub_pred_intents = [root_to_pred[rid] for rid in subset["root_id"]]

    reference_map = build_reference_map(subset, eval_df)
    t1 = time.monotonic()
    # A2: pre-embed the whole reply-subset in ONE batched call. Each query
    # would otherwise be embedded one-at-a-time inside nearest_reply's and
    # grounded_reply's own retrieve() call, each paced a full
    # EMBED_BATCH_INTERVAL_S apart -- about an hour for 60 messages. Doing
    # it here means retrieve() below hits the per-text cache instead.
    llm_client.embed(subset["message"].tolist())
    trivial_replies = [draft_reply.trivial_reply(pi) for pi in sub_pred_intents]
    nearest_replies = [draft_reply.nearest_reply(m) for m in subset["message"]]
    grounded_replies = [draft_reply.grounded_reply(m, pi)
                         for m, pi in zip(subset["message"], sub_pred_intents)]
    logger.info("drafted replies for %d-row subset x 3 systems in %.1fs",
                len(subset), time.monotonic() - t1)

    replies_by_system = {"trivial": trivial_replies, "nearest": nearest_replies, "grounded": grounded_replies}

    t2 = time.monotonic()
    reply_rows_recs = []
    judge_scores_by_system = {s: [] for s in REPLY_SYSTEMS}
    for system in REPLY_SYSTEMS:
        for row, pi, reply in zip(subset.itertuples(), sub_pred_intents, replies_by_system[system]):
            reference = reference_map[row.root_id]
            scores = judge_reply(row.message, reply, reference)
            logger.debug("judge: root_id=%s system=%s scores=%s", row.root_id, system, scores)
            judge_scores_by_system[system].append(scores)
            reply_rows_recs.append({
                "root_id": row.root_id, "message": row.message, "system": system,
                "pred_intent": pi, "reply": reply, "reference": reference,
                **scores,
            })
    logger.info("judged %d (row, system) pairs in %.1fs", len(reply_rows_recs), time.monotonic() - t2)
    reply_rows = pd.DataFrame(reply_rows_recs)

    reply_quality = {
        system: {k: float(np.mean([s[k] for s in judge_scores_by_system[system]])) for k in JUDGE_KEYS}
        for system in REPLY_SYSTEMS
    }

    # --- bootstrap CIs (95%, 1000 resamples, seed 42) ---------------------
    y_true_arr = np.array(y_true)
    pred_arrs = {"trivial": np.array(trivial_preds), "simple_tfidf": np.array(simple_preds),
                 "llm": np.array(pred_intents)}
    gold_escalate_arr = np.array(gold_escalate)
    e2e_pred_arr, policy_pred_arr = np.array(e2e_pred), np.array(policy_pred)

    def acc_stat(preds):
        return lambda idx: float(accuracy_score(y_true_arr[idx], preds[idx]))

    def f1_stat(preds):
        return lambda idx: float(f1_score(y_true_arr[idx], preds[idx], labels=INTENT_NAMES,
                                           average="macro", zero_division=0))

    def escalation_stat(preds, key):
        def stat(idx):
            m = escalation_metrics(gold_escalate_arr[idx].tolist(), preds[idx].tolist())
            return m[key]
        return stat

    n_golden = len(golden)
    bootstrap_ci_block = {
        "classification_accuracy": {
            k: bootstrap_ci(acc_stat(v), n_golden) for k, v in pred_arrs.items()},
        "classification_macro_f1": {
            k: bootstrap_ci(f1_stat(v), n_golden) for k, v in pred_arrs.items()},
        "reply_overall": {
            system: bootstrap_ci(
                lambda idx, s=system: float(np.mean(
                    [judge_scores_by_system[s][i]["overall"] for i in idx])),
                len(subset))
            for system in REPLY_SYSTEMS},
        "escalation_precision": {
            "end_to_end": bootstrap_ci(escalation_stat(e2e_pred_arr, "precision"), n_golden),
            "policy_only": bootstrap_ci(escalation_stat(policy_pred_arr, "precision"), n_golden),
        },
        "escalation_recall": {
            "end_to_end": bootstrap_ci(escalation_stat(e2e_pred_arr, "recall"), n_golden),
            "policy_only": bootstrap_ci(escalation_stat(policy_pred_arr, "recall"), n_golden),
        },
    }

    metadata = {
        "n_golden": n_golden,
        "n_reply_subset": len(subset),
        "n_spotcheck": int(subset["in_spotcheck"].sum()),
        "gen_model": config.GEN_MODEL,
        "embed_model": config.EMBED_MODEL,
        "kb_size": config.KB_SIZE,
        "seed": config.SEED,
        "bootstrap_n": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    results = {
        "classification": classification,
        "reply_quality": reply_quality,
        "escalation": escalation,
        "bootstrap_ci": bootstrap_ci_block,
        "metadata": metadata,
    }

    # --- per-row output tables (ruling 7) ----------------------------------
    classification_rows = pd.DataFrame({
        "root_id": golden["root_id"],
        "message": golden["message"],
        "gold_intent": y_true,
        "trivial_pred": trivial_preds,
        "simple_pred": simple_preds,
        "llm_pred": pred_intents,
        "llm_confidence": pred_confs,
        "gold_escalate": gold_escalate,
        "pred_escalate": e2e_pred,
        "pred_escalate_reason": e2e_reason,
    })

    spotcheck = subset[subset["in_spotcheck"]].reset_index(drop=True)
    human_template = build_human_scoring_template(spotcheck, reply_rows)

    _write_results(results, classification_rows, reply_rows, human_template)
    _print_table(results)
    logger.info("full eval run complete in %.1fs", time.monotonic() - t_start)
    return results


if __name__ == "__main__":
    config.setup_logging("INFO")
    main(estimate_only="--estimate" in sys.argv)
