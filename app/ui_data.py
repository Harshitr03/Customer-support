"""Pure data functions for the Support Agent Console demo UI.

No Streamlit import here on purpose: everything in this module is plain
Python/pandas so it's testable without spinning up an app (see
tests/test_ui_data.py). app/streamlit_app.py is kept thin -- layout and
widgets only -- and calls into these functions.

Cache-presence checks reuse eval/run_eval.py's own --estimate helpers
(_classify_cache_key, _cached_llm_classify, _cached_grounded_reply, ...)
instead of re-deriving cache keys or prompt text here: those helpers are
already the source of truth for "is this call provably servable from
data/cache/ or data/llm_cache/ with zero network calls".
"""
from pathlib import Path

import pandas as pd

from eval import run_eval
from support_agent import config
from support_agent.draft_reply import clean_reply

# ---------------------------------------------------------------------------
# Copy / tokens shared with the app
# ---------------------------------------------------------------------------

NOT_AFFILIATED_NOTE = (
    "Demo built on public SpotifyCares tweets from a Kaggle dataset. "
    "Not affiliated with Spotify."
)

EMPTY_RESULTS_MESSAGE = (
    "No results yet. Run `python scripts/run_demo.py` to reproduce them "
    "offline, or `--live` to compute them."
)

OFFLINE_ERROR_MESSAGE = (
    "This message isn't in the offline cache. To run new messages, start "
    "the app with `-- --live` and set GEMINI_API_KEY (uses API quota)."
)

COLOR_GREEN = "#2FD470"
COLOR_AMBER = "#F5A524"

VERDICT_AUTO = {"label": "Auto-reply", "color": COLOR_GREEN}
VERDICT_ESCALATE = {"label": "Hand to a human", "color": COLOR_AMBER}

REPLY_CHAR_LIMIT = 280


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

def truncate(text: str, limit: int = 100) -> str:
    """Collapse whitespace and cut to one line of at most `limit` chars,
    ending with a single ellipsis character when cut."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def format_intent(intent: str) -> str:
    """Sentence-case, space-separated intent label, e.g.
    'technical_bug' -> 'Technical bug'."""
    words = (intent or "").replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else words


# ---------------------------------------------------------------------------
# Verdict formatting
# ---------------------------------------------------------------------------

def format_verdict(escalate: bool, reason: str) -> dict:
    """Map decide()'s (escalate, reason) to the verdict banner's label,
    color token, and reason text."""
    base = VERDICT_ESCALATE if escalate else VERDICT_AUTO
    return {**base, "reason": reason}


# ---------------------------------------------------------------------------
# Evidence ("Grounded on") formatting
# ---------------------------------------------------------------------------

def format_evidence(evidence: list[dict]) -> list[dict]:
    """Number retrieved evidence rows 1..N (they're rank-ordered by
    similarity) and clean historical-reply artifacts (@handles, sign-offs,
    t.co links) out of both the customer and Spotify text for display."""
    out = []
    for i, e in enumerate(evidence or [], start=1):
        out.append({
            "rank": i,
            "customer": truncate(clean_reply(e.get("customer_open", "")), 100),
            "reply": truncate(clean_reply(e.get("spotify_reply", "")), 100),
            "score": float(e.get("score", 0.0)),
        })
    return out


# ---------------------------------------------------------------------------
# Cached-example discovery (Tab 1's example picker)
# ---------------------------------------------------------------------------

def find_cached_examples(golden: pd.DataFrame) -> list[dict]:
    """Golden-set messages whose full pipeline response -- classify, the
    query embedding, and the grounded reply -- all resolve from
    data/cache/ or the committed data/llm_cache/ replay cache with no
    network call.

    Reuses eval/run_eval.py's own cache-presence helpers
    (_cached_llm_classify, _cached_grounded_reply), which in turn use
    llm_client.gen_cache_key / the embed cache key / _cache_path -- the
    exact same functions generate()/embed() use to decide a cache hit --
    so this can't drift from what a real offline run can actually serve.
    """
    out = []
    for row in golden.itertuples():
        message = row.message
        pred = run_eval._cached_llm_classify(message)
        if pred is None:
            continue
        pred_intent, _pred_conf = pred
        grounded = run_eval._cached_grounded_reply(message, pred_intent)
        if grounded is None:
            continue
        out.append({"root_id": row.root_id, "message": message, "intent": pred_intent})
    return out


def load_golden(golden_path: Path | None = None) -> pd.DataFrame:
    """Thin wrapper so the app doesn't hardcode the golden CSV path."""
    path = Path(golden_path) if golden_path is not None else config.GOLDEN_DIR / "golden_eval.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Results loader (Tab 2)
# ---------------------------------------------------------------------------

def load_results(results_dir: Path | None = None) -> dict | None:
    """Read results/eval_results.json + classification_rows.csv +
    reply_rows.csv (exact names/schema written by eval/run_eval.py's
    _write_results). Returns None -- the "no results yet" empty state --
    if results_dir (or its eval_results.json) doesn't exist. Never raises
    for a missing directory."""
    import json

    results_dir = Path(results_dir) if results_dir is not None else config.RESULTS_DIR
    results_path = results_dir / "eval_results.json"
    if not results_path.exists():
        return None
    eval_results = json.loads(results_path.read_text())
    classification_rows = pd.read_csv(results_dir / "classification_rows.csv")
    reply_rows = pd.read_csv(results_dir / "reply_rows.csv")
    return {
        "eval_results": eval_results,
        "classification_rows": classification_rows,
        "reply_rows": reply_rows,
    }


# ---------------------------------------------------------------------------
# Headline tables
# ---------------------------------------------------------------------------

def intent_headline_table(eval_results: dict) -> pd.DataFrame:
    """Accuracy and macro-F1 (with 95% CI) for trivial / TF-IDF / Gemini."""
    rows = []
    labels = {"trivial": "Trivial", "simple_tfidf": "TF-IDF", "llm": "Gemini"}
    ci = eval_results["bootstrap_ci"]
    for key, label in labels.items():
        c = eval_results["classification"][key]
        acc_ci = ci["classification_accuracy"][key]
        f1_ci = ci["classification_macro_f1"][key]
        rows.append({
            "system": label,
            "accuracy": c["accuracy"],
            "accuracy 95% CI": f"[{acc_ci['lo']:.3f}, {acc_ci['hi']:.3f}]",
            "macro F1": c["macro_f1"],
            "macro F1 95% CI": f"[{f1_ci['lo']:.3f}, {f1_ci['hi']:.3f}]",
        })
    return pd.DataFrame(rows)


def reply_quality_headline_table(eval_results: dict) -> pd.DataFrame:
    """Judge overall mean (with CI) for canned / nearest / grounded, plus
    parse failures."""
    rows = []
    labels = {"trivial": "Canned", "nearest": "Nearest", "grounded": "Grounded"}
    ci = eval_results["bootstrap_ci"]["reply_overall"]
    failures = eval_results["metadata"]["judge_parse_failures"]
    for key, label in labels.items():
        rq = eval_results["reply_quality"][key]
        c = ci[key]
        rows.append({
            "system": label,
            "judge overall": rq["overall"],
            "95% CI": f"[{c['lo']:.2f}, {c['hi']:.2f}]",
            "judge parse failures": failures.get(key, 0),
        })
    return pd.DataFrame(rows)


def escalation_headline_table(eval_results: dict) -> pd.DataFrame:
    """Precision/recall for end-to-end and gold-intent (policy-only), plus
    the always/never/TF-IDF baselines."""
    rows = []
    esc = eval_results["escalation"]
    ci = eval_results["bootstrap_ci"]
    labels = {
        "end_to_end": "End to end",
        "policy_only": "Gold intent (policy only)",
        "always_escalate": "Always escalate",
        "never_escalate": "Never escalate",
        "simple_tfidf": "TF-IDF baseline",
    }
    for key, label in labels.items():
        e = esc[key]
        row = {"variant": label, "precision": e["precision"], "recall": e["recall"],
               "escalate rate": e["escalate_rate"]}
        if key in ("end_to_end", "policy_only"):
            p_ci = ci["escalation_precision"][key]
            r_ci = ci["escalation_recall"][key]
            row["precision 95% CI"] = f"[{p_ci['lo']:.3f}, {p_ci['hi']:.3f}]"
            row["recall 95% CI"] = f"[{r_ci['lo']:.3f}, {r_ci['hi']:.3f}]"
        else:
            row["precision 95% CI"] = ""
            row["recall 95% CI"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def per_intent_f1(eval_results: dict) -> pd.DataFrame:
    """Per-intent F1 for the main (Gemini) classifier, for a horizontal
    bar chart."""
    per_class = eval_results["classification"]["llm"]["per_class_f1"]
    rows = [{"intent": format_intent(k), "f1": v} for k, v in per_class.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compare replies
# ---------------------------------------------------------------------------

def compare_reply_options(reply_rows: pd.DataFrame) -> list[dict]:
    """One entry per distinct message in the reply-quality subset, for a
    picker -- root_id plus a truncated preview."""
    seen = reply_rows.drop_duplicates(subset=["root_id"])
    return [{"root_id": r.root_id, "message": truncate(r.message, 80)} for r in seen.itertuples()]


def compare_replies(reply_rows: pd.DataFrame, root_id) -> dict:
    """The three systems' replies for one message, each with its judge
    scores, plus the shared reference (Spotify's actual reply)."""
    rows = reply_rows[reply_rows["root_id"] == root_id]
    judge_keys = ("grounded", "factual", "tone", "actionable", "overall")
    by_system = {}
    reference = None
    message = None
    for r in rows.itertuples():
        message = r.message
        reference = r.reference
        by_system[r.system] = {
            "reply": r.reply,
            "scores": {k: getattr(r, k) for k in judge_keys},
        }
    return {"message": message, "reference": reference, "systems": by_system}


# ---------------------------------------------------------------------------
# Escalation mistakes
# ---------------------------------------------------------------------------

def escalation_mistakes(classification_rows: pd.DataFrame, mistake_filter: str = "all") -> pd.DataFrame:
    """Rows where predicted escalation != gold, optionally narrowed to
    false escalations (predicted True, gold False) or missed escalations
    (predicted False, gold True). Columns: message, gold, predicted,
    reason, intent."""
    mismatched = classification_rows[
        classification_rows["gold_escalate"] != classification_rows["pred_escalate"]]
    if mistake_filter == "false_escalations":
        mismatched = mismatched[mismatched["pred_escalate"] & ~mismatched["gold_escalate"]]
    elif mistake_filter == "missed_escalations":
        mismatched = mismatched[~mismatched["pred_escalate"] & mismatched["gold_escalate"]]
    return mismatched[["message", "gold_escalate", "pred_escalate", "pred_escalate_reason", "llm_pred"]].rename(
        columns={"gold_escalate": "gold", "pred_escalate": "predicted",
                 "pred_escalate_reason": "reason", "llm_pred": "intent"}
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Escalation-mistakes styling (missed vs. false escalations)
# ---------------------------------------------------------------------------

# Amber and muted at ~25% opacity (8-digit hex alpha) so the row highlight
# sits on top of the dark table background without needing a third accent
# colour -- the brief's palette is green/amber/muted only, and amber is
# already "hand to a human" elsewhere in the app.
MISSED_ESCALATION_BG = "#F5A52440"
FALSE_ESCALATION_BG = "#9BA3AE40"


def missed_escalation_mask(df: pd.DataFrame) -> pd.Series:
    """True where gold says escalate but the agent predicted auto -- the
    case the brief's amber token is reserved for. Expects `gold` and
    `predicted` boolean columns (escalation_mistakes()'s output schema)."""
    return df["gold"] & ~df["predicted"]


def false_escalation_mask(df: pd.DataFrame) -> pd.Series:
    """True where the agent predicted escalate but gold says auto."""
    return df["predicted"] & ~df["gold"]


def style_escalation_mistakes(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Row-highlight the escalation-mistakes table: amber for missed
    escalations, muted grey for false escalations. Returns a pandas Styler
    -- st.dataframe(styler) renders it natively, no extra dependency."""
    missed = missed_escalation_mask(df)
    false_esc = false_escalation_mask(df)

    def _row_style(row: pd.Series) -> list[str]:
        if missed.loc[row.name]:
            return [f"background-color: {MISSED_ESCALATION_BG}"] * len(row)
        if false_esc.loc[row.name]:
            return [f"background-color: {FALSE_ESCALATION_BG}"] * len(row)
        return [""] * len(row)

    return df.style.apply(_row_style, axis=1)
