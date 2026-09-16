"""Agreement between the LLM judge and a human rater on reply overall scores.

This module computes evidence of how well the LLM judge agrees with a
human, from `data/golden/human_scores.csv`, the filled-in blind scoring
sheet: a human rater filled in `human_overall` for
`results/human_scoring_blind.csv`'s 120 (message, reply) pairs, which
carries neither the drafting system's name nor the judge's own score for
that pair -- see `results/human_scoring_rubric.md` for the rubric they
scored against. This module re-attaches that identity from
`results/human_scoring_key.csv` (which system/root_id each item_id
actually was) and the judge's own score from `results/reply_rows.csv`,
purely for analysis; the human never saw either.

Design decisions worth calling out:

  1. Input shape: main() reads the filled blind sheet (item_id, message,
     reply, reference, human_overall), joins it against
     results/human_scoring_key.csv (item_id -> pair_id, root_id, system)
     and results/reply_rows.csv (root_id, system -> judge_overall's
     "overall" column) -- see join_human_scores(). Every item_id must join
     exactly once on both sides; validated at this boundary with clear
     errors, not silently dropped/duplicated rows.
  2. Metrics: agreement() returns n, cohen_kappa_binned (low 1-2 / mid 3 /
     high 4-5), weighted_kappa_quadratic (raw 1-5, quadratic weights),
     spearman, exact_agreement, within_one, mean_human, mean_judge, and
     mean_diff (judge minus human, i.e. judge bias). Degenerate inputs
     (e.g. constant scores) yield NaN, never an exception or a warning.
  3. Also: per-system breakdown, a 3x3 binned confusion table, and a
     bootstrap 95% CI for cohen_kappa_binned (1000 paired row resamples,
     seed 42; resamples where kappa is undefined are skipped and counted).
     Per-system agreement (~13 pairs each) is the stricter read: pooling
     across three systems of different quality inflates agreement with
     between-system score spread that has nothing to do with whether the
     judge and the human actually agree on any one reply -- see the
     "note" field in the written JSON.
  4. Output: results/judge_human_agreement.json (overall + per-system +
     confusion + CI + note) plus a concise printed summary.
  6. Logging: module logger, INFO summary lines only, never message/reply
     text at INFO+; setup_logging('INFO') only under __main__.
  7. No network/API calls -- none are needed for this task.
"""
import json
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from support_agent import config

logger = logging.getLogger(__name__)

REQUIRED_HUMAN_SCORES_COLUMNS = ("item_id", "message", "reply", "reference", "human_overall")
REQUIRED_KEY_COLUMNS = ("item_id", "pair_id", "root_id", "system")
BIN_LABELS = ("low", "mid", "high")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = config.SEED
_BOOTSTRAP_ALPHA = 0.05
POOLED_VS_PER_SYSTEM_NOTE = (
    "Agreement pooled across all three systems (\"overall\" above) is "
    "inflated relative to any one system's agreement: the three systems "
    "differ in quality, so the pooled correlation partly reflects the "
    "judge and the human both noticing that a grounded-generation reply "
    "beats a canned one, not that they agree on any single reply's score. "
    "Per-system agreement (~13 pairs each, in \"per_system\" below) is the "
    "stricter test -- it holds quality roughly constant within each group."
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _bin(score: int) -> int:
    """low:1-2 -> 0, mid:3 -> 1, high:4-5 -> 2."""
    return 0 if score <= 2 else (1 if score == 3 else 2)


def _safe_kappa(a, b, **kwargs) -> float:
    """cohen_kappa_score, returning NaN (never raising, never warning) when
    the statistic is undefined -- e.g. every label identical across both
    raters, which makes sklearn compute 0/0 internally."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = cohen_kappa_score(a, b, **kwargs)
    return float(val)


def _safe_spearman(human, judge) -> float:
    """spearmanr's correlation, returning NaN (never raising, never
    warning) when undefined -- fewer than 2 points, or zero variance in
    either series."""
    if len(human) < 2 or len(set(human)) == 1 or len(set(judge)) == 1:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(human, judge).correlation
    return float(rho) if rho is not None else float("nan")


def agreement(human: list[int], judge: list[int]) -> dict:
    """Agreement metrics between a human rater's and the LLM judge's
    `overall` scores (1-5 integers) on the same (message, reply) pairs."""
    n = len(human)
    hb = [_bin(x) for x in human]
    jb = [_bin(x) for x in judge]

    diffs = [j - h for h, j in zip(human, judge)]

    return {
        "n": n,
        "cohen_kappa_binned": _safe_kappa(hb, jb, labels=[0, 1, 2]),
        "weighted_kappa_quadratic": _safe_kappa(
            human, judge, weights="quadratic", labels=[1, 2, 3, 4, 5]),
        "spearman": _safe_spearman(human, judge),
        "exact_agreement": sum(1 for d in diffs if d == 0) / n,
        "within_one": sum(1 for d in diffs if abs(d) <= 1) / n,
        "mean_human": float(np.mean(human)),
        "mean_judge": float(np.mean(judge)),
        "mean_diff": float(np.mean(diffs)),
    }


def confusion_table(human: list[int], judge: list[int]) -> dict:
    """3x3 confusion table of binned scores: rows = human bin, cols = judge
    bin, in low/mid/high order."""
    table = [[0, 0, 0] for _ in range(3)]
    for h, j in zip(human, judge):
        table[_bin(h)][_bin(j)] += 1
    return {"labels": list(BIN_LABELS), "table": table}


def bootstrap_kappa_ci(human: list[int], judge: list[int],
                        n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """95% CI for cohen_kappa_binned over `n_boot` paired resamples of rows
    (same row index drawn for both human and judge, with replacement).
    Deterministic for a fixed (human, judge, n_boot, seed). Resamples where
    the binned kappa is undefined (e.g. all-one-bin) are skipped; the count
    skipped is reported rather than silently dropped."""
    n = len(human)
    human_arr = np.array(human)
    judge_arr = np.array(judge)
    rng = np.random.RandomState(seed)
    vals = []
    n_skipped = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        hb = [_bin(int(x)) for x in human_arr[idx]]
        jb = [_bin(int(x)) for x in judge_arr[idx]]
        k = _safe_kappa(hb, jb, labels=[0, 1, 2])
        if k != k:  # NaN
            n_skipped += 1
            continue
        vals.append(k)

    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "mean": float("nan"),
                "n_boot": n_boot, "n_skipped": n_skipped}

    lo, hi = np.percentile(vals, [100 * _BOOTSTRAP_ALPHA / 2, 100 * (1 - _BOOTSTRAP_ALPHA / 2)])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(vals)),
            "n_boot": n_boot, "n_skipped": n_skipped}


# ---------------------------------------------------------------------------
# main(): load, validate, compute, write, print
# ---------------------------------------------------------------------------

def _validate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Validate + coerce the human_overall and judge_overall columns:
    integers in 1..5, no blanks. Returns df with both columns coerced to
    int (a column round-tripped through CSV with a blank elsewhere reads
    back as float64 even where every value is integral, e.g. 4.0). Raises
    ValueError with a specific message otherwise."""
    df = df.copy()
    human_col = df["human_overall"]
    blank = human_col.isna()
    if human_col.dtype == object:
        blank = blank | (human_col.astype(str).str.strip() == "")
    n_blank = int(blank.sum())
    if n_blank:
        raise ValueError(
            f"{n_blank} row(s) of data/golden/human_scores.csv have a blank "
            "human_overall score -- fill in every row before computing agreement.")

    for col in ("human_overall", "judge_overall"):
        try:
            numeric = pd.to_numeric(df[col])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"column {col!r} must contain integers: {exc}") from exc

        non_integer = numeric[numeric % 1 != 0]
        if len(non_integer):
            raise ValueError(
                f"column {col!r} must contain only integer scores, found "
                f"{len(non_integer)} non-integer value(s): {non_integer.tolist()}")

        df[col] = numeric.astype(int)
        bad = df.loc[~df[col].between(1, 5), col].tolist()
        if bad:
            raise ValueError(
                f"column {col!r} must be in the 1..5 range, found out-of-range "
                f"value(s): {bad}")
    return df


def join_human_scores(human_df: pd.DataFrame, key_df: pd.DataFrame,
                       reply_rows: pd.DataFrame) -> pd.DataFrame:
    """Re-attach the drafting system and the judge's own score to the
    filled-in blind sheet, purely for analysis -- the human rater saw
    neither. `human_df` is the filled results/human_scoring_blind.csv
    (item_id, message, reply, reference, human_overall); `key_df` is
    results/human_scoring_key.csv (item_id, pair_id, root_id, system);
    `reply_rows` is results/reply_rows.csv (has root_id, system, and an
    "overall" column -- the judge's score for that (root_id, system)).

    Every item_id must join exactly once against key_df, and every
    resulting (root_id, system) must join exactly once against
    reply_rows -- raises ValueError with the specific mismatch otherwise
    (never silently drops or duplicates a row). Returns a DataFrame with
    columns (pair_id, root_id, message, system, reply, reference,
    judge_overall, human_overall), NOT yet score-range-validated (see
    _validate_scores)."""
    missing_cols = [c for c in REQUIRED_HUMAN_SCORES_COLUMNS if c not in human_df.columns]
    if missing_cols:
        raise ValueError(
            f"data/golden/human_scores.csv is missing required column(s): {missing_cols}")
    missing_key_cols = [c for c in REQUIRED_KEY_COLUMNS if c not in key_df.columns]
    if missing_key_cols:
        raise ValueError(
            f"results/human_scoring_key.csv is missing required column(s): {missing_key_cols}")

    dup_human = human_df.loc[human_df["item_id"].duplicated(), "item_id"].tolist()
    if dup_human:
        raise ValueError(
            f"data/golden/human_scores.csv has duplicate item_id(s): {dup_human}")
    dup_key = key_df.loc[key_df["item_id"].duplicated(), "item_id"].tolist()
    if dup_key:
        raise ValueError(
            f"results/human_scoring_key.csv has duplicate item_id(s): {dup_key}")

    human_ids, key_ids = set(human_df["item_id"]), set(key_df["item_id"])
    only_in_human = sorted(human_ids - key_ids)
    if only_in_human:
        raise ValueError(
            "item_id(s) in data/golden/human_scores.csv not found in "
            f"results/human_scoring_key.csv: {only_in_human}")
    only_in_key = sorted(key_ids - human_ids)
    if only_in_key:
        raise ValueError(
            "item_id(s) in results/human_scoring_key.csv missing from "
            "data/golden/human_scores.csv -- every distributed item must "
            f"come back scored: {only_in_key}")

    merged = human_df.merge(key_df, on="item_id", how="inner")
    if len(merged) != len(human_df):
        raise ValueError(
            "item_id join between data/golden/human_scores.csv and "
            "results/human_scoring_key.csv was not one-to-one")

    dup_reply_keys = (
        reply_rows.loc[reply_rows.duplicated(subset=["root_id", "system"]), ["root_id", "system"]]
        .apply(tuple, axis=1).tolist()
    )
    if dup_reply_keys:
        raise ValueError(
            f"results/reply_rows.csv has duplicate (root_id, system) row(s): {dup_reply_keys}")

    judge_lookup = reply_rows.set_index(["root_id", "system"])["overall"]
    keys = list(zip(merged["root_id"], merged["system"]))
    missing_in_replies = sorted(set(k for k in keys if k not in judge_lookup.index))
    if missing_in_replies:
        raise ValueError(
            f"no results/reply_rows.csv entry for (root_id, system): {missing_in_replies}")

    merged["judge_overall"] = [judge_lookup.loc[k] for k in keys]
    return merged[["pair_id", "root_id", "message", "system", "reply",
                    "reference", "judge_overall", "human_overall"]]


def load_scores() -> pd.DataFrame | None:
    """Load data/golden/human_scores.csv (the filled-in blind sheet), join
    it against results/human_scoring_key.csv and results/reply_rows.csv
    (see join_human_scores), and validate the result. Returns None (after
    printing how to produce whichever piece is missing) if any of the
    three required files doesn't exist yet."""
    human_path = config.GOLDEN_DIR / "human_scores.csv"
    key_path = config.RESULTS_DIR / "human_scoring_key.csv"
    reply_rows_path = config.RESULTS_DIR / "reply_rows.csv"

    missing_paths = [p for p in (human_path, key_path, reply_rows_path) if not p.exists()]
    if missing_paths:
        print(
            "Missing input(s) for judge/human agreement: "
            + ", ".join(str(p) for p in missing_paths) + ".\n"
            "To produce them:\n"
            "  1. Run the eval harness -- offline by default (zero API calls,\n"
            "     replays results from data/llm_cache/; pass --live to make real\n"
            "     calls for anything not already cached) -- it writes\n"
            "     results/reply_rows.csv, results/human_scoring_blind.csv, and\n"
            "     results/human_scoring_key.csv:\n"
            "       .venv/bin/python -m eval.run_eval\n"
            "  2. Send results/human_scoring_blind.csv (with\n"
            "     results/human_scoring_rubric.md) to a human rater -- never send\n"
            "     results/human_scoring_key.csv, it reveals which system drafted\n"
            "     each reply.\n"
            "  3. Have them fill in the human_overall column for every row.\n"
            "  4. Save the filled sheet as data/golden/human_scores.csv, then "
            "re-run this module."
        )
        return None

    human_df = pd.read_csv(human_path)
    key_df = pd.read_csv(key_path)
    reply_rows = pd.read_csv(reply_rows_path)
    df = join_human_scores(human_df, key_df, reply_rows)
    return _validate_scores(df)


def _nan_to_none(obj):
    """Recursively replace float('nan') with None so json.dumps(allow_nan=False)
    produces strict, parser-portable JSON (`null`) instead of the bare `NaN`
    token that Python's json module would otherwise emit -- degenerate groups
    (e.g. a system where every score is identical) legitimately produce NaN
    metrics, and strict JSON parsers reject that token."""
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _write_results(results: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "judge_human_agreement.json"
    out_path.write_text(json.dumps(_nan_to_none(results), indent=2, allow_nan=False))
    logger.info("wrote judge/human agreement results to %s", out_path)


def _print_summary(results: dict) -> None:
    o = results["overall"]
    print(f"\n=== Judge vs human agreement (n={o['n']}) ===")
    print(f"  cohen_kappa_binned      = {o['cohen_kappa_binned']:.3f} "
          f"[{results['bootstrap_ci_kappa_binned']['lo']:.3f}, "
          f"{results['bootstrap_ci_kappa_binned']['hi']:.3f}] (95% CI)")
    print(f"  weighted_kappa_quadratic= {o['weighted_kappa_quadratic']:.3f}")
    print(f"  spearman                = {o['spearman']:.3f}")
    print(f"  exact_agreement         = {o['exact_agreement']:.3f}")
    print(f"  within_one              = {o['within_one']:.3f}")
    print(f"  mean_diff (judge-human) = {o['mean_diff']:+.3f}")
    print("\n  Per-system:")
    for system, m in results["per_system"].items():
        print(f"    {system:10s} n={m['n']:3d} kappa_binned={m['cohen_kappa_binned']:.3f} "
              f"within_one={m['within_one']:.3f} mean_diff={m['mean_diff']:+.3f}")


def main() -> dict | None:
    df = load_scores()
    if df is None:
        return None

    human = df["human_overall"].tolist()
    judge = df["judge_overall"].tolist()

    overall = agreement(human, judge)
    per_system = {
        system: agreement(g["human_overall"].tolist(), g["judge_overall"].tolist())
        for system, g in df.groupby("system")
    }
    confusion = confusion_table(human, judge)
    ci = bootstrap_kappa_ci(human, judge, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)

    logger.info(
        "judge/human agreement: n=%d kappa_binned=%.3f weighted_kappa=%.3f "
        "spearman=%.3f exact=%.3f within_one=%.3f mean_diff=%.3f",
        overall["n"], overall["cohen_kappa_binned"], overall["weighted_kappa_quadratic"],
        overall["spearman"], overall["exact_agreement"], overall["within_one"],
        overall["mean_diff"])

    results = {
        "overall": overall,
        "per_system": per_system,
        "confusion": confusion,
        "bootstrap_ci_kappa_binned": ci,
        "note": POOLED_VS_PER_SYSTEM_NOTE,
    }
    _write_results(results)
    _print_summary(results)
    return results


if __name__ == "__main__":
    config.setup_logging("INFO")
    main()
