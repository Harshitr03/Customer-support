"""Agreement between the LLM judge and a human rater on reply overall scores.

Task 12: the assignment asks for "evidence of how well your LLM judge agrees
with a human". This module computes that evidence from
`data/golden/human_scores.csv`, a file where a human has filled in
`human_overall` for the same (message, reply) pairs the judge already scored
in `results/human_scoring_template.csv` (Task 11).

Controller rulings applied here (see task-12-brief.md for the base spec,
and the ruling list in the task prompt for what overrides it):

  1. Input shape: main() reads the Task-11 human_scoring_template columns
     (pair_id, root_id, message, system, reply, reference, judge_overall,
     human_overall) rather than the brief's 2-3 column shape, so
     per-system agreement is possible. Validated at this boundary.
  2. Metrics: agreement() returns n, cohen_kappa_binned (low 1-2 / mid 3 /
     high 4-5), weighted_kappa_quadratic (raw 1-5, quadratic weights),
     spearman, exact_agreement, within_one, mean_human, mean_judge, and
     mean_diff (judge minus human, i.e. judge bias). Degenerate inputs
     (e.g. constant scores) yield NaN, never an exception or a warning.
  3. Also: per-system breakdown, a 3x3 binned confusion table, and a
     bootstrap 95% CI for cohen_kappa_binned (1000 paired row resamples,
     seed 42; resamples where kappa is undefined are skipped and counted).
  4. Output: results/judge_human_agreement.json (overall + per-system +
     confusion + CI) plus a concise printed summary.
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

REQUIRED_COLUMNS = ("pair_id", "root_id", "message", "system", "reply",
                     "reference", "judge_overall", "human_overall")
BIN_LABELS = ("low", "mid", "high")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
_BOOTSTRAP_ALPHA = 0.05


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

def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"data/golden/human_scores.csv is missing required column(s): {missing}")

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
            df[col] = df[col].astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"column {col!r} must contain integers: {exc}") from exc
        bad = df.loc[~df[col].between(1, 5), col].tolist()
        if bad:
            raise ValueError(
                f"column {col!r} must be in the 1..5 range, found out-of-range "
                f"value(s): {bad}")


def load_scores() -> pd.DataFrame | None:
    """Load and validate data/golden/human_scores.csv. Returns None (after
    printing how to produce the file) if it doesn't exist yet -- neither the
    eval run nor the human-scoring pass has happened."""
    path = config.GOLDEN_DIR / "human_scores.csv"
    if not path.exists():
        print(
            "No data/golden/human_scores.csv found yet. To produce it:\n"
            "  1. Run the eval harness (writes results/human_scoring_template.csv):\n"
            "       .venv/bin/python -m eval.run_eval\n"
            "  2. Fill in the human_overall column for every row of that CSV.\n"
            "  3. Save it as data/golden/human_scores.csv, then re-run this module."
        )
        return None
    df = pd.read_csv(path)
    _validate(df)
    return df


def _write_results(results: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "judge_human_agreement.json"
    out_path.write_text(json.dumps(results, indent=2))
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
    }
    _write_results(results)
    _print_summary(results)
    return results


if __name__ == "__main__":
    config.setup_logging("INFO")
    main()
