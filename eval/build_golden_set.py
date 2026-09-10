"""Build the golden eval set: stratified sample + keyword pre-labels for author review.

Stratification uses the free keyword-rule labeler (`weak_labels.weak_label`), not an
LLM call, so building the golden set costs no API usage. The resulting `pre_*`
columns are a prefill only -- the `gold_*` columns are hand-labeled by the author
afterwards by reading each message against the intent taxonomy and escalation
rubric (see eval/golden_labeling_notes.md).
"""
import logging
from pathlib import Path

import pandas as pd

from support_agent import config, data_prep
from support_agent.escalate import decide
from support_agent.weak_labels import weak_label

logger = logging.getLogger(__name__)


def stratified_sample(eval_df: pd.DataFrame, size: int, min_per_intent: int) -> pd.DataFrame:
    df = eval_df.copy().reset_index(drop=True)
    df["pre_intent"] = [weak_label(m) for m in df["customer_open"]]
    df["pre_confidence"] = float("nan")  # keyword rules have no confidence score
    picked = []
    # guarantee the minimum per observed intent, then fill the rest at random
    for intent in df["pre_intent"].unique():
        grp = df[df["pre_intent"] == intent]
        take = min(len(grp), min_per_intent)
        picked.append(grp.sample(take, random_state=config.SEED))
        logger.info("stratum %s: took %d (available %d)", intent, take, len(grp))
    base = pd.concat(picked)
    remaining = df.drop(base.index)
    need = size - len(base)
    if need > 0 and len(remaining) > 0:
        base = pd.concat([base, remaining.sample(min(need, len(remaining)),
                                                 random_state=config.SEED)])
    return base.sample(min(size, len(base)), random_state=config.SEED).reset_index(drop=True)


def build() -> Path:
    config.ensure_dirs()
    out = config.GOLDEN_DIR / "golden_eval.csv"
    if out.exists():
        logger.info("golden set already exists at %s, skipping build", out)
        return out
    _, eval_df = data_prep.load_pools()
    g = stratified_sample(eval_df, config.GOLDEN_SIZE, config.GOLDEN_MIN_PER_INTENT)
    dec = [decide(r.pre_intent, 1.0, [], r.customer_open) for r in g.itertuples()]
    g["pre_escalate"] = [d[0] for d in dec]
    g["pre_reason"] = [d[1] for d in dec]
    g = g.rename(columns={"customer_open": "message"})
    g["gold_intent"] = g["pre_intent"]       # author edits these in place
    g["gold_escalate"] = g["pre_escalate"]
    g["gold_reason"] = g["pre_reason"]
    g["in_spotcheck"] = False
    sc = g.sample(min(40, len(g)), random_state=config.SEED).index
    g.loc[sc, "in_spotcheck"] = True
    cols = ["root_id", "message", "pre_intent", "pre_confidence", "pre_escalate",
            "pre_reason", "gold_intent", "gold_escalate", "gold_reason", "in_spotcheck"]
    g[cols].to_csv(out, index=False)
    logger.info("wrote %d rows to %s", len(g), out)
    return out


if __name__ == "__main__":
    config.setup_logging("INFO")
    print("Wrote", build())
