# SpotifyCares AI Support Agent

Classifies a customer tweet's intent, drafts a reply grounded in Spotify's
historical resolutions, and decides auto-handle vs. escalate — with an
evaluation harness behind the reported numbers.

Full writeup and the decisions behind this design live in
[`report/REPORT.md`](report/REPORT.md) and
[`report/DECISION_LOG.md`](report/DECISION_LOG.md).

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env           # only needed for --live (see below)
```

### Dataset

Download the Kaggle dataset
[`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
and place the CSV at `data/raw/twcs/twcs.csv`.

## Reproduce headline numbers (default: offline, zero API calls)

```bash
python scripts/run_demo.py
```

This is the command that matters: it needs **no `GEMINI_API_KEY`** and makes
**zero network calls**. Every LLM/embedding response behind the reported
numbers is replayed from the committed cache at `data/llm_cache/` (see
"What is committed and why" below), so the numbers you see are exactly the
ones in `report/REPORT.md`.

Runtime is dominated by one thing: parsing the ~500MB raw CSV on the very
first run (roughly a minute; the parsed threads are cached to the gitignored
`data/interim/`, so every run after that skips straight to the retrieval
index and golden set, both already committed as no-ops). Once
`data/interim/` is populated, expect the whole thing — eval harness included
— to finish in well under a minute.

It runs 6 stages and prints a banner + elapsed time for each:

1. Build the thread pool from the raw CSV (parses on first run only)
2. Build the retrieval index — no-op, `data/kb/` is already committed
3. Build the golden set — no-op, `data/golden/golden_eval.csv` is already
   committed and hand-labeled; this never overwrites it
4. Run the eval harness — prints the classification / reply-quality /
   escalation tables and writes them to `results/`
5. Judge/human agreement — runs if `data/golden/human_scores.csv` exists,
   otherwise prints one line explaining it hasn't been added yet
6. A live, end-to-end example on one fixed message, printing intent,
   confidence, escalation decision and reason, top retrieval evidence score,
   and the drafted reply

If offline mode ever hits a call that isn't in the replay cache, it stops
cleanly with a message naming the stage and telling you to rerun with
`--live` — never a raw traceback.

## Recompute live (optional)

Needs `GEMINI_API_KEY` in `.env`. This makes real Gemini API calls and will
reproduce (not just replay) the headline numbers, subject to the free-tier
quota (see `report/REPORT.md` for what that means for runtime).

```bash
python scripts/run_demo.py --live
```

Preview exactly how many API calls a live eval run would make, without
calling the network:

```bash
python -m eval.run_eval --estimate
```

`--export-cache` (used with `--live`) copies every call that run actually
touched into `data/llm_cache/`, skipping files already there — this is how
the committed replay cache above was produced and refreshed:

```bash
python scripts/run_demo.py --live --export-cache
```

## Tests

```bash
pytest -q          # all unit tests, no network (every LLM/embedding call is mocked)
```

## What is committed and why

- `data/golden/` — the hand-labeled golden evaluation set (`golden_eval.csv`)
  plus labeling notes (`eval/golden_labeling_notes.md`). Labels are
  LLM-proposed and author-reviewed, not independently hand-labeled from
  scratch — see "Honest caveats" below.
- `data/kb/` — the retrieval index (embedded vectors + metadata) over the
  first `KB_SIZE` corpus threads. Capped there by the Gemini free-tier
  embedding quota, not by design — it covers a subset of the full corpus,
  not all of history.
- `data/llm_cache/` — the replay cache: the exact LLM and embedding
  responses behind the reported headline numbers, keyed by a hash of each
  call's parameters. This is what makes `python scripts/run_demo.py`
  reproduce the numbers offline, in minutes, with no API key.
- `results/` — the eval harness's output: `eval_results.json` (metrics +
  bootstrap confidence intervals), `classification_rows.csv`,
  `reply_rows.csv`, and `human_scoring_template.csv`.

Not committed: `data/raw/` (the Kaggle CSV — download it yourself),
`data/interim/` (parsed thread pools, rebuilt from the CSV on first run),
`data/cache/` (a working cache of any *new* calls a `--live` run makes,
distinct from the curated `data/llm_cache/` replay cache).

## Headline numbers

Don't take our word for it — read them straight from the source:

- `results/eval_results.json` — classification accuracy/F1, reply-quality
  judge scores, escalation precision/recall, all with 95% bootstrap CIs
- `report/REPORT.md` — the writeup, with the "what's misleading about my
  headline number" honesty section

## Human-agreement workflow

The eval harness writes `results/human_scoring_template.csv` — one row per
(message, reply) pair from the spot-check subset, already scored by the LLM
judge. To add a human comparison point:

1. Fill in the `human_overall` column (1–5) for every row
2. Save the file as `data/golden/human_scores.csv`
3. Rerun `python scripts/run_demo.py` (or `python -m eval.human_agreement`
   directly) — it writes `results/judge_human_agreement.json` and prints
   Cohen's kappa, Spearman correlation, and per-system breakdowns

## Repo map

- `src/support_agent/` — data prep, taxonomy, retrieval, classification,
  reply drafting, escalation policy, the end-to-end pipeline, and the
  cached/offline-capable Gemini client
  - `config.py` — paths, model IDs, constants
  - `data_prep.py` — parse `twcs.csv`, reconstruct threads, filter to
    SpotifyCares, sample + split into corpus/eval pools
  - `taxonomy.py` — the intent taxonomy
  - `weak_labels.py` — free keyword-rule labeler (no LLM calls)
  - `classify.py` — trivial, TF-IDF, and LLM classifiers
  - `retrieve.py` — the embedding index + nearest-neighbor retrieval
  - `draft_reply.py` — trivial, nearest-neighbor, and LLM-grounded reply
    drafting
  - `escalate.py` — the auto-handle vs. escalate policy
  - `pipeline.py` — wires classify → retrieve → draft → escalate together
  - `llm_client.py` — the Gemini wrapper: disk cache, replay-cache
    fallback, offline guard, retry/backoff, free-tier embed pacing
- `eval/` — the evaluation harness
  - `build_golden_set.py` — stratified sample + keyword pre-labels
  - `llm_judge.py` — LLM-as-judge reply scoring
  - `run_eval.py` — the full metrics run (`--estimate` previews API-call
    counts with no network calls)
  - `human_agreement.py` — judge-vs-human agreement metrics
- `scripts/run_demo.py` — the one-command entrypoint this README describes
- `report/REPORT.md`, `report/DECISION_LOG.md` — writeup + design decisions
- `data/golden/`, `data/kb/`, `data/llm_cache/`, `results/` — see "What is
  committed and why" above
- `tests/` — the unit test suite (`pytest -q`), no network required

## Honest caveats

- Golden labels were produced by the author with LLM assistance (keyword
  pre-labels reviewed and corrected against the taxonomy), not
  independently hand-labeled by a separate rater.
- The LLM judge and the reply generator share the same Gemini model family,
  which can inflate agreement between "the model's own idea of a good
  reply" and "the model's own judgment of a good reply."
- The retrieval index covers a `KB_SIZE`-row prefix of the corpus (capped by
  the free-tier embedding quota), not the full historical thread pool.
- The golden and reply-quality subsets are small (see `n_golden` and
  `n_reply_subset` in `results/eval_results.json`'s metadata), so treat
  point estimates cautiously — the bootstrap confidence intervals in that
  same file are the more honest read.
- Judge/human agreement is only computed once `data/golden/human_scores.csv`
  exists; until it does, that piece of evidence is simply absent, not
  assumed favorable.
