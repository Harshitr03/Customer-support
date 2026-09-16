# SpotifyCares AI Support Agent

Classifies a customer tweet's intent, drafts a reply grounded in how Spotify
has actually resolved similar issues before, and decides whether to
auto-handle it or hand it to a human.

The full writeup and the reasoning behind the design choices are in
[`report/REPORT.md`](report/REPORT.md) and
[`report/DECISION_LOG.md`](report/DECISION_LOG.md).

## Quickstart

This reproduces the numbers in `report/REPORT.md`. No API key, no dataset
download, under a minute once `pip install` finishes. Needs Python 3.11+.

```bash
git clone https://github.com/Harshitr03/Customer-support.git
cd Customer-support
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_demo.py
```

Every LLM and embedding call gets replayed from the cache committed at
`data/llm_cache/`, so what prints to your terminal is exactly what's in
the report. It also lands in `results/eval_results.json` if you'd rather
read it programmatically. Everything past this point is detail on top of
that.

```bash
pytest -q          # unit tests, no network
```

## Setup notes

`pyproject.toml` pins the dependency bounds I actually tested against. If
you want the exact pinned environment, use `pip install -r
requirements.lock` instead of the `pip install -e ".[dev]"` line above.

You only need `.env` for `--live` (see "Recompute live" below): `cp
.env.example .env` and drop in your `GEMINI_API_KEY`.

You don't need the raw dataset for the quickstart. `data/interim/` (the
parsed threads), `data/kb/` (the retrieval index), and `data/golden/`
(the golden set) are all already committed, so `run_demo.py` never opens
the CSV. Only grab it if you actually want to rebuild `data/interim/`
from scratch: [the Kaggle dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
→ `data/raw/twcs/twcs.csv`.

## What `run_demo.py` actually does

Six stages, each one prints a banner and how long it took:

1. Build the thread pool from the raw CSV — no-op, `data/interim/` is already committed
2. Build the retrieval index — no-op, `data/kb/` is already committed
3. Build the golden set — no-op too, `data/golden/golden_eval.csv` is
   committed and hand-labeled, this step never touches it
4. Run the eval harness — prints the classification, reply-quality, and
   escalation tables, writes them to `results/`
5. Judge/human agreement — runs if `data/golden/human_scores.csv` exists,
   otherwise just says so
6. One live example end to end: intent, confidence, the escalation call
   and its reason, the top retrieval score, the drafted reply

If offline mode hits something that isn't cached, it stops cleanly and
tells you to rerun with `--live`. It won't hand you a raw traceback.

## Recompute live (optional)

You'll need `GEMINI_API_KEY` in `.env` for any of this.

| Mode | What it does |
|---|---|
| default | offline replay only, zero network calls |
| `--live` | calls the API only for what isn't cached anywhere yet (local or replay) |
| `--live --no-replay` | ignores the committed replay cache too, so anything not already local hits the network — delete `data/cache/` as well for a true from-scratch recompute |

```bash
python scripts/run_demo.py --live                  # fills in whatever's missing
python -m eval.run_eval --estimate                  # see the call count first, no network
python scripts/run_demo.py --live --export-cache    # after a --live run, saves new responses into data/llm_cache/
```

One thing worth knowing: this isn't bit-identical to what's committed.
Replies generate at temperature 0.2–0.3, so a fresh call can come back
with different wording for the same prompt.

## Demo UI (optional, not part of the deliverable)

```bash
pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

Pick a cached golden-set message or type your own — typing needs `--
--live` and a `GEMINI_API_KEY`. You can also browse `results/` from
here. Offline by default, same caching model as everything else. No
Spotify logo, wordmark, icon, or brand green; there's a "not affiliated
with Spotify" note in the app itself.

## What is committed and why

| Path | Contents |
|---|---|
| `data/golden/` | The golden set (`golden_eval.csv`) plus labeling notes and rubric |
| `data/kb/` | Retrieval index: embedded vectors and metadata, all 5,400 history threads |
| `data/llm_cache/` | The replay cache — exact LLM/embedding responses behind the reported numbers |
| `results/` | Eval harness output: metrics, per-row CSVs, blind human-scoring files |

`data/interim/` (the parsed thread pools) is committed too, see Setup
above. Not committed: `data/raw/` (the Kaggle CSV itself) and
`data/cache/`, your own local working cache from `--live` calls, which
is separate from the curated `data/llm_cache/` replay cache.

## Human-agreement workflow

This is blind by construction, enforced in code, not just by convention
— the file a rater actually sees never carries the system name or the
judge's score, and row order gets shuffled with a fixed seed.

- `results/human_scoring_blind.csv` — this is what you send out. 120
  pairs (all 3 systems for each of the 40 spot-check messages, not a
  rotation across them), item ids `h001` through `h120` assigned after
  the shuffle, `human_overall` left blank.
- `results/human_scoring_key.csv` — never send this one. It's the
  de-anonymizing key: item_id back to pair_id, root_id, system.
- `results/human_scoring_rubric.md` — send this alongside the blind sheet.

To add a human comparison point: send the blind sheet and rubric to a
rater, have them fill in `human_overall` (1 to 5), save the result as
`data/golden/human_scores.csv`, then rerun `python scripts/run_demo.py`
(or just `python -m eval.human_agreement`). It joins everything back
against the key, checks that every item matches up exactly once, and
writes `results/judge_human_agreement.json` with kappa, Spearman, and
per-system breakdowns. The per-system numbers are the stricter read —
pooling agreement across all three systems gets inflated by how
different the systems are from each other, not just by how well the
rater and judge actually agree.

## Repo map

- `src/support_agent/`
  - `config.py` — paths, model IDs, constants
  - `data_prep.py` — parses `twcs.csv`, reconstructs threads, filters to
    SpotifyCares, samples and splits into corpus/eval pools
  - `taxonomy.py` — the intent taxonomy
  - `weak_labels.py` — a free keyword-rule labeler, no LLM calls
  - `classify.py` — trivial, TF-IDF, and LLM classifiers
  - `retrieve.py` — the embedding index and nearest-neighbor retrieval
  - `draft_reply.py` — trivial, nearest-neighbor, and LLM-grounded reply drafting
  - `escalate.py` — the auto-handle vs. escalate policy
  - `pipeline.py` — wires classify, retrieve, draft, and escalate together
  - `llm_client.py` — the Gemini wrapper: disk cache, replay-cache
    fallback, offline guard, retry/backoff, free-tier embed pacing
- `eval/`
  - `build_golden_set.py` — stratified sample plus keyword pre-labels
  - `llm_judge.py` — LLM-as-judge reply scoring
  - `run_eval.py` — the full metrics run (`--estimate` previews the call
    count, no network)
  - `human_agreement.py` — judge-vs-human agreement metrics
- `scripts/run_demo.py` — the one-command entrypoint from above
- `report/` — `REPORT.md`, `DECISION_LOG.md`
- `tests/` — `pytest -q`, no network required

## Honest caveats

The full story on each of these is in `report/REPORT.md` §5–6.

Golden labels were AI-drafted and human-reviewed, not independently
hand-labeled. Claude drafted all 200 against a written rubric, never
against the keyword prefill or any model's own prediction, and I
reviewed every row across two rounds — 13 annotated, 6 changed,
including two deliberate exceptions to the bug-escalation rubric. The
second round happened while I was reading the classifier's own errors,
and it moved accuracy up, which is a real self-review bias I'm
disclosing rather than smoothing over. The full record is in
`eval/golden_labeling_notes.md`. If you want to spot-check anything
first, look at the 102 rows where `gold_intent` differs from the keyword
prefill (`pre_intent`), and the 102 where `gold_escalate` is `True`.

The judge and the reply generator are the same model family (Gemini),
which can inflate agreement between what the model thinks a good reply
looks like and what the model judges a good reply to be.

Retrieval covers all 5,400 history threads now. It was capped at 3,800
for the early runs by the free-tier embed quota; growing it by 42% only
moved grounded reply quality by about 0.02, so that clearly wasn't the
constraint holding things back.

The golden set and the reply-quality subset are both small, so treat any
single point estimate with some caution — the bootstrap confidence
intervals in `results/eval_results.json` are the more honest number to
look at.

Judge/human agreement on the full 120-pair sheet is weaker than a
smaller, earlier sample suggested. Binned kappa went from 0.342
[−0.074, 0.692] on 40 pairs down to 0.155 [−0.001, 0.305] on the full
120 — the confidence interval more than halved but the point estimate
dropped, which is what it looks like when a small-sample overestimate
gets corrected, not a real regression. The `nearest` system collapsed to
essentially chance agreement (0.304 down to −0.017): the judge just
can't tell a reply that was written for a different customer from one
that's actually good.
