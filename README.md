# SpotifyCares AI Support Agent

Classifies a customer tweet's intent, drafts a reply grounded in Spotify's
historical resolutions, and decides auto-handle vs. escalate — with an
evaluation harness behind the reported numbers.

Full writeup and the decisions behind this design: [`report/REPORT.md`](report/REPORT.md),
[`report/DECISION_LOG.md`](report/DECISION_LOG.md).

## Quickstart

Reproduces `report/REPORT.md`'s numbers, no API key, no dataset download,
under a minute after `pip install`. Requires **Python 3.11+**.

```bash
git clone https://github.com/Harshitr03/Customer-support.git
cd Customer-support
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_demo.py
```

Every LLM/embedding call is replayed from the committed cache at
`data/llm_cache/`, so the classification / reply-quality / escalation
tables **printed to your terminal** are exactly the numbers in the report
— also written to `results/eval_results.json` for programmatic reading.
Everything below is detail on top of this.

```bash
pytest -q          # unit tests, no network
```

## Setup notes

`pyproject.toml` pins the tested dependency bounds. For an exact,
fully-pinned environment, use `pip install -r requirements.lock` instead
of the `pip install -e ".[dev]"` line above.

For `--live` (optional, see "Recompute live" below): `cp .env.example .env`
and add your `GEMINI_API_KEY`.

**Dataset — not needed for the quickstart above.** `data/interim/` (parsed
thread pools), `data/kb/` (retrieval index), and `data/golden/` (golden
set) are all committed, so `scripts/run_demo.py` never touches the raw
CSV. Only download it if you want to rebuild `data/interim/` from scratch:
[Kaggle: `thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
→ `data/raw/twcs/twcs.csv`.

## What `run_demo.py` does

Six stages, each printing a banner + elapsed time:

1. Build the thread pool from the raw CSV — no-op, `data/interim/` is committed
2. Build the retrieval index — no-op, `data/kb/` is committed
3. Build the golden set — no-op, `data/golden/golden_eval.csv` is committed
   and hand-labeled; never overwritten
4. Run the eval harness — prints classification / reply-quality /
   escalation tables, writes them to `results/`
5. Judge/human agreement — runs if `data/golden/human_scores.csv` exists,
   otherwise says so
6. One live end-to-end example: intent, confidence, escalation
   decision + reason, top retrieval score, drafted reply

If offline mode hits a call that isn't cached, it stops cleanly and tells
you to rerun with `--live` — never a raw traceback.

## Recompute live (optional)

Needs `GEMINI_API_KEY` in `.env`.

| Mode | What it does |
|---|---|
| default | offline replay only — zero network calls |
| `--live` | calls the API only for what's cached nowhere yet (local or replay) |
| `--live --no-replay` | ignores the committed replay cache; also delete `data/cache/` for a full recompute |

```bash
python scripts/run_demo.py --live                  # fills in whatever isn't cached
python -m eval.run_eval --estimate                  # preview call counts first, zero network calls
python scripts/run_demo.py --live --export-cache    # after a --live run, commits new responses to data/llm_cache/
```

Not bit-identical to the committed cache — replies generate at
temperature 0.2–0.3, so a fresh call can return different text for the
same prompt.

## Demo UI (optional, not part of the deliverable)

```bash
pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

Try a cached golden-set message or type your own (typing needs `-- --live`
and `GEMINI_API_KEY`); browse `results/` interactively. Offline by default,
same replay-cache model as above. No Spotify logo, wordmark, icon, or
brand green — shows a "not affiliated with Spotify" note in-app.

## What is committed and why

| Path | Contents |
|---|---|
| `data/golden/` | Golden set (`golden_eval.csv`) + labeling notes/rubric |
| `data/kb/` | Retrieval index — embedded vectors + metadata, all 5,400 history threads |
| `data/llm_cache/` | The replay cache: exact LLM/embedding responses behind the reported numbers |
| `results/` | Eval harness output — metrics, per-row CSVs, blind human-scoring files |

`data/interim/` (parsed thread pools) is also committed — see Setup above.
**Not committed:** `data/raw/` (the Kaggle CSV) and `data/cache/` (your
local working cache of new `--live` calls, separate from the curated
`data/llm_cache/` replay cache).

## Human-agreement workflow

Blind by construction, enforced in code — the file a rater sees carries no
system name and no judge score, row order shuffled (seeded):

- `results/human_scoring_blind.csv` — send this. 120 pairs (40 spot-check
  messages x all 3 systems, not a rotation), `item_id` (`h001`…`h120`)
  assigned after shuffling, `human_overall` blank.
- `results/human_scoring_key.csv` — **never send this.** It's the
  de-anonymizing key (`item_id` → `pair_id`, `root_id`, `system`).
- `results/human_scoring_rubric.md` — send alongside the blind sheet.

To add a human comparison point: send the blind sheet + rubric to a rater,
have them fill `human_overall` (1–5), save as
`data/golden/human_scores.csv`, then rerun `python scripts/run_demo.py`
(or `python -m eval.human_agreement`). It re-joins the key, validates
every item joins exactly once, and writes
`results/judge_human_agreement.json` — kappa, Spearman, and per-system
breakdowns (the stricter read: pooled agreement is inflated by the
systems' own quality differences, not just rater/judge agreement).

## Repo map

- `src/support_agent/`
  - `config.py` — paths, model IDs, constants
  - `data_prep.py` — parse `twcs.csv`, reconstruct threads, filter to
    SpotifyCares, sample + split into corpus/eval pools
  - `taxonomy.py` — the intent taxonomy
  - `weak_labels.py` — free keyword-rule labeler (no LLM calls)
  - `classify.py` — trivial, TF-IDF, and LLM classifiers
  - `retrieve.py` — embedding index + nearest-neighbor retrieval
  - `draft_reply.py` — trivial, nearest-neighbor, and LLM-grounded reply drafting
  - `escalate.py` — the auto-handle vs. escalate policy
  - `pipeline.py` — wires classify → retrieve → draft → escalate together
  - `llm_client.py` — Gemini wrapper: disk cache, replay-cache fallback,
    offline guard, retry/backoff, free-tier embed pacing
- `eval/`
  - `build_golden_set.py` — stratified sample + keyword pre-labels
  - `llm_judge.py` — LLM-as-judge reply scoring
  - `run_eval.py` — the full metrics run (`--estimate` for a network-free preview)
  - `human_agreement.py` — judge-vs-human agreement metrics
- `scripts/run_demo.py` — the one-command entrypoint above
- `report/` — `REPORT.md`, `DECISION_LOG.md`
- `tests/` — `pytest -q`, no network required

## Honest caveats

Full narrative for each of these: `report/REPORT.md` §5–6.

- **Golden labels are AI-drafted, human-reviewed — not independent
  hand-labeling.** Claude drafted all 200 against a written rubric (never
  against the keyword prefill or any model prediction); the author
  reviewed every row across two rounds (13 annotated, 6 changed,
  including two deliberate bug-escalation-rubric exceptions). Round 2 was
  done while reading the classifier's own errors and moved accuracy *up*
  — a real self-review bias, disclosed rather than hidden. Full record:
  `eval/golden_labeling_notes.md`. Worth re-checking first: the 102 rows
  where `gold_intent` differs from the keyword prefill (`pre_intent`), and
  the 102 where `gold_escalate` is `True`.
- **The judge and the reply generator share a model family** (Gemini) —
  can inflate "the model's idea of a good reply" agreeing with "the
  model's judgment of a good reply."
- **Retrieval covers all 5,400 history threads**, capped at 3,800 for
  early runs by the free-tier embed quota; growing it 42% moved grounded
  quality by only ~0.02 — not the binding constraint.
- **The golden and reply-quality subsets are small** — treat point
  estimates cautiously; the bootstrap CIs in `results/eval_results.json`
  are the more honest read.
- **Judge/human agreement (n=120) is weaker than an earlier small sample
  (n=40) suggested.** Binned kappa 0.342 [−0.074, 0.692] → **0.155
  [−0.001, 0.305]** — CI more than halved, point estimate dropped: a
  small-sample overestimate getting corrected, not a regression. `nearest`
  collapsed to essentially chance (0.304 → **−0.017**) — the judge can't
  tell a reply copied for a different customer from a good one.
