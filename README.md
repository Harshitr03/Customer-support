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

`pyproject.toml`'s dependency bounds are the versions actually tested.
`requirements.lock` (`pip freeze --exclude-editable`) pins every transitive
dependency exactly, for a fully reproducible environment:
`pip install -r requirements.lock` instead of the `pip install -e ".[dev]"`
line above if you want that.

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

Needs `GEMINI_API_KEY` in `.env`. What each mode actually does:

| Mode | What it does |
|---|---|
| default (no flags) | offline replay — every call is served from the committed replay cache at `data/llm_cache/`, zero network calls |
| `--live` | calls the API **only** for responses cached in neither the local `data/cache/` nor the committed `data/llm_cache/` replay cache — everything already cached (local or replay) is still reused, not refetched |
| `--live --no-replay` | ignores the committed replay cache entirely, so anything not already in the local `data/cache/` hits the network — delete `data/cache/` too for a full recompute from scratch |

```bash
python scripts/run_demo.py --live                 # fills in whatever isn't cached yet
python scripts/run_demo.py --live --no-replay      # recompute, ignoring the committed replay cache
```

Recomputing is **not bit-identical** to the committed replay cache: reply
generation runs at temperature 0.2–0.3, so a fresh `--live` (or
`--live --no-replay`) call for the same prompt can return different text
than what's replayed. `--no-replay` is only valid together with `--live`.

Preview exactly how many API calls a live eval run would make, without
calling the network:

```bash
python -m eval.run_eval --estimate
```

`--export-cache` (used with `--live`) copies every call that run actually
touched into `data/llm_cache/` — writing new files, overwriting any whose
bytes have changed (so a stale replay entry doesn't linger), and skipping
ones that are already identical. This is how the committed replay cache
above was produced and refreshed:

```bash
python scripts/run_demo.py --live --export-cache
```

## Tests

```bash
pytest -q          # all unit tests, no network (every LLM/embedding call is mocked)
```

## Demo UI (optional)

A small local Streamlit app ("Support Agent Console") for demoing the
agent and exploring the evaluation results interactively. It's a demo
tool, not part of the deliverable — the evaluation in `report/` is.

```bash
pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

- **"Try a message"** — pick from the golden-set messages whose full
  pipeline response (classify, the query embedding, and the grounded
  reply) is already cached, or type your own, then run the agent and see
  the verdict, intent, drafted reply, and the retrieved evidence it was
  grounded on.
- **"Results"** — reads `results/eval_results.json` and the per-row CSVs
  written by `python -m eval.run_eval`; shows an empty state if `results/`
  doesn't exist yet.

**Offline by default, zero network calls** — same replay-cache model as
`scripts/run_demo.py`: picking a cached example and running it never hits
the network. Typing a brand-new message does need a live call, so it only
works when the app is started with `-- --live`:

```bash
streamlit run app/streamlit_app.py -- --live   # needs GEMINI_API_KEY, uses API quota
```

`streamlit` is an optional extra (`ui`), not a core dependency, and isn't
in `requirements.lock` — that lock file is generated from `pip install -e
".[dev]"` (see below), which the UI extra is deliberately outside of, so
installing the UI never pulls streamlit into the reproducible core
environment.

This demo does not use Spotify's logo, wordmark, icon, or brand font, and
does not reuse Spotify's brand green. It shows this note in the app:

> Demo built on public SpotifyCares tweets from a Kaggle dataset. Not
> affiliated with Spotify.

## What is committed and why

- `data/golden/` — the golden evaluation set (`golden_eval.csv`) plus
  labeling notes and rubric (`eval/golden_labeling_notes.md`). Gold labels
  were drafted by Claude (an AI assistant) against the written rubric, not
  independently hand-labeled by a human from scratch, and are reviewed by
  the author before the evaluation is run — see "Honest caveats" below for
  the review status and what to spot-check first.
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
  `reply_rows.csv`, and the blind human-scoring set (`human_scoring_blind.csv`,
  `human_scoring_key.csv`, `human_scoring_rubric.md` — see "Human-agreement
  workflow" below).

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

Human scoring is **blind by construction**, enforced in code, not just by
convention — the file a human rater sees never carries the drafting
system's name or the judge's own score for that reply, and its row order
is shuffled (seeded) so neither leaks through position either:

- `results/human_scoring_blind.csv` — the 40 spot-check (message, reply)
  pairs, shuffled, with an opaque `item_id` (`h01`…`h40`) assigned *after*
  the shuffle. Columns: `item_id`, `message`, `reply`, `reference`,
  `human_overall` (blank). This is the file to send out.
- `results/human_scoring_key.csv` — `item_id`, `pair_id`, `root_id`,
  `system`. Kept back; **never send this to the rater**, it's exactly what
  de-anonymizes each row.
- `results/human_scoring_rubric.md` — the judge's "overall" rubric in
  plain words, on the same 1–5 scale — send this alongside the blind sheet.

To add a human comparison point:

1. Send `results/human_scoring_blind.csv` and `results/human_scoring_rubric.md`
   to a human rater (not the key).
2. Have them fill in the `human_overall` column (1–5) for every row.
3. Save the filled sheet as `data/golden/human_scores.csv`.
4. Rerun `python scripts/run_demo.py` (or `python -m eval.human_agreement`
   directly) — it joins the filled sheet back against
   `results/human_scoring_key.csv` (by `item_id`) and `results/reply_rows.csv`
   (by `root_id`+`system`) to re-attach the system and the judge's score,
   validates every item joined exactly once, then writes
   `results/judge_human_agreement.json` and prints Cohen's kappa, Spearman
   correlation, and per-system breakdowns. Per-system agreement (~13 pairs
   each) is the stricter read — agreement pooled across all three systems
   is inflated by their quality differences, not just by rater/judge
   agreement on any single reply.

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

- Golden labels (intent, escalation, reason) were drafted by Claude (an AI
  assistant) reading each message against the written rubric — never
  against the keyword prefill or any model's own prediction — and are
  then reviewed by the author before the reported evaluation run — one AI
  labeler and one human reviewer, **not** independent hand-labeling by a
  separate rater or a panel. The author reviewed all 200 rows, annotated 11
  and changed 4 gold labels; the record, including a deliberate exception to
  the bug-escalation rubric on row 43, is in
  `eval/golden_labeling_notes.md`. The review moved the headline numbers
  slightly *down* (Gemini intent accuracy 0.790 → 0.785), which is the
  expected direction for a real review. The rows most likely to still carry
  a labeling mistake are the 101 where `gold_intent` differs from the
  keyword prefill (`pre_intent`) and the 101 where `gold_escalate` is
  `True` — that is, wherever the gold label disagrees with the cheap
  mechanical baseline.
- The LLM judge and the reply generator share the same Gemini model family,
  which can inflate agreement between "the model's own idea of a good
  reply" and "the model's own judgment of a good reply."
- The retrieval index now covers all 5,400 historical threads (`KB_SIZE`).
  It was capped at 3,800 for the first runs by the free-tier embedding
  quota; growing it by 42% moved grounded reply quality by about 0.02 on
  the judge's 1–5 scale, so retrieval size was not the binding constraint.
- The golden and reply-quality subsets are small (see `n_golden` and
  `n_reply_subset` in `results/eval_results.json`'s metadata), so treat
  point estimates cautiously — the bootstrap confidence intervals in that
  same file are the more honest read.
- Judge/human agreement is only computed once `data/golden/human_scores.csv`
  exists; until it does, that piece of evidence is simply absent, not
  assumed favorable.
