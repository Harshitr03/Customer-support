# Golden Set — Sampling & Labeling Methodology

- **Source:** `eval_pool` (10% of the 6k-thread working pool), disjoint by thread
  from the `corpus_pool` used for retrieval/weak-label training -- so no golden-set
  message is grounded against itself.
- **Size:** 200 examples, stratified by a keyword-rule intent label (see
  `stratified_sample`, `weak_labels.weak_label`) with a minimum of 10 per observed
  intent, then topped up at random from the remaining pool. Fixed seed 42.
- **Stratification method (no API cost):** `weak_labels.weak_label` runs a free
  regex/keyword classifier over `customer_open` to produce `pre_intent` for every
  row of the 600-message eval pool, purely to balance the sample across intents
  before any hand-labeling happens. `pre_confidence` is left `NaN` because the
  keyword rules don't produce a confidence score -- the column is kept so the CSV
  schema stays fixed for downstream tooling. `pre_escalate` / `pre_reason` are
  computed by calling `escalate.decide(pre_intent, 1.0, [], message)` -- i.e. the
  escalation rule is evaluated against the keyword-rule intent at full assumed
  confidence, purely as a starting prefill.
- **`pre_*` columns are NOT gold.** They are a cheap, mechanical prefill to save
  typing during hand-labeling and to give the stratification something to balance
  on. They are produced by simple regex keyword matching, not by an LLM or by a
  human reading the message.
- **`gold_intent` / `gold_escalate` / `gold_reason` are the ground truth.** Every
  one of the 200 rows was drafted by Claude (an AI assistant) reading the message
  against the written rubric below, independent of any model prediction -- the
  `pre_*` prefill is never consulted while drafting the gold label, only used
  afterward as a diff to sanity-check disagreement counts. `gold_reason` is a
  short free-text note explaining the escalation call. Every row is reviewed by
  the author before the evaluation is run; see "Author review" below.
  - **Intent rubric:** the intent definitions in `src/support_agent/taxonomy.py`
    (the single source of truth for what each intent name means).
  - **Escalation rubric:** `report/DECISION_LOG.md` item 7, quoted here in full
    so this file doesn't drift from that one:

    > **What "escalate" means in the golden set.** A human should take the
    > message when it needs account-specific action (login, payment or plan
    > change, a refund, cancelling an account the customer can't reach), when
    > it's a security or fraud issue, when it follows up an open DM case, or
    > when it reports a technical bug. Catalog questions, feature feedback,
    > how-tos, and praise are auto-handleable.

    **Rubric change, 2026-09-12 — disclosed because it happened after the
    first results were in.** The rubric originally called bug reports
    auto-handleable, on the evidence that Spotify's own first reply to a
    plain bug report asked for a DM or the account email only 28% of the
    time (versus 64% once the customer said the standard fixes had failed).
    After seeing the first run, the author decided that a support agent can
    rarely resolve a bug from the first reply and that bug reports should go
    to a human by default. That changed `ESCALATE_INTENTS` in
    `src/support_agent/escalate.py` **and** this rubric, so all 37
    `technical_bug` rows flipped to `gold_escalate = True`; their
    `gold_reason` records the change. Scoring a policy against a rubric
    revised to match it inflates the escalation numbers, so the report's
    "what's misleading" section states this plainly.

    This rubric is **not** "read `src/support_agent/escalate.py`'s rules and
    apply them" -- doing that would grade the rule-based policy against its
    own definition of correct, which is circular. The rubric above is an
    independent, human-readable judgment call about what actually needs a
    human, and `escalate.py`'s rule-based `pre_escalate` prefill is graded
    against it exactly like every other prediction, via `policy_only` in
    `eval_results.json`.
- **`in_spotcheck`:** a **seeded simple random sample of 40** rows out of the
  200 (`g.sample(40, random_state=SEED)` in `build_golden_set.py`) -- **not**
  stratified by intent or by anything else. It marks the subset used for a
  separate human reply-quality scoring pass (draft-reply grading; see
  `results/human_scoring_blind.csv` and the README's "Human-agreement
  workflow"), not for re-checking intent/escalation labels -- all 200 rows
  get full intent/escalation labels regardless of this flag.

## Labeling results

- Gold intent differs from the keyword prefill (`pre_intent`) on **101 of
  200** rows.
- Gold escalate rate: **102 of 200** rows (51%), after the 2026-09-12 rubric
  change flipped all 37 `technical_bug` rows to escalate (65 of 200 before).
  Escalation now applies to every `technical_bug`, `account_access`,
  `billing_subscription`, and `cancellation_refund` row plus a few `other`
  rows (open DM follow-ups); no `content_catalog` or `feature_complaint` row
  escalates.
- Gold escalation differs from the rule prefill (`pre_escalate`) on **66 of
  200** rows. The prefill applies the escalation rules to the *keyword* intent
  at assumed full confidence, so this gap mixes keyword-intent error with
  genuine policy error; `policy_only` in `results/eval_results.json` measures
  policy error against the gold intent instead.
- The per-system escalation scores are recomputed on every evaluation run;
  `results/eval_results.json` is the current source.
- Gold intent counts:

  | intent | count |
  |---|---|
  | feature_complaint | 45 |
  | technical_bug | 37 |
  | billing_subscription | 31 |
  | other | 30 |
  | account_access | 27 |
  | content_catalog | 21 |
  | cancellation_refund | 9 |

## Author review

The author reviews every row before the evaluation is run; the number of labels changed will be recorded here.
