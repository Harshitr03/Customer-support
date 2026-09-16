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
  short free-text note explaining the escalation call. All 200 rows were
  reviewed by the author before the reported run; see "Author review" below.
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

- Gold intent differs from the keyword prefill (`pre_intent`) on **102 of
  200** rows (101 after round 1 of the author review, +1 from round 2 below).
- Gold escalate rate: **102 of 200** rows (51.0%). The 2026-09-12 rubric change
  flipped all 37 `technical_bug` rows present at the time to escalate (65 of 200
  before that), and the author's review then set one bug row (43) back to
  auto-handle (see the exception below). Round 2 of the review (below) added a
  38th `technical_bug` row (913386) by relabeling it from `other`, and the
  author kept it auto-handle too, as a second deliberate exception -- so gold
  now holds 38 `technical_bug` rows, 36 escalating and 2 exceptions.
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
  | technical_bug | 38 |
  | billing_subscription | 32 |
  | other | 27 |
  | account_access | 27 |
  | content_catalog | 22 |
  | cancellation_refund | 9 |

  (post-round-2-review counts; see below)

## Author review

The author reviewed all 200 rows before the reported evaluation run, working from a
sheet that showed each drafted label and its reason with blank columns for
disagreements (blank = agree), across two review rounds.
**13 rows were annotated and 6 gold labels changed:**

| row | change | author's reason |
|---|---|---|
| 43 | escalate yes -> **no** | "no need to escalate rn this can be handled as generic bug reply" |
| 60 | `other` -> **`feature_complaint`** | asking for a capability (raise the download limit) |
| 71 | `feature_complaint` -> **`content_catalog`** | blocking an artist read as a catalog concern |
| 78 | `other` -> **`feature_complaint`** | device-capability question read as a capability request |

The other 7 annotations were questions about the drafted label, answered without
changing it (rows 1, 3, 12, 34, 36, 37 and a confirmation on row 50).

**Row 43 is a deliberate exception to the bug-escalation rubric.** "Spotify is broken
and annoying @115888 sort ittttttt" is a `technical_bug`, which the rubric says a human
should take, but the author judged that content-free venting gives a human nothing to
act on. Gold holds 38 `technical_bug` rows total, 36 escalating and 2 deliberate
auto-handled exceptions (43 and, from round 2 below, 913386), and the rule policy
scores a false escalation against each exception. Per-row human judgment overrides the
blanket rule, and the exception is recorded rather than smoothed away.

**Round 2 (2026-09-16), during a live code walkthrough with the classifier's error
analysis on screen.** Reading the 30 messages the LLM classifier still got wrong after
a taxonomy-definition fix (see `report/DECISION_LOG.md`), the author corrected 2 more
gold labels:

| row | change | author's reason |
|---|---|---|
| 861396 | `feature_complaint` -> **`billing_subscription`**, escalate no -> **yes** | student-discount/plan request; the original label was also inconsistent with two other golden rows on the same topic (980320, 1908670), both already `billing_subscription`/escalate |
| 913386 | `other` -> **`technical_bug`**, escalate stays **no** | "finally able to play frank ocean again after 30 mysterious hours where spotify would not let me" literally describes a resolved 30-hour playback failure -- but it's past-tense and resolved, so a human has nothing to act on (same exception as row 43) |

Both corrections happened to match what the LLM classifier had already predicted for
those rows, so applying them moves reported intent accuracy up by 2/200 (a side effect
of fixing the label, not of improving the model) -- disclosed here rather than folded
silently into the headline number.

The review made the headline numbers slightly worse (Gemini intent accuracy 0.790 ->
0.785, escalation precision 0.873 -> 0.864), which is the expected direction for a real
review rather than a rubber stamp.
