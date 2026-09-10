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
  one of the 200 rows is hand-labeled by the author: the message is read against
  the intent definitions in `src/support_agent/taxonomy.py` (the single source of
  truth for what each intent name means) and against the escalation rubric in
  `src/support_agent/escalate.py` (escalate if the intent is account/billing/
  cancellation-related, if there is legal/security/risk language, if confidence
  would be low, or if the thread has 3+ unresolved rounds). `gold_reason` is a
  short free-text note explaining the escalation call. The `pre_*` columns are
  pre-filled into `gold_*` at build time purely as an editing convenience -- every
  row is reviewed and corrected (or confirmed) by hand, not left unexamined.
- **`in_spotcheck`:** marks a seeded, stratified ~40-row subset of the 200. This
  subset is used for a separate human reply-quality scoring pass (draft-reply
  grading), not for re-checking intent/escalation labels -- all 200 rows get full
  intent/escalation labels regardless of this flag.
