# SpotifyCares Support Agent: Report

An AI support agent for **@SpotifyCares**, built on the Kaggle *Customer Support on Twitter* dataset. For each incoming customer tweet it does three things:

1. Classifies the tweet into one of seven intents.
2. Drafts a reply grounded in how Spotify historically answered similar messages.
3. Decides whether to auto-handle the tweet or escalate it to a human, with a stated reason.

The non-obvious choices behind it are in [`DECISION_LOG.md`](DECISION_LOG.md). How to reproduce everything is in the [README](../README.md).

## 1. Problem framing

### What "good" means for this brand

Spotify's support traffic on Twitter is mostly app bugs, account access, billing, catalog questions, cancellations, and feature feedback. For this brand, a good agent:

- **Never auto-handles something only a human can fix.** Hacked accounts, charges, refunds, and locked-out users need identity checks or account actions a bot can't take. A wrong "auto-handle" there is the costliest failure, so escalation leans toward safety for money and security.
- **Doesn't waste humans on what a template can answer.** Troubleshooting steps, catalog-licensing answers, and "thanks, we've passed your feedback on" make up most of the volume, and Spotify's own replies to them follow a pattern.
- **Sounds like Spotify and invents nothing.** Replies are short, empathetic, Twitter-length, and grounded in real historical replies. They never promise refunds, quote account details, or copy another customer's @handle or a dead link.
- **Explains itself.** Every escalation decision carries a human-readable reason, so a team lead can audit why a message was or wasn't escalated.

### What I chose not to build

- **Multi-turn conversation handling.** The agent handles a customer's *opening* message only. The long-thread escalation rule exists, but the evaluation never exercises it (see §5).
- **Account actions, DMs, or any live Twitter integration.** The agent drafts replies; it never sends anything or touches an account.
- **Fine-tuning.** A small labeled set doesn't justify it, and a prompted model plus retrieval is easier to inspect.
- **Multiple brands, or the full dataset.** This is one brand, on a 6,000-thread sample. The dataset has about 3M tweets, and the assignment expects a subsample.
- **A second independent judge model** — out of scope; the single judge is a known weakness (§5). (A small Streamlit UI exists — `app/streamlit_app.py`, tested — as a convenience wrapper around this pipeline, not a separate deliverable.)

## 2. The system

```
customer tweet
   │
   ├─► classify      Gemini 3.5 Flash-Lite, taxonomy in the prompt → intent + confidence
   │
   ├─► retrieve      Gemini embeddings (768-d), cosine top-4 over 5,400 historical
   │                 (customer message → Spotify reply) pairs
   │
   ├─► draft reply   Gemini 3.5 Flash-Lite, conditioned on the predicted intent and the
   │                 4 retrieved pairs (handles, links, and agent sign-offs stripped)
   │
   └─► escalate?     5 auditable rules, in order:
                       1. risk language (legal / fraud / security)
                       2. sensitive intent (account, billing, cancellation)
                       3. "I already tried the fixes" language
                       4. classifier confidence < 0.55
                       5. 3+ unresolved rounds
                     otherwise auto-handle; every path returns a reason
```

**Data.** I rebuilt the reply threads from the raw tweet IDs and kept threads containing a SpotifyCares reply. From those I sampled 6,000 threads (seed 42) and split them by thread into 5,400 "history" threads and 600 held-out threads.

- **History threads** feed retrieval and the keyword baseline. All 5,400 are embedded; the index was capped at 3,800 for the first runs by the free-tier embedding quota (decision 8).
- **Held-out threads** are the only source of the golden set.
- **No overlap.** No test message can be grounded on its own thread.

**Intents** (`src/support_agent/taxonomy.py`): `technical_bug`, `account_access`, `billing_subscription`, `content_catalog`, `cancellation_refund`, `feature_complaint`, and `other`. The taxonomy was checked against real messages before it was fixed (decision 2).

## 3. How it is evaluated

**Golden set (200 messages).** A stratified sample from the 600 held-out threads, with at least 10 per intent. The sample was chosen with free keyword labels, so no model output decided which rows were included.

- **How labels were made:** every gold intent, escalation decision, and reason was drafted by Claude against the written rubric and reviewed by the author. The details are in [`eval/golden_labeling_notes.md`](../eval/golden_labeling_notes.md).
- **Differs from the keyword prefill:** gold intent disagrees with the prefill on 102 of 200 rows.
- **Differs from the rule policy:** the gold escalation rubric disagrees with the rules on a known set of rows, and that gap is exactly what the escalation metric measures.

**Three systems per task, compared on the same golden rows:**

| Task | Trivial baseline | Simple baseline | Main approach |
|---|---|---|---|
| Intent | Majority class of the *training* labels | TF-IDF + logistic regression, trained on keyword labels from the history | Gemini 3.5 Flash-Lite, taxonomy in the prompt |
| Reply | One canned template per intent | Nearest historical Spotify reply, copied | Gemini draft grounded in 4 retrieved pairs |
| Escalation | Always escalate / never escalate | The rules, fed TF-IDF intents | The rules, fed Gemini intent + confidence |

**Metrics.**

- **Intent:** accuracy and macro-F1, with 95% bootstrap confidence intervals.
- **Escalation:** precision and recall against the gold labels, scored twice. Once end to end, and once with the gold intent, which separates classifier error from policy error.
- **Replies:** scored by an LLM judge (Gemini 3.5 Flash-Lite) on a 60-message subset. It rates each reply 1–5 on grounded, factual, tone, actionable, and overall.
  - **Reference:** Spotify's *actual* reply in that same held-out thread, which no system could have seen.
  - **Blind:** the judge never learns which system wrote a reply.

**Does the judge agree with a human?** The author scores 120 reply pairs blind (all 3 systems x 40 spot-check messages, ~40 per system) — no judge scores, no system names, rows shuffled. I report binned and quadratic-weighted Cohen's kappa, Spearman correlation, and exact/within-one agreement.

## 4. Results

Every number below comes from `results/eval_results.json`, reproduced offline from the committed replay cache with zero API calls. Golden set: 200 held-out messages, labels drafted against the written rubric and reviewed by the author. Brackets are 95% bootstrap confidence intervals.

### Intent classification

| System | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (always the majority training label) | 0.135 [0.090, 0.185] | 0.034 |
| Simple (TF-IDF + logistic regression on keyword labels) | 0.500 [0.425, 0.570] | 0.524 |
| **Gemini 3.5 Flash-Lite, taxonomy in prompt** | **0.860 [0.810, 0.900]** | **0.867** |

The intervals don't overlap, so the ordering is real on this sample. The LLM's margin over TF-IDF (+0.36) is the clearest result in the project. Two taxonomy boundaries were under-specified — `technical_bug` vs. `billing_subscription` for errors inside a payment/signup flow, and `content_catalog` vs. `other` for "the service isn't in my country" — visible as the two largest confusion-matrix clusters, six errors each. Fixing both, isolated from any other change, moved accuracy 0.785 → 0.850 with no new confusions elsewhere; two gold-label corrections found in the same error review (§5) brought it to 0.860.

### Reply quality (LLM judge, 1–5 overall)

| System | Overall |
|---|---|
| Trivial (one canned template per intent) | 4.02 [3.73, 4.30] |
| Simple (nearest historical reply, copied) | 3.97 [3.68, 4.22] |
| **Grounded generation** | **4.58 [4.40, 4.75]** |

Two things are more interesting than the ranking. First, the canned template beats copying a real Spotify reply — a retrieved reply is often a mismatched answer to a different problem, while a template is at least on-topic. Second, a canned template scoring 4.10 should provoke suspicion about the judge, and §6 shows that suspicion is justified.

### Escalation

| System | Precision | Recall |
|---|---|---|
| Always escalate | 0.510 | 1.000 |
| Never escalate | 0.000 | 0.000 |
| Rules on TF-IDF intents | 0.926 | 0.618 |
| **Rules on Gemini intents (end to end)** | **0.864 [0.793, 0.920]** | **0.931 [0.875, 0.979]** |
| Rules on gold intents (policy only) | 0.934 [0.882, 0.979] | 0.971 [0.931, 1.000] |

The end-to-end/policy-only gap isolates blame: with perfect intents the policy scores 0.934/0.971, so most remaining escalation error is classifier error, not rule error. Gold escalation rate is 102/200, so "always escalate" is a 0.510-precision baseline — the recall-1.0 strawman worth naming explicitly.

### Judge versus human

| Metric | Value |
|---|---|
| Cohen's kappa (binned) | 0.155 [−0.001, 0.305] |
| Quadratic weighted kappa | 0.349 |
| Spearman | 0.382 |
| Exact agreement | 0.425 |
| Within one point | 0.767 |
| Judge − human mean | +0.017 |

Per system (n=40 each): grounded **0.388**, nearest **−0.017**, trivial 0.181.

n=120, all 3 systems, not a rotation. An earlier n=40 round (one system per message) reported a wider, more optimistic kappa (0.342 [−0.074, 0.692]) built on only ~13 pairs per system; the full 120 halved the CI but *lowered* the point estimate — a small-sample overestimate corrected, not a regression. `nearest` collapsed to essentially chance-level agreement; §5 explains why.

## 5. Failure analysis

**1. The judge cannot tell a reply written for someone else from a good one.** Kappa −0.017 on the `nearest` system (n=40) — essentially chance, visible only once the sheet widened past its original ~13-per-system sample. `nearest` copies a real historical Spotify reply verbatim; when that reply was written for a *different* customer, it reads fluent and on-brand while being wrong for the person in front of it. Example (item h093): a customer named Diana reports unauthorized charges; the copied reply opens "Hey Dylan, that's not cool!" — Judge: 4, Human: 1. Four more `nearest` items in the same session show the identical pattern. *Hypothesis:* the rubric's four axes (grounded, factual, tone, actionable) are all satisfiable by borrowed fluency — none of them asks "was this written for the person who sent this message."

**2. Non-English messages were answered in English.** Three golden rows are non-English. Row 186's opening words are "Mereka pake bhs. Inggris" — "they use English" — a complaint about exactly that. The fix (one prompt instruction) produced an Indonesian reply, but aggregate judge score *fell* 4.60 → 4.57 at the time, because the rubric has no language axis. (The headline grounded score reported in §4, 4.58, reflects a later, unrelated taxonomy fix that changed which intent a handful of messages draft their reply against — not a reversal of this finding.) *Hypothesis:* the judge can't measure a dimension it wasn't told about, so a real improvement is invisible to the headline metric.

**3. The classifier is confidently wrong on multi-intent messages.** LLM self-reported confidence is almost always 0.85–1.00, so the confidence-below-0.55 escalation rule almost never fires. Messages that mix a bug with a billing complaint get one label at high confidence. *Hypothesis:* self-reported confidence from a single forward pass is close to useless for routing; real calibration needs held-out agreement or an ensemble.

**4. The keyword baseline is defeated by vocabulary, not by difficulty.** TF-IDF at 0.505 initially caught only 6% of messages containing obvious bug words ("error", "glitch", "stopped working") because the hand-written keyword lists missed them; after fixing, 84%. It also misread the Indonesian cancellation as `other`, which Gemini got right at 0.85 confidence. *Hypothesis:* a keyword baseline measures the author's vocabulary coverage as much as the task's difficulty, which is worth stating when quoting the +0.28 margin.

**5. Bug-versus-feature is the boundary humans disagree on.** Round 1 of the author's review changed 4 labels, three of which were this boundary (rows 60, 71, 78 — "raise the download limit", "block an artist", "download podcasts to a watch"): removed-on-purpose features and platform-parity gaps read as bugs to some reviewers and feature requests to others. Round 2 found 2 more while reading the classifier's remaining errors: a student-discount request mislabeled `feature_complaint` instead of `billing_subscription` (inconsistent with two other golden rows on the same topic), and a resolved 30-hour playback outage mislabeled `other` instead of `technical_bug` — both checked against evidence outside the model's own prediction, not accepted just because the model agreed (the self-review bias risk this carries is in §6). *Hypothesis:* label ambiguity, not model error, sets a ceiling on measurable intent accuracy here — and the gold set has two deliberate escalation exceptions (rows 43 and 913386) where the author overrode the bug-escalation rubric because the message gives a human nothing to act on.

## 6. What is misleading about my headline number

The headline is "86% intent accuracy, 4.58 reply quality, 0.86/0.93 escalation." Each is qualified:

- **The escalation numbers are the least trustworthy figure in the report.** The rubric was revised *after* the first results: bug reports were originally auto-handleable (Spotify itself asked for a DM on only 28% of plain bug reports), then reclassified as escalate-by-default, changing both the rules and the gold labels — precision went 0.79 → 0.86 as a result. **Under the original rubric, the same policy scores 0.55.** Scoring a policy against a rubric revised to match it is not independent evidence.
- **Reply quality rests on a judge that agrees with a human at kappa 0.155, on a CI [−0.001, 0.305] that still just touches zero** (n=120; §4 has the small-sample comparison). Weighted kappa (0.349) and Spearman (0.382) read somewhat better when near-misses get partial credit, but on the strict measure I cannot fully exclude chance-level agreement. Per system: a real signal on `grounded` (0.388), essentially useless on `nearest` (−0.017) — precisely where the baseline comparison most needs the judge to catch a reply written for someone else (§5).
- **One AI labeler, one human reviewer, across two review rounds** — not independent hand-labeling or an inter-annotator study. Claude drafted every gold label; the author changed 4 of 200 in round 1 (moving accuracy down, 0.790 → 0.785 — the expected direction for a real review) and 2 more in round 2, found while reading the classifier's own errors. Both round-2 corrections matched what the classifier had already predicted, moving accuracy up (0.850 → 0.860) — a real bias risk (relabeling toward the model's answer because it's in view, not because it's right) round 1 didn't carry. Both were checked against evidence outside the model's prediction before being accepted (§5), but a single self-reviewing reviewer cannot fully bound that bias.
- **The judge and reply generator share a model family** — "the model's idea of a good reply" is graded by "the model's judgment of a good reply."
- **n = 200, rare intents rarer still.** `cancellation_refund` has 9 rows; per-intent F1 for small classes is nearly meaningless, and macro-F1 inherits that noise.
- **The trivial classifier is weaker than it looks** — it predicts the majority *training* label ("other," 14% of gold), an easy baseline. The majority *gold* class (23%) would be a stronger one.
- **Reply quality is measured on 60 of 200 rows**; intent and escalation use all 200.
- **Retrieval size was not the constraint** — growing the index 3,800 → 5,400 (+42%) moved grounded quality by only ~0.02.
- **Everything is one brand, one language of support, opening messages only** — the long-thread escalation rule never fires in evaluation.

## 7. What I would do next, with one more week

1. **Fix the judge before trusting any reply number.** Add a language-match axis and an "is this action possible for this customer" axis, re-score, re-measure agreement. Target kappa > 0.6.
2. **Get a second labeler on 50 rows** — converts "one AI plus one reviewer" into a defensible inter-annotator bound.
3. **Re-run escalation under both rubrics and report both** — the 0.55-versus-0.87 spread is the most honest single artifact I could ship.
4. **Calibrate confidence.** Replace self-reported confidence with agreement across two cheap samples, then re-tune the 0.55 threshold on held-out data.
5. **Handle multi-intent messages** — multi-label, or route on the highest-risk intent present.
6. **Escalate outages and data loss**, the two gaps the current rules miss: "Spotify is down in Brazil" is an engineering signal, and "all my playlists are gone" may need a human restore.
7. **Test taxonomy portability** by relabeling 50 AmazonHelp threads with the same rubric, to see how much of this is Spotify-specific.
