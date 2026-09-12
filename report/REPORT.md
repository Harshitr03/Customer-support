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
- **A UI, or a second independent judge model.** Both were out of scope for the time available. The single judge is a known weakness (§5).

## 2. The system

```
customer tweet
   │
   ├─► classify      Gemini 3.5 Flash-Lite, taxonomy in the prompt → intent + confidence
   │
   ├─► retrieve      Gemini embeddings (768-d), cosine top-4 over 3,800 historical
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
- **Differs from the keyword prefill:** gold intent disagrees with the prefill on 101 of 200 rows.
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

**Does the judge agree with a human?** The author scores 40 reply pairs blind: no judge scores, no system names, rows shuffled, rotated across the three systems. I report binned and quadratic-weighted Cohen's kappa, Spearman correlation, and exact and within-one agreement.

## 4. Results

Every number below comes from `results/eval_results.json`, reproduced offline from the committed replay cache with zero API calls. Golden set: 200 held-out messages, labels drafted against the written rubric and reviewed by the author. Brackets are 95% bootstrap confidence intervals.

### Intent classification

| System | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (always the majority training label) | 0.140 [0.095, 0.190] | 0.035 |
| Simple (TF-IDF + logistic regression on keyword labels) | 0.505 [0.430, 0.575] | 0.528 |
| **Gemini 3.5 Flash-Lite, taxonomy in prompt** | **0.785 [0.725, 0.840]** | **0.786** |

The intervals don't overlap, so the ordering is real on this sample. The LLM's margin over TF-IDF (+0.28) is the clearest result in the project.

### Reply quality (LLM judge, 1–5 overall)

| System | Overall |
|---|---|
| Trivial (one canned template per intent) | 4.10 [3.83, 4.37] |
| Simple (nearest historical reply, copied) | 3.97 [3.68, 4.22] |
| **Grounded generation** | **4.57 [4.38, 4.72]** |

Two things are more interesting than the ranking. First, the canned template beats copying a real Spotify reply — a retrieved reply is often a mismatched answer to a different problem, while a template is at least on-topic. Second, a canned template scoring 4.10 should provoke suspicion about the judge, and §6 shows that suspicion is justified.

### Escalation

| System | Precision | Recall |
|---|---|---|
| Always escalate | 0.505 | 1.000 |
| Never escalate | 0.000 | 0.000 |
| Rules on TF-IDF intents | 0.926 | 0.624 |
| **Rules on Gemini intents (end to end)** | **0.864 [0.792, 0.924]** | **0.941 [0.891, 0.981]** |
| Rules on gold intents (policy only) | 0.942 [0.892, 0.981] | 0.970 [0.930, 1.000] |

The end-to-end/policy-only gap isolates blame: with perfect intents the policy scores 0.942/0.970, so most remaining escalation error is classifier error, not rule error. Gold escalation rate is 101/200, so "always escalate" is a 0.505-precision baseline — the recall-1.0 strawman worth naming explicitly.

### Judge versus human

| Metric | Value |
|---|---|
| Cohen's kappa (binned) | 0.342 [−0.074, 0.692] |
| Quadratic weighted kappa | 0.293 |
| Spearman | 0.245 |
| Exact agreement | 0.525 |
| Within one point | 0.800 |
| Judge − human mean | −0.35 |

Per system: grounded **0.755**, nearest 0.304, trivial **−0.120**.

## 5. Failure analysis

**1. The judge is worse than chance on canned templates.** Kappa −0.120 on the trivial system. It rewards fluent, polite, well-formed replies; the author scored the same replies on whether they actually answered the customer. Example (item h09): a locked-out Indonesian customer is told, in English, to cancel under Account > Subscription — judge 4, human 1. *Hypothesis:* the rubric's four axes (grounded, factual, tone, actionable) are all satisfiable by a generic template, because "actionable" doesn't require the action to be *possible for this customer*.

**2. Non-English messages were answered in English.** Three golden rows are non-English. Row 186's opening words are "Mereka pake bhs. Inggris" — "they use English" — a complaint about exactly that. The fix (one prompt instruction) now produces an Indonesian reply, but aggregate judge score *fell* 4.60 → 4.57, because the rubric has no language axis. *Hypothesis:* the judge can't measure a dimension it wasn't told about, so a real improvement is invisible to the headline metric.

**3. The classifier is confidently wrong on multi-intent messages.** LLM self-reported confidence is almost always 0.85–1.00, so the confidence-below-0.55 escalation rule almost never fires. Messages that mix a bug with a billing complaint get one label at high confidence. *Hypothesis:* self-reported confidence from a single forward pass is close to useless for routing; real calibration needs held-out agreement or an ensemble.

**4. The keyword baseline is defeated by vocabulary, not by difficulty.** TF-IDF at 0.505 initially caught only 6% of messages containing obvious bug words ("error", "glitch", "stopped working") because the hand-written keyword lists missed them; after fixing, 84%. It also misread the Indonesian cancellation as `other`, which Gemini got right at 0.85 confidence. *Hypothesis:* a keyword baseline measures the author's vocabulary coverage as much as the task's difficulty, which is worth stating when quoting the +0.28 margin.

**5. Bug-versus-feature is the boundary humans disagree on.** The author's review changed 4 labels, three of which were this boundary (rows 60, 71, 78 — "raise the download limit", "block an artist", "download podcasts to a watch"). Removed-on-purpose features and platform-parity gaps read as bugs to some reviewers and as feature requests to others. *Hypothesis:* label ambiguity, not model error, sets a ceiling on measurable intent accuracy here — and the gold set has one deliberate exception (row 43) where the author overrode the rubric for content-free venting.

## 6. What is misleading about my headline number

The headline is "79% intent accuracy, 4.57 reply quality, 0.86/0.94 escalation." Each is qualified:

- **The escalation numbers are the least trustworthy figure in the report.** The rubric was revised *after* the first results: bug reports were originally auto-handleable (on the brand's own evidence — Spotify asked for a DM on 28% of plain bug reports), then reclassified as escalate-by-default, changing both the rules and the gold labels. Precision went 0.79 → 0.87 as a result. **Under the original rubric the same policy scores 0.55.** Scoring a policy against a rubric revised to match it is not independent evidence.
- **Reply quality rests on a judge that agrees with a human at kappa 0.342, whose CI crosses zero.** With n=40 I cannot exclude chance-level agreement. The judge is trustworthy on the grounded system (0.755) and anti-correlated on canned templates (−0.120) — precisely where the baseline comparison needs it most.
- **One AI labeler, one human reviewer.** Gold labels were drafted by Claude and reviewed by the author, who changed 4 of 200. That is not independent hand-labeling and not an inter-annotator agreement study. The review moved accuracy *down* (0.790 → 0.785), which suggests it was real, but a single reviewer cannot bound their own bias.
- **The judge and the reply generator share a model family**, so "the model's idea of a good reply" is graded by "the model's judgment of a good reply."
- **n = 200, and rare intents are rarer still.** `cancellation_refund` has 9 rows. Per-intent F1 for the small classes is nearly meaningless, and macro-F1 inherits that noise.
- **The trivial classifier is weaker than it looks.** It predicts the majority *training* label ("other"), which is only 14% of gold — an easy baseline to beat. A stronger trivial baseline would be the majority gold class (23%).
- **Reply quality is measured on 60 of 200 rows**, the subset that was drafted and judged; intent and escalation use all 200.
- **Retrieval size was not the constraint.** Growing the index 3,800 → 5,400 (+42%) moved grounded quality by ~0.02. Either retrieval was already sufficient or the judge can't resolve the difference.
- **Everything is one brand, one language of support, opening messages only.** The long-thread escalation rule never fires in evaluation, because every golden row is a conversation's first message.

## 7. What I would do next, with one more week

1. **Fix the judge before trusting any reply number.** Add a language-match axis and an "is this action possible for this customer" axis; re-score; re-measure agreement. Target kappa > 0.6 before quoting reply quality at all.
2. **Get a second labeler on 50 rows** and report real inter-annotator agreement, which converts "one AI plus one reviewer" into a defensible bound.
3. **Re-run escalation under both rubrics and report both.** The 0.55-versus-0.87 spread is the most honest single artifact I could ship.
4. **Calibrate confidence.** Replace self-reported confidence with agreement across two cheap samples, then re-tune the 0.55 threshold on held-out data so the low-confidence rule actually fires.
5. **Handle multi-intent messages**, either as multi-label or by routing on the highest-risk intent present.
6. **Escalate outages and data loss**, the two gaps the current rules miss: "Spotify is down in Brazil" is an engineering signal, and "all my playlists are gone" may need a human restore.
7. **Test taxonomy portability** by relabeling 50 AmazonHelp threads with the same rubric, to see how much of this is Spotify-specific.
