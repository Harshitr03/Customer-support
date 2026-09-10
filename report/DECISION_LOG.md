# Decision Log

The non-obvious decisions behind this agent, and why. Each one names what it costs if it's wrong. Detail on the taxonomy work is in the appendix.

1. **Brand: @SpotifyCares.** AmazonHelp and AppleSupport cover so many product lines that the intent taxonomy would sprawl. Airline accounts reply mostly "please DM us", which leaves nothing to ground a drafted reply on. Spotify's issues fall into a small set (bugs, account, billing, catalog, cancellation, feedback), and its replies contain real troubleshooting steps. *Cost if wrong:* the conclusions may not transfer to brands with broader or more transactional support.

2. **Intents come from the data, not from assumptions.** I started from a 7-intent hypothesis and read real messages before fixing it. `playback_bug` became `technical_bug` (a third of the bug reports weren't about playback). `account_login` became `account_access` (profile, email, and name changes). I also added a "classify by the customer's primary ask" rule so "charged twice, refund me" is `cancellation_refund`, not `billing_subscription`. *Cost if wrong:* a label boundary a human would draw differently.

3. **The history and the test set never share a thread.** I sampled 6,000 Spotify threads (seed 42) and split them 90/10 by thread root, so retrieval can never ground a test message on its own thread. While building this I fixed a bug in the reference code that picked the smallest tweet ID in a thread as its "root", which is sometimes Spotify's reply rather than the customer's opening tweet. A regression test now covers it.

4. **The golden set is sampled with free keyword labels, then labeled independent of any model prediction.** The original plan had the LLM pre-label the golden set, leaving most gold labels equal to the classifier's own output. That grades the classifier against itself. Instead, keyword rules (no API cost) pick a stratified sample of 200, and every row's gold intent, escalation, and reason were drafted by Claude (an AI assistant) reading the message against the written rubric -- never against the keyword prefill or any classifier's own prediction -- and are reviewed by the author before the evaluation is run, with the review record (how many labels the author changed) kept in `eval/golden_labeling_notes.md`. Gold intent differs from the keyword prefill on 101 of 200 rows. *Cost if wrong:* a single AI labeler with one human reviewer, not an independent panel (see the report's "misleading" section).

5. **Baselines never see test labels.** The trivial baseline predicts the majority class of the *training* labels; the plan's version took the majority of the test set. The TF-IDF baseline trains on keyword-derived labels from the 5,400-thread history, never on golden rows. I checked the keyword rules for recall, not just precision: they caught only 6% of messages with obvious bug-report words ("error", "glitch", "stopped working") until fixed, and 84% after. A baseline trained on the broken labels would have been a strawman.

6. **Escalation is a small rule layer, not a second LLM call.** It escalates on four rules, and every decision carries a human-readable reason:
   - risk language (legal, fraud, security)
   - sensitive intents (account, billing, cancellation)
   - classifier confidence below 0.55
   - 3+ unresolved back-and-forth rounds

   Auditable beats clever when a wrong "auto-handle" on a hacked account is the costliest failure. Auditing the risk rule turned up a trailing word-boundary bug that silently missed "unauthorized", "fraudulent", "scammed", "sued", and "charged back". *Cost if wrong:* the rules over-escalate borderline account questions.

7. **What "escalate" means in the golden set.** A human should take the message when it needs account-specific action (login, payment or plan change, a refund, cancelling an account the customer can't reach), when it's a security or fraud issue, or when it follows up an open DM case. General troubleshooting, catalog questions, feature feedback, how-tos, and praise are auto-handleable. This rubric differs from the rule policy on 29 of 200 rows. That gap is deliberate: it measures policy error.

8. **Retrieval uses Gemini embeddings and numpy cosine similarity, with no vector database.** A few thousand 768-dimensional vectors fit in memory. Keeping the index as a committed `.npy` file keeps the repo small and the retrieval step easy to read. 768 dimensions (instead of 3,072) keeps the committed index at about 6 MB.

9. **The retrieval index covers a capped slice of history (`KB_SIZE`, 2,000 threads at the time of the reported run).** Gemini's free tier allows 1,000 embeddings per day per project, so I embedded a prefix of the shuffled history rather than wait days or pay. The index is sized to what's embedded and grows by raising `KB_SIZE`. *Cost if wrong:* rarer issues have fewer close historical matches, which understates the grounded system's quality.

10. **The judge compares against Spotify's real reply to that same message.** The plan used the nearest retrieved reply as the judge's reference, which is exactly what the "copy the nearest reply" baseline outputs, so that baseline would have graded itself. The golden thread's own reply comes from the held-out pool, so no system could have seen it. The judge never learns which system wrote a reply.

11. **Replies are drafted from the classifier's *predicted* intent, not the gold intent.** This measures the agent end to end, as it would run in production. Using gold intents would hide the cost of classifier mistakes.

12. **Escalation is scored twice.** Once end to end (predicted intent and confidence), and once with gold intent. The difference separates "the classifier was wrong" from "the policy was wrong", which is what failure analysis needs.

13. **Human-vs-judge agreement is scored blind.** The human scores 40 reply pairs, rotated across the three systems, without seeing the judge's score or which system wrote the reply, so the judge can't anchor the human. I report agreement as binned Cohen's kappa, quadratic-weighted kappa, Spearman correlation, and exact and within-one agreement, with a bootstrap confidence interval.

14. **Results reproduce offline in seconds.** Every LLM and embedding call goes through a content-hash disk cache. The exact responses behind the reported numbers are committed as a read-only replay cache, and an offline guard makes the default demo unable to reach the API. A grader reproduces the headline numbers with no key and no cost. `--live` recomputes everything.

15. **Cost is controlled by design.**
    - Gemini 2.5 Flash runs with its thinking budget set to 0; the tasks are short.
    - Embedding batches are paced to the free tier's 100 per minute.
    - Retries wait as long as the server's `retryDelay` asks.
    - `python -m eval.run_eval --estimate` prints the exact number of API calls before anything is spent.

    The real run makes at most 442 text calls and 61 embeddings, counting
    `scripts/run_demo.py`'s stage-6 live example on top of the eval harness
    itself (200 classify + 60 grounded-reply + 180 judge = 440 text calls,
    plus 1 classify and 1 grounded-reply call for the demo message; 60
    query embeddings for the reply-quality subset, plus 1 for the demo
    message). The reply-subset's 60 query embeddings go out in a single
    batched `embed()` call, not one network round-trip per message.

---

## Appendix — taxonomy validation detail

### Task 3: intent taxonomy validated against real data (2026-09-10)

**Method:** Read a stratified sample of 40 real customer-open messages from
`corpus_pool` (`c.customer_open.sample(40, random_state=42)`) against the
working-hypothesis 7-intent taxonomy from the task brief.

**Finding:** The 7-intent structure held up well overall (no need to add or
remove intents — 4/40 messages landed cleanly in `other`, which is a healthy
catch-all rate, not evidence of a missing category). Two intents were too
narrowly defined for what actually showed up in the data:

1. **`playback_bug` → `technical_bug`.** About a third of the "bug" messages
   in the sample were not about playback specifically: memory usage growing
   the longer the app runs, ~20% CPU use while idle, a website "vote" button
   that didn't work, Hulu-account linking broken, a broken share-to-Twitter
   link, a black-screen/false "check your internet" error. A classifier
   prompted with a strict "songs won't play" definition would likely dump
   these into `other` or `feature_complaint`. Renamed to `technical_bug` and
   broadened the definition to cover app/website malfunctions generally
   (crashes, playback failures, resource usage, broken buttons/links/
   integrations), keeping the original playback examples and adding two
   examples from the broader class.

2. **`account_login` → `account_access`.** Several real messages were about
   managing account details rather than literally failing to log in:
   changing display name/profile URL, changing email address, and being
   billed in an account's registered country while living elsewhere are
   account-settings issues, not login failures — but they land with the
   same support workflow (account team, identity-adjacent). Renamed to
   `account_access` and extended the definition to explicitly include
   email/username/display-name changes, which was already implicit for
   "email change" in the original definition but not for other account
   fields.

`billing_subscription`, `content_catalog`, `cancellation_refund`,
`feature_complaint`, and `other` fit the sample as originally hypothesized
and were left unchanged (only `feature_complaint`'s and
`billing_subscription`'s definitions/examples were lightly extended to cite
concrete real-sample cases — feature *requests*, not just complaints; and
plan-change/currency confusion for billing — without changing their scope).

One edge case (an artist/creator asking about a "Spotify for Artists" gig
listing that disappeared from their profile, with no reply from artist
support) doesn't fit any of the six specific intents — it's not a regular
listener support flow. Left it under `other`; it was a single instance in
the 40-message sample, not a recurring cluster, so it doesn't warrant a
dedicated intent per the brief's guidance (revise only for clusters that
are large and systematically miscategorized).

**Source of truth:** `src/support_agent/taxonomy.py`.

### Task 3 fix: billing_subscription / cancellation_refund boundary (2026-09-10)

Code review flagged that `billing_subscription` ("unexpected/duplicate
charges") and `cancellation_refund` ("request a refund for a charge")
overlapped on charge-driven refund requests (e.g. "I was charged twice,
refund me"), with no rule to say which wins. Added an explicit
primary-ask rule to both definitions in `taxonomy.py`: if the customer
asks to cancel, get a refund, or get money back, classify as
`cancellation_refund` even if a charge is mentioned; `billing_subscription`
is only for charge/payment/plan problems where no cancellation or refund
is requested. Also tightened `feature_complaint` vs. `technical_bug` with
one clause (malfunctioning app vs. working-as-designed-but-disliked/missing
capability).
