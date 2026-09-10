# Decision Log

## Task 3 — Intent taxonomy validated against real data (2026-09-10)

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
