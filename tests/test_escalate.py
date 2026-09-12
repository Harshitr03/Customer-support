from support_agent import escalate as esc


def test_escalate_by_intent():
    # Message deliberately avoids risk-language words (e.g. "hacked") so this
    # isolates the sensitive-intent rule from the risk-language rule, which
    # has its own dedicated tests below.
    e, reason = esc.decide("account_access", 0.99, [], "I can't log into my account")
    assert e and "sensitive intent" in reason


def test_escalate_low_confidence():
    e, reason = esc.decide("feature_complaint", 0.20, [], "meh")
    assert e and "confidence" in reason


def test_escalate_long_unresolved_thread():
    turns = [{"inbound": True}, {"inbound": False}, {"inbound": True},
             {"inbound": False}, {"inbound": True}, {"inbound": False}, {"inbound": True}]
    e, reason = esc.decide("feature_complaint", 0.95, turns, "still broken")
    assert e and "back-and-forth" in reason


def test_auto_handle():
    e, reason = esc.decide("feature_complaint", 0.95, [{"inbound": True}], "love it")
    assert not e and reason


def test_safety_language_escalates():
    e, reason = esc.decide("other", 0.99, [], "I will sue you, this is fraud")
    assert e and "risk language" in reason


def test_unauthorized_charge_escalates_with_risk_reason():
    # Regression: the reference regex's trailing \b after the "unauthoriz" stem
    # required a word boundary immediately following it, which "unauthorized"
    # (ends in "ed") never satisfies. This is the exact phrase fraud victims use.
    e, reason = esc.decide("billing_subscription", 0.95, [], "there's an unauthorized charge on my card")
    assert e and "risk language" in reason


def test_hacked_account_message_escalates():
    e, reason = esc.decide("other", 0.95, [], "someone hacked my account")
    assert e and "risk language" in reason


def test_no_risk_language_false_positive():
    e, reason = esc.decide("feature_complaint", 0.95, [], "the new UI is terrible")
    assert not e


def test_fraudulent_charge_escalates_with_risk_reason():
    # Regression: \bfraud\b never matched "fraudulent" (same stem/word-boundary
    # bug class as "unauthoriz" -> "unauthorized"). 5 real occurrences in
    # data/interim/corpus_pool.parquet were false negatives before this fix.
    e, reason = esc.decide("other", 0.95, [], "I noticed a fraudulent charge on my account")
    assert e and "risk language" in reason


def test_scammed_message_escalates_with_risk_reason():
    e, reason = esc.decide("other", 0.95, [], "they scammed me out of my subscription fee")
    assert e and "risk language" in reason


def test_suing_message_escalates_with_risk_reason():
    e, reason = esc.decide("other", 0.95, [], "I'm suing you over this")
    assert e and "risk language" in reason


def test_disputed_charge_escalates_with_risk_reason():
    e, reason = esc.decide("billing_subscription", 0.95, [], "I disputed the charge with my bank")
    assert e and "risk language" in reason


def test_charged_back_message_escalates_with_risk_reason():
    e, reason = esc.decide("billing_subscription", 0.95, [], "I charged back the payment")
    assert e and "risk language" in reason


# --- tried_fixes rule (Task 15) ---

def test_tried_everything_escalates_with_tried_fixes_reason():
    # Intent is feature_complaint (not technical_bug) so this isolates the
    # tried_fixes rule from the sensitive_intent rule, which now also fires
    # on technical_bug and would otherwise mask the reason this test checks.
    e, reason = esc.decide(
        "feature_complaint", 0.9, [],
        "I have tried everything! My Spotify keeps connecting to other devices",
    )
    assert e and "standard troubleshooting already failed" in reason


def test_already_tried_browsers_escalates_with_tried_fixes_reason():
    e, reason = esc.decide(
        "feature_complaint", 0.9, [],
        "I already tried 2 different browsers and it still won't load",
    )
    assert e and "standard troubleshooting already failed" in reason


def test_still_happening_escalates_with_tried_fixes_reason():
    e, reason = esc.decide(
        "feature_complaint", 0.9, [],
        "the app keeps pausing, still happening after the update",
    )
    assert e and "standard troubleshooting already failed" in reason


def test_curly_apostrophe_ive_tried_escalates_with_tried_fixes_reason():
    e, reason = esc.decide(
        "feature_complaint", 0.9, [],
        "I’ve tried reinstalling it",
    )
    assert e and "standard troubleshooting already failed" in reason


def test_single_fix_mention_does_not_trigger_tried_fixes():
    # Intent switched from technical_bug to feature_complaint: technical_bug
    # is now itself a sensitive (auto-escalating) intent, which would make
    # this case escalate for the wrong reason and destroy the point of the
    # test -- that a single, non-exhausted fix mention does not trip the
    # tried_fixes rule.
    e, reason = esc.decide(
        "feature_complaint", 0.9, [],
        "I reinstalled the app and now it crashes",
    )
    assert not e and "standard troubleshooting already failed" not in reason


def test_feature_complaint_does_not_trigger_tried_fixes():
    e, reason = esc.decide("feature_complaint", 0.9, [], "the new shuffle is terrible")
    assert not e and "standard troubleshooting already failed" not in reason


def test_tried_fixes_language_loses_to_risk_language():
    e, reason = esc.decide(
        "other", 0.9, [],
        "I already tried everything, I'm going to sue",
    )
    assert e and "risk language" in reason


def test_tried_fixes_language_loses_to_sensitive_intent():
    e, reason = esc.decide("billing_subscription", 0.9, [], "tried everything")
    assert e and "sensitive intent" in reason


def test_tried_fixes_wins_over_low_confidence():
    # feature_complaint (not technical_bug), same reasoning as above -- keeps
    # this isolated to the tried_fixes-vs-low_confidence rule ordering rather
    # than tripping the sensitive_intent rule first.
    e, reason = esc.decide("feature_complaint", 0.2, [], "tried everything")
    assert e and "standard troubleshooting already failed" in reason


# --- technical_bug is a sensitive intent (Task 18) ---

def test_technical_bug_escalates_by_intent():
    # High confidence, no risk/tried-fixes language: isolates the
    # sensitive_intent rule for technical_bug, mirroring
    # test_escalate_by_intent above for account_access.
    e, reason = esc.decide("technical_bug", 0.9, [], "the app crashes every time I open it")
    assert e and "sensitive intent" in reason
