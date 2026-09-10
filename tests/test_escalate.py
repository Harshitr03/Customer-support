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
