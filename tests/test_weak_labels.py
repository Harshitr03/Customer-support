from support_agent import weak_labels as wl


def test_keyword_routing():
    assert wl.weak_label("I was charged twice for premium") == "billing_subscription"
    assert wl.weak_label("how do I cancel my subscription") == "cancellation_refund"
    assert wl.weak_label("the app keeps crashing") == "technical_bug"
    assert wl.weak_label("random unrelated hello") == "other"


def test_account_access_routing():
    assert wl.weak_label("I can't log into my account") == "account_access"
    assert wl.weak_label("someone hacked my Spotify account") == "account_access"


def test_bare_premium_does_not_win_over_bug_report():
    # Controller ruling 2: "I'm Premium" is common in bug reports and must
    # not get pulled into billing_subscription by a bare "premium" keyword.
    assert wl.weak_label("I'm Premium and the app keeps crashing") == "technical_bug"
