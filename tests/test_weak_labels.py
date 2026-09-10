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
    # "I'm Premium" is common in bug reports and must not get pulled into
    # billing_subscription by a bare "premium" keyword.
    assert wl.weak_label("I'm Premium and the app keeps crashing") == "technical_bug"


def test_bug_report_vocabulary_routes_to_technical_bug():
    # A recall check against the real corpus found these phrasings were
    # falling through to other/content_catalog because "error", "bug",
    # "glitch", "broken", "freeze(ing)", "doesn't work", "stopped working",
    # "not playing", and "won't load" weren't in KEYWORDS["technical_bug"].
    assert wl.weak_label("an error occurred") == "technical_bug"
    assert wl.weak_label(
        "my app has an error, and nothing is loading") == "technical_bug"
    assert wl.weak_label(
        "CarPlay integration indeed is extremely buggy") == "technical_bug"
    assert wl.weak_label(
        "can you fix this bug on the app where my music automatically "
        "pauses") == "technical_bug"
    assert wl.weak_label(
        "a bunch of songs have stopped working on my Iphone 7") == "technical_bug"
