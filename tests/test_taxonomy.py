from support_agent import taxonomy


def test_taxonomy_wellformed():
    names = taxonomy.INTENT_NAMES
    assert len(names) == len(set(names))        # unique
    assert taxonomy.OTHER in names              # catch-all present
    assert 6 <= len(names) <= 8
    for it in taxonomy.INTENTS:
        assert it.definition.strip()
        assert len(it.examples) >= 2 or it.name == taxonomy.OTHER


def test_describe_mentions_every_intent():
    block = taxonomy.describe()
    for n in taxonomy.INTENT_NAMES:
        assert n in block
