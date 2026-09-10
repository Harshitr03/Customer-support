import pandas as pd
from support_agent import data_prep

def _toy_df():
    # thread: customer(3,inbound) -> spotify(1) -> customer(2)
    return pd.DataFrame([
        {"tweet_id":3,"author_id":"115712","inbound":True,
         "created_at":"Tue Oct 31 22:10:00 +0000 2017",
         "text":"@SpotifyCares app keeps crashing",
         "response_tweet_id":"1","in_response_to_tweet_id":None},
        {"tweet_id":1,"author_id":"SpotifyCares","inbound":False,
         "created_at":"Tue Oct 31 22:10:47 +0000 2017",
         "text":"@115712 Try reinstalling the app.",
         "response_tweet_id":"2","in_response_to_tweet_id":"3"},
        {"tweet_id":2,"author_id":"115712","inbound":True,
         "created_at":"Tue Oct 31 22:11:45 +0000 2017",
         "text":"@SpotifyCares that worked, thanks",
         "response_tweet_id":None,"in_response_to_tweet_id":"1"},
        # unrelated non-spotify thread
        {"tweet_id":10,"author_id":"999","inbound":True,
         "created_at":"Tue Oct 31 20:00:00 +0000 2017",
         "text":"@AppleSupport help","response_tweet_id":None,
         "in_response_to_tweet_id":None},
    ])

def test_reconstruct_orders_by_time():
    threads = data_prep.reconstruct_threads(_toy_df())
    t = [x for x in threads if any(tr["author_id"]=="SpotifyCares" for tr in x["turns"])][0]
    # root_id must resolve to the actual thread-root tweet (3, the customer's
    # opening tweet with no parent), not min(seen) over the connected
    # component (which would incorrectly give 1, the Spotify reply).
    assert t["root_id"] == 3
    texts = [tr["text"] for tr in t["turns"]]
    assert texts[0].startswith("@SpotifyCares app keeps")
    assert "reinstalling" in texts[1]

def test_spotify_filter_and_fields():
    threads = data_prep.reconstruct_threads(_toy_df())
    sp = data_prep.spotify_threads(threads)
    assert len(sp) == 1
    assert sp[0]["customer_open"].startswith("@SpotifyCares app keeps")
    assert any("reinstalling" in r for r in sp[0]["spotify_replies"])
