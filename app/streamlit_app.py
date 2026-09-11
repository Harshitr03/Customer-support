"""Support Agent Console -- a small, optional, local demo UI.

Thin on purpose: layout and widgets only. Everything testable without
Streamlit lives in app/ui_data.py.

Offline is the default (zero network calls, everything served from
data/cache/ or the committed data/llm_cache/ replay cache): set
SUPPORT_AGENT_OFFLINE=1 *before* importing support_agent.pipeline, unless
the app was started with `-- --live`
(`streamlit run app/streamlit_app.py -- --live`), which needs
GEMINI_API_KEY and uses real API quota.

This demo tool is not the deliverable -- the evaluation is. It does not
touch data/, results/, or report/; it only reads them.
"""
import os
import sys

_LIVE = "--live" in sys.argv[1:]
if not _LIVE:
    os.environ.setdefault("SUPPORT_AGENT_OFFLINE", "1")

import streamlit as st  # noqa: E402

from support_agent import pipeline  # noqa: E402
from support_agent.llm_client import OfflineModeError, QuotaExhaustedError  # noqa: E402

from app import ui_data  # noqa: E402

st.set_page_config(page_title="Support Agent Console", layout="wide")

# ---------------------------------------------------------------------------
# Theme CSS: Figtree (falls back to system-ui offline), the verdict banner,
# track-list rows, pill buttons, and visible focus. Kept small and specific
# so rules don't cancel each other (see task brief's "Avoid" list: no
# ALL-CAPS labels, no gradient washes, no fade-in animation, amber carries
# meaning rather than decorating).
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Figtree', system-ui, -apple-system, "Segoe UI", sans-serif;
        font-variant-numeric: tabular-nums;
    }

    .sa-note {
        color: #9BA3AE;
        font-size: 13px;
    }

    .sa-verdict {
        border-radius: 16px;
        padding: 28px 32px;
        margin: 8px 0 20px 0;
    }
    .sa-verdict-label {
        font-size: 40px;
        line-height: 48px;
        font-weight: 700;
        color: #16181D;
    }
    .sa-verdict-reason {
        font-size: 15px;
        font-weight: 400;
        color: #16181D;
        margin-top: 8px;
    }
    .sa-verdict.sa-escalate { background: #F5A524; }
    .sa-verdict.sa-auto { background: #2FD470; }

    .sa-track-row {
        background: #1F232A;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .sa-track-rank {
        color: #9BA3AE;
        font-size: 15px;
        font-weight: 600;
    }
    .sa-track-customer {
        color: #F2F4F5;
        font-size: 15px;
    }
    .sa-track-reply {
        color: #9BA3AE;
        font-size: 13px;
        margin-top: 4px;
    }
    .sa-track-score {
        color: #9BA3AE;
        font-size: 13px;
        font-variant-numeric: tabular-nums;
    }

    .sa-reply-card {
        background: #1F232A;
        border-radius: 12px;
        padding: 16px;
        font-size: 16px;
        color: #F2F4F5;
    }
    .sa-reply-meta {
        color: #9BA3AE;
        font-size: 13px;
        font-variant-numeric: tabular-nums;
        margin-top: 8px;
    }

    div[data-testid="stButton"] > button {
        border-radius: 999px;
    }

    :focus-visible {
        outline: 2px solid #2FD470 !important;
        outline-offset: 2px !important;
    }

    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Support agent console")
st.markdown(f'<p class="sa-note">{ui_data.NOT_AFFILIATED_NOTE}</p>', unsafe_allow_html=True)

tab_try, tab_results = st.tabs(["Try a message", "Results"])

# ---------------------------------------------------------------------------
# Tab 1: Try a message
# ---------------------------------------------------------------------------
with tab_try:
    golden = ui_data.load_golden()
    cached_examples = ui_data.find_cached_examples(golden)

    st.subheader("Pick a message")
    st.caption(
        f"{len(cached_examples)} golden-set messages have a fully cached response "
        "(offline, no network calls)."
    )

    example_labels = ["(type your own)"] + [
        f"#{ex['root_id']} — {ui_data.truncate(ex['message'], 70)}" for ex in cached_examples
    ]
    chosen_label = st.selectbox("Choose a cached example", options=example_labels, key="example_picker")

    custom_message = st.text_area(
        "Or type your own message", key="custom_message_input", height=100,
        placeholder="e.g. the app keeps crashing when I hit play",
    )

    if chosen_label != "(type your own)" and not custom_message.strip():
        idx = example_labels.index(chosen_label) - 1
        message_to_run = cached_examples[idx]["message"]
    else:
        message_to_run = custom_message.strip()

    run_clicked = st.button("Run the agent", key="run_agent_button", type="primary")

    if run_clicked:
        if not message_to_run:
            st.session_state["sa_error"] = "Type a message or pick a cached example first."
            st.session_state["sa_result"] = None
        else:
            try:
                st.session_state["sa_result"] = pipeline.handle(message_to_run)
                st.session_state["sa_error"] = None
            except OfflineModeError:
                st.session_state["sa_result"] = None
                st.session_state["sa_error"] = ui_data.OFFLINE_ERROR_MESSAGE
            except QuotaExhaustedError as exc:
                st.session_state["sa_result"] = None
                st.session_state["sa_error"] = str(exc)

    error = st.session_state.get("sa_error")
    result = st.session_state.get("sa_result")

    if error:
        st.error(error)

    if result:
        verdict = ui_data.format_verdict(result["escalate"], result["reason"])
        banner_class = "sa-escalate" if result["escalate"] else "sa-auto"
        st.markdown(
            f'<div class="sa-verdict {banner_class}">'
            f'<div class="sa-verdict-label">{verdict["label"]}</div>'
            f'<div class="sa-verdict-reason">{verdict["reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Intent")
        col_label, col_meter = st.columns([1, 2])
        with col_label:
            st.write(ui_data.format_intent(result["intent"]))
        with col_meter:
            st.progress(min(max(result["confidence"], 0.0), 1.0))
            st.caption(f"Confidence: {result['confidence']:.2f}")

        st.subheader("Drafted reply")
        reply = result["reply"]
        st.markdown(f'<div class="sa-reply-card">{reply}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sa-reply-meta">{len(reply)} / {ui_data.REPLY_CHAR_LIMIT} characters</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Grounded on")
        rows = ui_data.format_evidence(result.get("evidence") or [])
        if not rows:
            st.caption("No retrieval evidence for this message.")
        for row in rows:
            bar_pct = int(min(max(row["score"], 0.0), 1.0) * 100)
            st.markdown(
                f'<div class="sa-track-row">'
                f'<span class="sa-track-rank">{row["rank"]}.</span> '
                f'<span class="sa-track-customer">{row["customer"]}</span>'
                f'<div class="sa-track-reply">{row["reply"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(bar_pct / 100)
            st.caption(f"Similarity: {row['score']:.3f}")

# ---------------------------------------------------------------------------
# Tab 2: Results
# ---------------------------------------------------------------------------
with tab_results:
    loaded = ui_data.load_results()

    if loaded is None:
        st.info(ui_data.EMPTY_RESULTS_MESSAGE)
    else:
        eval_results = loaded["eval_results"]

        st.subheader("Intent classification")
        st.dataframe(ui_data.intent_headline_table(eval_results), hide_index=True)

        st.subheader("Reply quality")
        st.dataframe(ui_data.reply_quality_headline_table(eval_results), hide_index=True)

        st.subheader("Escalation")
        st.dataframe(ui_data.escalation_headline_table(eval_results), hide_index=True)

        st.subheader("Per-intent F1 (Gemini classifier)")
        f1_df = ui_data.per_intent_f1(eval_results)
        st.bar_chart(f1_df.set_index("intent")["f1"], horizontal=True, color="#2FD470")

        st.subheader("Compare replies")
        reply_rows = loaded["reply_rows"]
        options = ui_data.compare_reply_options(reply_rows)
        if not options:
            st.caption("No replies in the reply-quality subset.")
        else:
            labels = [f"#{o['root_id']} — {o['message']}" for o in options]
            picked = st.selectbox("Choose a message", options=labels, key="compare_picker")
            picked_root_id = options[labels.index(picked)]["root_id"]
            comparison = ui_data.compare_replies(reply_rows, picked_root_id)
            st.caption(f"Reference (Spotify's actual reply): {comparison['reference']}")
            cols = st.columns(len(comparison["systems"]) or 1)
            for col, (system, data) in zip(cols, comparison["systems"].items()):
                with col:
                    st.write(system)
                    st.markdown(f'<div class="sa-reply-card">{data["reply"]}</div>', unsafe_allow_html=True)
                    for k, v in data["scores"].items():
                        st.caption(f"{k}: {v}")

        st.subheader("Escalation mistakes")
        mistake_filter_label = st.radio(
            "Filter",
            options=["All", "False escalations", "Missed escalations"],
            horizontal=True,
            key="mistake_filter",
        )
        filter_map = {
            "All": "all",
            "False escalations": "false_escalations",
            "Missed escalations": "missed_escalations",
        }
        mistakes = ui_data.escalation_mistakes(
            loaded["classification_rows"], filter_map[mistake_filter_label])
        st.dataframe(mistakes, hide_index=True)
