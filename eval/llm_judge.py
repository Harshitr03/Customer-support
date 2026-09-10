"""Gemini LLM-as-judge: score a drafted reply 1-5 on a fixed rubric.

The reference reply passed in is the REAL historical
Spotify reply sent to this exact customer thread (joined by root_id in
eval/run_eval.py), never the output of any system under evaluation -- so no
baseline is ever judged against its own output. The prompt below shows the
judge that reference purely as a guide to what a helpful, on-brand answer
looks like for this kind of message, not as an answer key to match
verbatim, and never reveals which system produced the drafted reply.
"""
import json
import logging

from support_agent import llm_client

logger = logging.getLogger(__name__)

JUDGE_KEYS = ("grounded", "factual", "tone", "actionable", "overall")

_RUBRIC = """You are evaluating a drafted reply from a Spotify customer-support agent \
(@SpotifyCares) on Twitter. Score the DRAFTED REPLY below from 1-5 (integers) on each \
axis:
- grounded: consistent with how Spotify has historically handled similar issues
- factual: makes no invented or unsupported claims about the customer's account or \
Spotify's product
- tone: empathetic, on-brand, concise
- actionable: gives the customer a concrete next step
- overall: holistic quality of the drafted reply

For reference only, here is a REAL reply Spotify sent to THIS SAME customer in this \
thread. Treat it as a guide to what a helpful, on-brand answer looks like for this \
kind of message -- it is not an answer key, and the drafted reply does not need to \
match it verbatim.
Real historical Spotify reply: {reference}

Customer message: {message}

Drafted reply to score: {reply}

Respond ONLY with JSON keys grounded, factual, tone, actionable, overall, each an \
integer 1-5."""


def judge_reply(message: str, reply: str, reference: str) -> dict:
    """Score `reply` on JUDGE_KEYS, 1-5 each, plus a `parse_ok` flag: False
    when the judge's JSON response failed to parse, wasn't an object, or
    was missing/had a non-numeric value for any required key -- in every
    one of those cases the affected score(s) still fall back to a clamped
    default (3) rather than raising, but `parse_ok` tells the caller not to
    trust that row's numbers for aggregate stats (A6)."""
    raw = llm_client.generate(
        _RUBRIC.format(reference=reference, message=message, reply=reply),
        json_mode=True, temperature=0.0)
    parse_ok = True
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("judge_reply: unparseable JSON response from judge")
        obj = {}
        parse_ok = False
    if not isinstance(obj, dict):
        logger.warning("judge_reply: judge response was not a JSON object")
        obj = {}
        parse_ok = False
    out = {}
    for k in JUDGE_KEYS:
        if k not in obj:
            parse_ok = False
            v = 3
        else:
            try:
                if isinstance(obj[k], bool):
                    raise TypeError
                v = int(round(float(obj[k])))
            except (ValueError, TypeError):
                parse_ok = False
                v = 3
        out[k] = max(1, min(5, v))
    out["parse_ok"] = parse_ok
    logger.debug("judge_reply: scores=%s parse_ok=%s", out, parse_ok)
    return out
