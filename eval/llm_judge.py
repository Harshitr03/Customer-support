"""Gemini LLM-as-judge: score a drafted reply 1-5 on a fixed rubric.

Controller ruling 2: the reference reply passed in is the REAL historical
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
    raw = llm_client.generate(
        _RUBRIC.format(reference=reference, message=message, reply=reply),
        json_mode=True, temperature=0.0)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("judge_reply: unparseable JSON response from judge")
        obj = {}
    if not isinstance(obj, dict):
        logger.warning("judge_reply: judge response was not a JSON object")
        obj = {}
    out = {}
    for k in JUDGE_KEYS:
        try:
            v = int(round(float(obj.get(k, 3))))
        except (ValueError, TypeError):
            v = 3
        out[k] = max(1, min(5, v))
    logger.debug("judge_reply: scores=%s", out)
    return out
