"""Step 15: check the answer is supported by the cited passages.

Uses s4_trace.traced_chat with a judge model from a different family than the tutor.
"""


def check(answer, chunks, judge_model):
    """Return {"supported": bool, "reason": str}."""
    raise NotImplementedError
