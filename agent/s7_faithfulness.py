"""Step 15: check the answer is supported by the student's notes.

The judge is a model from a different family than the tutor (an AI marking its own homework is
too generous). It reads the final answer and every passage retrieved in the turn, with their pictures, through
s4_trace.traced_chat (purpose "judge"), so its calls write llm rows and count toward the turn.
"""
from agent import s2_tools
from agent.prompts import FAITHFULNESS_PROMPT

# shortcut: hardcoded, move to config.yaml once the team agrees on it
JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"


def check(ctx, answer):
    """Judge the answer against the passages retrieved this turn. Returns {"supported": bool, "reason": str}."""
    # unique, a passage found again counting as newest, so the judge keeps the same pictures as the tutor
    ids = list(dict.fromkeys(reversed(ctx.passage_ids)))[::-1]
    passages = ctx.passages_fn(ctx.student_id, ids) if ids else []
    verdict = s2_tools.ask(ctx, "judge", FAITHFULNESS_PROMPT, passages, f"Answer to check:\n{answer}", "{",
                           model=JUDGE_MODEL)
    if not (isinstance(verdict, dict) and isinstance(verdict.get("supported"), bool)):
        # fail closed: an unreadable verdict must not let an unchecked answer through without a warning
        return {"supported": False, "reason": "The judge's reply had no valid verdict"}
    return {"supported": verdict["supported"], "reason": str(verdict.get("reason", ""))}
