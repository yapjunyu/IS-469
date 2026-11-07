"""Step 15: check the answer is supported by the student's notes.

The judge is a model from a different family than the tutor (an AI marking its own homework is
too generous). It reads the final answer, every passage retrieved in the turn with their pictures, and the student's record
the tutor was given (concepts, weak spots, wrong beliefs, session notes, open quiz), through
s4_trace.traced_chat (purpose "judge"), so its calls write llm rows and count toward the turn.
"""
from agent import s2_tools
from agent.prompts import FAITHFULNESS_PROMPT

# shortcut: hardcoded, move to config.yaml once the team agrees on it
JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"


def check(ctx, answer):
    """Judge the answer against the passages retrieved this turn. Returns {"supported": bool, "reason": str}."""
    # unique, a passage found again counting as newest, so the judge keeps the same pictures as the tutor
    passages = list({p["id"]: p for p in reversed(ctx.passages)}.values())[::-1]
    record = f"Student's record:\n{ctx.record}\n\n" if ctx.record else ""
    # said outright: with nothing to read, the judge passed course claims that nothing contradicted (#18)
    none = "<no-documents/> No passages were retrieved this turn.\n\n" if not passages else ""
    verdict = s2_tools.ask(ctx, "judge", FAITHFULNESS_PROMPT, passages, f"{none}{record}Answer to check:\n{answer}",
                           "{", model=JUDGE_MODEL)
    if not (isinstance(verdict, dict) and isinstance(verdict.get("supported"), bool)):
        # fail closed: an unreadable verdict must not let an unchecked answer through without a warning
        return {"supported": False, "reason": "The judge's reply had no valid verdict"}
    return {"supported": verdict["supported"], "reason": str(verdict.get("reason", ""))}
