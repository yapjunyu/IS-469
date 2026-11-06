"""Step 14: JSONL trace, one line per event, three kinds.

Every row starts with the same fields, in this order:
    kind, ts (ISO 8601, local time with offset), student, turn, step
Then, by kind:
- llm:   purpose, model, tokens {in, out}, cached
         One per LLM call anywhere. purpose is one of: tutor, forced_answer, concepts,
         quiz, grading, judge, regenerate (later search-again). Only llm rows carry tokens.
         Concept extraction runs at upload, outside a chat turn, so its row has turn 0.
- tool:  tool, inputs (parsed object, or the raw string if the JSON was bad),
         ok, error, duration_ms, result_summary (first 200 chars of the result)
- final: steps (including a forced answer and a regeneration), cap_hit, searches, tokens {in, out} (sum of the turn's llm rows),
         duration_ms (whole turn), faithfulness (the judge's last verdict {supported, reason},
         null with the check off), answer_preview (first 100 chars). Always the turn's last row.
"""

import json
from datetime import datetime
from pathlib import Path

from agent import s1_llm

# shortcut: hardcoded, move to config.yaml once the team agrees on it
TRACE_PATH = Path("logs/trace.jsonl")


def log_row(ctx, kind, **fields):
    """Append one row of the given kind, stamped with who/where from the turn context."""
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"kind": kind, "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
           "student": ctx.student_id, "turn": ctx.turn, "step": ctx.step, **fields}
    with TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def traced_chat(ctx, purpose, messages, model, **kwargs):
    """Call s1_llm.chat, add its tokens to the turn's totals and write an llm row."""
    reply = s1_llm.chat(messages, model, **kwargs)
    for k in ctx.tokens:
        ctx.tokens[k] += reply["usage"][k]
    log_row(ctx, "llm", purpose=purpose, model=model, tokens=reply["usage"], cached=reply["cached"])
    return reply
