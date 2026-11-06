"""Step 6: the notebook's concept list, built from the files at upload.

- extract_concepts(): the app calls it once per uploaded file. One LLM call returns the
  file's concepts as a JSON list; the known concepts go in the prompt so the LLM reuses them.
- learner.remove_file(): the app calls it when a file is deleted.
"""
import json

from agent import s4_trace
from agent.memory import learner
from agent.prompts import CONCEPTS_PROMPT
from agent.s2_tools import TurnContext
from agent.s3_loop import MODEL

# shortcut: reads only the start of a file to stay under Groq's free-tier token limit, so concepts
# late in a long PDF are missed; upgrade to per-passage extraction on a paid tier
MAX_CHARS = 12000


def _parse(content):
    """The JSON list of names in the reply, or [] if there is no valid one."""
    try:
        names = json.loads(content[content.find("["):content.rfind("]") + 1])
    except ValueError:
        return []
    return names if isinstance(names, list) and all(isinstance(n, str) for n in names) else []


def extract_concepts(student_id, file, text):
    """Find one file's concepts and add them to the notebook's list. Returns the names stored."""
    ctx = TurnContext(student_id=student_id)  # turn 0: upload, not a chat turn
    known = learner.concepts(student_id)
    messages = [{"role": "system", "content": CONCEPTS_PROMPT.format(known=", ".join(known) or "none yet")},
                {"role": "user", "content": text[:MAX_CHARS]}]
    names = _parse(s4_trace.traced_chat(ctx, "concepts", messages, MODEL)["message"].get("content") or "")
    return learner.add_concepts(student_id, file, names)
