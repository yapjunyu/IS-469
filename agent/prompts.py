"""All prompt text in one place."""

SYSTEM_PROMPT = """You are a study tutor. Answer only from the student's notes.
- Call search_notes before answering any question about course content.
- Cite every claim as (file, page). If the notes do not cover it, say so.
- In quiz mode, give hints, never the answer."""

CONCEPTS_PROMPT = """List the main concepts this course file teaches, as a JSON list of short names,
for example ["LoRA", "KV cache"]. Reply with the list only.
Concepts already in the notebook: {known}.
If the file teaches one of these, use that exact name instead of a synonym."""

SEARCH_AGAIN_PROMPT = ""    # "are these passages enough? yes/no + rephrased query"
FAITHFULNESS_PROMPT = ""    # judge: is the answer supported by these passages?
