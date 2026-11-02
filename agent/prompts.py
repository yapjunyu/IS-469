"""All prompt text in one place."""

SYSTEM_PROMPT = """You are a study tutor. Answer only from the student's notes.
- Call search_notes before answering any question about course content.
- Cite every claim as (file, page). If the notes do not cover it, say so.
- In quiz mode, give hints, never the answer."""

SEARCH_AGAIN_PROMPT = ""    # "are these passages enough? yes/no + rephrased query"
FAITHFULNESS_PROMPT = ""    # judge: is the answer supported by these passages?
