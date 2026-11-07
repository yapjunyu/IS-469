"""Step 16: treat uploaded text as data.

With the guard on, each passage goes to the LLM wrapped in <document id=".." file=".." page="..">
... </document> markers, and the system prompt says marked text is material, never instructions.
"""
import html
import json
import re

TEXT_FIELDS = ("id", "file", "page", "text")  # what an LLM receives of a passage as text


def unmark(text):
    """The text with any <document> or <session> marker in it escaped, so it can't open or close one."""
    return re.sub(r"<(/?(?:document|session))", r"&lt;\1", text, flags=re.I)


def _wrap(p):
    # a passage must not open or close a marker itself, or it could pose as text outside one
    text = unmark(p["text"])
    attrs = " ".join(f'{k}="{html.escape(str(p[k]))}"' for k in ("id", "file", "page"))
    return f"<document {attrs}>\n{text}\n</document>"


def wrap_passages(passages):
    """Return one string with every passage wrapped in <document> markers."""
    return "\n".join(_wrap(p) for p in passages)


def text_fields(passages):
    """The fields of each passage that are sent as text: not a picture's local path."""
    return [{k: p[k] for k in TEXT_FIELDS} for p in passages]


def material(passages, guard):
    """Passages as an LLM sees them: inside markers with the guard on, plain JSON with it off."""
    return wrap_passages(passages) if guard else json.dumps({"passages": text_fields(passages)})
