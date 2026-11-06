"""Step 16: treat uploaded text as data.

With the guard on, each passage goes to the LLM wrapped in <document id=".." file=".." page="..">
... </document> markers, and the system prompt says marked text is material, never instructions.
"""
import html
import json
import re


def _wrap(p):
    # a passage must not open or close a marker itself, or it could pose as text outside one
    text = re.sub(r"<(/?document)", r"&lt;\1", p["text"], flags=re.I)
    attrs = " ".join(f'{k}="{html.escape(str(p[k]))}"' for k in ("id", "file", "page"))
    return f"<document {attrs}>\n{text}\n</document>"


def wrap_passages(passages):
    """Return one string with every passage wrapped in <document> markers."""
    return "\n".join(_wrap(p) for p in passages)


def material(passages, guard):
    """Passages as an LLM sees them: inside markers with the guard on, plain JSON with it off."""
    return wrap_passages(passages) if guard else json.dumps({"passages": passages})
