"""Step 11: the four tools the LLM can call.

- TOOL_SCHEMAS: JSON schemas passed to the LLM (name, description, typed params).
- run_tool(): parse the JSON `arguments` string, validate inputs, dispatch.
  Bad JSON or an unknown tool name returns an error dict, never raises.
- search_notes comes from retrieval (teammate); stub it until ready.
- update_mastery talks to the learner model (teammate); stub it until ready.
"""

import json
from dataclasses import dataclass

# Context to be sent in for every tool call as "ctx"
@dataclass
class TurnContext:
    student_id: str
    turn: int = 0
    search_again: bool = True
    guard: bool = True
    step_cap: int = 6

# Shortcut: stub, swap for retrieval teammate's search(student_id, query) when ready
def search_notes(ctx, query):
    return {
        "chunks":
            [
                {
                    "file": "stub.pdf", "page": 1,
                    "text": f"Stub passage about {query}."
                 }
            ]
    }

# Shortcut: stub, replace with an s1_llm.chat call over retrieved passages
def make_quiz(ctx, topic, n=3):
    n = max(1, min(n, 5))
    return {
        "questions":
            [
                f"Stub question {i + 1} on {topic}?" for i in range(n)
            ]
    }

# Shortcut: stub, replace with an s1_llm.chat call that grades and gives a hint
def grade_answer(ctx, question, student_answer):
    return {
        "correct": False, "hint": "Stub hint: check your notes again."
    }

# Shortcut: stub, replace with a call into memory/learner.py
def update_mastery(ctx, concept, correct):
    return {
        "ok": True, "concept": concept, "correct": correct
    }

# Hashmap iteration so the name of tool linking to the tool itself
TOOLS = {f.__name__: f for f in (search_notes, make_quiz, grade_answer, update_mastery)}

# Wraps tool's detail in the format Groq expects
def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name, "description": description,
            "parameters": {
                "type": "object", "properties": properties, "required": required
            }
        }
    }

# Desc of tools sent to LLM to suggest which tool to use
TOOL_SCHEMAS = [
    _schema("search_notes",
            "Search the student's uploaded notes. Call this before answering any question about course content.",
            {"query": {"type": "string", "description": "What to look for, in your own words."}},
            ["query"]),
    _schema("make_quiz",
            "Make practice questions on a topic from the student's notes.",
            {"topic": {"type": "string"},
             "n": {"type": "integer", "description": "Number of questions, 1 to 5."}},
            ["topic"]),
    _schema("grade_answer",
            "Grade the student's answer to a quiz question. Returns correct or not and a hint, never the answer.",
            {"question": {"type": "string"}, "student_answer": {"type": "string"}},
            ["question", "student_answer"]),
    _schema("update_mastery",
            "Record whether the student got a concept right, after grading or after a clear misconception in chat.",
            {"concept": {"type": "string"}, "correct": {"type": "boolean"}},
            ["concept", "correct"])
]

# Runs tool asked by LLM
def run_tool(name, arguments_json, ctx):
    """Run one tool call. ctx holds per-turn state (student_id, switches, trace info).

    Returns a dict that becomes the content of the `role: tool` message.
    """
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}
    try:
        args = json.loads(arguments_json or "{}")
        return TOOLS[name](ctx, **args)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Bad arguments for {name}: {e}"}