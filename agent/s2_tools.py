"""Step 11: the four tools the LLM can call.

- TOOL_SCHEMAS: JSON schemas passed to the LLM (name, description, typed params).
- run_tool(): parse the JSON `arguments` string, validate inputs, dispatch.
  Bad JSON or an unknown tool name returns an error dict, never raises.
- search_notes comes from retrieval (teammate); stub it until ready.
- update_mastery talks to the learner model (teammate); stub it until ready.
"""

import json
from dataclasses import dataclass, field
from agent.memory import  learner

# Context to be sent in for every tool call as "ctx"
@dataclass
class TurnContext:
    student_id: str
    turn: int = 0
    step: int = 0
    # running totals for the turn, so LLM calls made inside tools count too
    tokens: dict = field(default_factory=lambda: {"in": 0, "out": 0})
    searches: int = 0
    search_again: bool = True
    guard: bool = True
    step_cap: int = 6

# Concept-taking tools accept only names on the notebook's concept list; the error lists them so the LLM can retry
def _check_concept(ctx, concept):
    if not isinstance(concept, str):
        return {"error": "concept must be a string"}
    known = learner.concepts(ctx.student_id)
    # shortcut: an empty list accepts any name, drop once ingest always runs extraction
    if known and learner.norm(concept) not in known:
        return {"error": f"Unknown concept {concept!r}. Use one of: {', '.join(known)}"}

# Shortcut: stub, swap for retrieval teammate's search(student_id, query) when ready
def search_notes(ctx, query):
    ctx.searches += 1
    return {
        "chunks":
            [
                {
                    "file": "stub.pdf", "page": 1,
                    "text": f"Stub passage about {query}."
                 }
            ]
    }

# Shortcut: stub, replace with an s4_trace.traced_chat call over retrieved passages
def make_quiz(ctx, concept, n=3):
    if err := _check_concept(ctx, concept):
        return err
    n = max(1, min(n, 5))
    return {
        "questions":
            [
                f"Stub question {i + 1} on {concept}?" for i in range(n)
            ]
    }

# Shortcut: stub, replace with an s4_trace.traced_chat call that grades and gives a hint
def grade_answer(ctx, question, student_answer):
    return {
        "correct": False, "hint": "Stub hint: check your notes again."
    }

# Shortcut: stub, replace with a call into memory/learner.py
def update_mastery(ctx, concept, correct, wrong_belief=None):
    if not isinstance(correct,bool):
        return {"error": "correct must be true or false"}
    if err := _check_concept(ctx, concept):
        return err

    score = learner.update_mastery(ctx.student_id, concept, correct, wrong_belief)

    return {
        "ok": True, "concept": concept, "mastery": score
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
            "Make practice questions on a concept from the student's notes.",
            {"concept": {"type": "string", "description": "A concept from the notebook's concept list."},
             "n": {"type": "integer", "description": "Number of questions, 1 to 5."}},
            ["concept"]),
    _schema("grade_answer",
            "Grade the student's answer to a quiz question. Returns correct or not and a hint, never the answer.",
            {"question": {"type": "string"}, "student_answer": {"type": "string"}},
            ["question", "student_answer"]),
    _schema("update_mastery",
            "Record whether the student got a concept right, after grading or after a clear misconception in chat.",
            {
                "concept": {"type": "string", "description": "A concept from the notebook's concept list."},
                "correct": {"type": "boolean"},
                "wrong_belief": {"type": "string", "description":"The student's misconception in one sentence, only if they showed one."}
            },
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