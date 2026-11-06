"""Step 11: the four tools the LLM can call.

- TOOL_SCHEMAS: JSON schemas passed to the LLM (name, description, typed params).
- run_tool(): parse the JSON `arguments` string, validate inputs, dispatch.
  Bad JSON or an unknown tool name returns an error dict, never raises.
- search_notes calls the search function passed into run_turn (retrieval teammate's);
  stub_search until it lands.
- make_quiz searches the concept, then one LLM call (purpose "quiz") writes questions from
  those passages. It returns each question with its passage ids, never an answer.
- grade_answer looks those ids up through the passages function passed into run_turn
  (stub_passages until retrieval lands), then one LLM call (purpose "grading") returns
  correct or not and a hint.
- update_mastery talks to the learner model.
"""

import json
from dataclasses import dataclass, field
from typing import Callable

from agent import s4_trace, s6_guard
from agent.memory import  learner
from agent.prompts import GRADE_PROMPT, GUARD_PROMPT, QUIZ_PROMPT

# shortcut: hardcoded, move to config.yaml once the team agrees on it
MODEL = "llama-3.3-70b-versatile"


# shortcut: stub, the app passes the retrieval teammate's search(student_id, query) once it lands
def stub_search(student_id, query):
    """The search function's shape: a list of passages, each with id, file, page and text."""
    return [{"id": "stub-1", "file": "stub.pdf", "page": 1, "text": f"Stub passage about {query}."}]


# shortcut: stub, the app passes the retrieval teammate's get_passages(student_id, ids) once it lands
def stub_passages(student_id, ids):
    """The lookup function's shape: the passages with these ids, same fields as a search result."""
    return [{"id": i, "file": "stub.pdf", "page": 1, "text": f"Stub passage {i}."} for i in ids]

# Context to be sent in for every tool call as "ctx"
@dataclass
class TurnContext:
    student_id: str
    turn: int = 0
    step: int = 0
    # running totals for the turn, so LLM calls made inside tools count too
    tokens: dict = field(default_factory=lambda: {"in": 0, "out": 0})
    searches: int = 0
    passage_ids: list = field(default_factory=list)  # every passage retrieved this turn
    search_fn: Callable = stub_search
    passages_fn: Callable = stub_passages
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

# keep only the fields an LLM needs, whatever else retrieval returns (scores etc.)
def _fields(passages):
    return [{k: p[k] for k in ("id", "file", "page", "text")} for p in passages]


def search_notes(ctx, query):
    ctx.searches += 1
    passages = _fields(ctx.search_fn(ctx.student_id, query))
    ctx.passage_ids += [p["id"] for p in passages]
    return {"passages": passages}

def _ask(ctx, purpose, prompt, passages, user_text, start):
    """One LLM call over the passages. Returns the reply's JSON value starting at `start` ("[" or "{"), or None."""
    system = prompt + (GUARD_PROMPT if ctx.guard else "")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": s6_guard.material(passages, ctx.guard) + "\n\n" + user_text}]
    content = s4_trace.traced_chat(ctx, purpose, messages, MODEL)["message"].get("content") or ""
    try:
        return json.loads(content[content.find(start):content.rfind("]" if start == "[" else "}") + 1])
    except ValueError:
        return None


def make_quiz(ctx, concept, n=3):
    if err := _check_concept(ctx, concept):
        return err
    n = max(1, min(n, 5))
    passages = search_notes(ctx, concept)["passages"]
    if not passages:
        return {"error": f"No passages in the notes about {concept!r}"}
    items = _ask(ctx, "quiz", QUIZ_PROMPT.format(n=n), passages, f"Concept: {concept}", "[")
    ids = [p["id"] for p in passages]
    questions = []
    # keep only the question and real source ids: any answer the LLM wrote must not reach the tutor
    for q in items or []:
        if isinstance(q, dict) and isinstance(q.get("question"), str):
            got = q.get("passage_ids")
            got = [i for i in got if i in ids] if isinstance(got, list) else []
            # no valid source id: grade against every passage the quiz was made from
            questions.append({"question": q["question"], "passage_ids": got or ids})
    questions = questions[:n]
    return {"questions": questions} if questions else {"error": "The quiz reply had no valid questions"}

def grade_answer(ctx, question, student_answer, passage_ids):
    if not (isinstance(passage_ids, list) and passage_ids and all(isinstance(i, str) for i in passage_ids)):
        return {"error": "passage_ids must be the non-empty list of ids make_quiz gave for this question"}
    passages = _fields(ctx.passages_fn(ctx.student_id, passage_ids))
    if not passages:
        return {"error": f"No passages found with ids {passage_ids}"}
    verdict = _ask(ctx, "grading", GRADE_PROMPT, passages,
                   f"Question: {question}\nStudent's answer: {student_answer}", "{")
    if not (isinstance(verdict, dict) and isinstance(verdict.get("correct"), bool)):
        return {"error": "The grading reply had no valid verdict"}
    return {"correct": verdict["correct"], "hint": str(verdict.get("hint", ""))}

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
            "Make practice questions on a concept from the student's notes. Returns questions without answers, "
            "each with the passage ids to pass to grade_answer.",
            {"concept": {"type": "string", "description": "A concept from the notebook's concept list."},
             "n": {"type": "integer", "description": "Number of questions, 1 to 5."}},
            ["concept"]),
    _schema("grade_answer",
            "Grade the student's answer to a quiz question. Returns correct or not and a hint, never the answer.",
            {"question": {"type": "string"}, "student_answer": {"type": "string"},
             "passage_ids": {"type": "array", "items": {"type": "string"},
                             "description": "The passage ids make_quiz gave for this question."}},
            ["question", "student_answer", "passage_ids"]),
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