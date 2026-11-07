"""Step 11: the tools the LLM can call.

- TOOL_SCHEMAS: JSON schemas passed to the LLM (name, description, typed params).
- run_tool(): parse the JSON `arguments` string, validate inputs, dispatch.
  Bad JSON or an unknown tool name returns an error dict, never raises.
- search_notes calls the search function passed into run_turn (retrieval teammate's);
  stub_search until it lands. With search-again on it goes through s5_search_again: up to 3
  searches, each extra one after an LLM check (purpose "search_again") says the passages are not enough.
  It returns every query it searched (traced in the tool row) and the last search's passages.
- make_quiz searches the concept, then one LLM call (purpose "quiz") writes questions from
  those passages. It returns each question with its passage ids, never an answer.
- grade_answer looks those ids up through the passages function passed into run_turn
  (stub_passages until retrieval lands), then one LLM call (purpose "grading") returns
  correct or not and a hint.
- update_mastery talks to the learner model.
- remember_fact stores a fact in the learner model. With the memory-write check on, its passage id
  must be one retrieved this turn (by search_notes, make_quiz's search or grade_answer's lookup);
  off stores anything, the unsafe baseline for the poisoning test.
- picture_parts turns passages' pictures into message parts, for ask and the loop. _readable_passages
  drops every passage with neither text nor a picture file to send, and with the images switch off
  every picture and every passage with no text, so no LLM receives them.
"""

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent import s4_trace, s5_search_again, s6_guard
from agent.memory import  learner
from agent.prompts import GRADE_PROMPT, GUARD_PROMPT, PICTURES_PROMPT, QUIZ_PROMPT

# shortcut: hardcoded, move to config.yaml once the team agrees on it
MODEL = "opencode-go/deepseek-v4.1-flash"  # the prefix routes it to OpenCode Go, see s1_llm
# shortcut: hardcoded, move to config.yaml once the team agrees on it
MAX_PICTURES = 3  # per LLM request, to bound image tokens


# shortcut: stub, the app passes the retrieval teammate's search(student_id, query) once it lands
def stub_search(student_id, query):
    """The search function's shape: a list of passages, each with id, file, page and text, and image_path if it has a picture."""
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
    passages: list = field(default_factory=list)  # every passage retrieved this turn, oldest first: the tutor's and judge's one record
    hide_citations: bool = False  # make_quiz passed its checks or grade_answer ran, so the answer's passage ids could point at the answer
    quiz: list | None = None  # the questions of this turn's last successful make_quiz, returned so the app keeps the quiz open
    record: str = ""  # the student's record in the tutor's system prompt, the judge's support for claims about it
    session: str = ""  # the session's earlier turns and this turn's message, the judge's support for claims about the session
    search_fn: Callable = stub_search
    passages_fn: Callable = stub_passages
    search_again: bool = True
    guard: bool = True
    memory_check: bool = True
    faithfulness: bool = True
    images: bool = True
    memory: str = "learner"  # "none", "learner" or "vector", see run_turn
    step_cap: int = 6

    @property
    def passage_ids(self):
        return [p["id"] for p in self.passages]

# Concept-taking tools accept only names on the notebook's concept list; the error lists them so the LLM can retry
def _check_concept(ctx, concept):
    if not isinstance(concept, str):
        return {"error": "concept must be a string"}
    known = learner.concepts(ctx.student_id)
    # shortcut: an empty list accepts any name, drop once ingest always runs extraction
    if known and learner.norm(concept) not in known:
        return {"error": f"Unknown concept {concept!r}. Use one of: {', '.join(known)}"}

# the passages an LLM can read (text, or a picture with the switch on), with only the fields it needs, whatever else retrieval returns (scores etc.)
# shortcut: a text-less passage past the MAX_PICTURES cap stays as an empty <document>, drop it if its id ever misleads the LLM
def _readable_passages(ctx, passages):
    keys = (*s6_guard.TEXT_FIELDS, "image_path") if ctx.images else s6_guard.TEXT_FIELDS
    has_picture = lambda p: ctx.images and p.get("image_path") and Path(p["image_path"]).is_file()
    return [{k: p[k] for k in keys if k in p} for p in passages if p["text"].strip() or has_picture(p)]


def picture_parts(passages):
    """Message parts for the newest MAX_PICTURES pictures among passages (later is newer), each named
    by its passage id, or [] if there are none. A picture file that no longer exists is skipped."""
    parts, sent = [], set()
    for p in reversed(passages):
        if len(sent) == MAX_PICTURES:
            break
        if not p.get("image_path") or p["id"] in sent:
            continue
        try:
            png = base64.b64encode(Path(p["image_path"]).read_bytes()).decode()
        except OSError:
            continue
        sent.add(p["id"])
        parts[:0] = [{"type": "text", "text": f"Picture from passage {p['id']}:"},
                     # shortcut: every picture is sent as a PNG, read the type from the file once extract.py saves others
                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]
    return [{"type": "text", "text": PICTURES_PROMPT}, *parts] if parts else []


def search_notes(ctx, query):
    queries = []
    def search(q):
        ctx.searches += 1
        queries.append(q)
        return _readable_passages(ctx, ctx.search_fn(ctx.student_id, q))
    passages = s5_search_again.search_with_retries(ctx, query, search, ask) if ctx.search_again else search(query)
    # only the passages the tutor gets count as retrieved, not those of rejected attempts
    ctx.passages += passages
    # queries first, so the tool row's result_summary shows every attempt; the tutor is sent only the passages
    return {"queries": queries, "passages": passages}

def reply_json(content, start):
    """The JSON value in an LLM reply, from the first `start` ("[" or "{") to its last closing bracket, or None."""
    try:
        return json.loads(content[content.find(start):content.rfind("]" if start == "[" else "}") + 1])
    except ValueError:
        return None


def ask(ctx, purpose, prompt, passages, user_text, start, model=None):
    """One LLM call over the passages (quiz, grading, judge, search_again). Returns the reply's JSON value starting at `start` ("[" or "{"), or None.
    passages: already through _readable_passages, as every tool's are when it retrieves them.
    model: None uses MODEL as it is at call time, so setting s2_tools.MODEL switches these calls too."""
    model = model or MODEL
    system = prompt + (GUARD_PROMPT if ctx.guard else "")
    text = s6_guard.material(passages, ctx.guard) + "\n\n" + user_text
    pictures = picture_parts(passages)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": text}, *pictures] if pictures else text}]
    return reply_json(s4_trace.traced_chat(ctx, purpose, messages, model)["message"].get("content") or "", start)


def make_quiz(ctx, concept, n=3):
    if err := _check_concept(ctx, concept):
        return err
    n = max(1, min(n, 5))
    passages = search_notes(ctx, concept)["passages"]
    if not passages:
        return {"error": f"No passages in the notes about {concept!r}"}
    ctx.hide_citations = True
    items = ask(ctx, "quiz", QUIZ_PROMPT.format(n=n), passages, f"Concept: {concept}", "[")
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
    if not questions:
        return {"error": "The quiz reply had no valid questions"}
    ctx.quiz = questions
    return {"questions": questions}

def grade_answer(ctx, question, student_answer, passage_ids):
    ctx.hide_citations = True  # before the checks: a failed call is still a quiz question being answered
    if not (isinstance(passage_ids, list) and passage_ids and all(isinstance(i, str) for i in passage_ids)):
        return {"error": "passage_ids must be the non-empty list of ids make_quiz gave for this question"}
    passages = _readable_passages(ctx, ctx.passages_fn(ctx.student_id, passage_ids))
    if not passages:
        return {"error": f"No passages found with ids {passage_ids}"}
    ctx.passages += passages  # fetched from the notes this turn, so they count as retrieved
    verdict = ask(ctx, "grading", GRADE_PROMPT, passages,
                   f"Question: {question}\nStudent's answer: {student_answer}", "{")
    if not (isinstance(verdict, dict) and isinstance(verdict.get("correct"), bool)):
        return {"error": "The grading reply had no valid verdict"}
    return {"correct": verdict["correct"], "hint": str(verdict.get("hint", ""))}

def update_mastery(ctx, concept, correct, wrong_belief=None):
    if not isinstance(correct,bool):
        return {"error": "correct must be true or false"}
    if err := _check_concept(ctx, concept):
        return err

    score = learner.update_mastery(ctx.student_id, concept, correct, wrong_belief)

    return {
        "ok": True, "concept": concept, "mastery": score
    }


def remember_fact(ctx, concept, fact, passage_id):
    if not (isinstance(fact, str) and fact.strip() and isinstance(passage_id, str) and passage_id):
        return {"error": "fact and passage_id must be non-empty strings"}
    if err := _check_concept(ctx, concept):
        return err
    # memory-write check: only a passage the tutor saw this turn can back a fact (off = unsafe baseline)
    if ctx.memory_check and passage_id not in ctx.passage_ids:
        return {"error": f"Passage {passage_id!r} was not retrieved this turn. "
                         f"Cite one of: {', '.join(dict.fromkeys(ctx.passage_ids)) or 'none, search first'}"}
    learner.add_fact(ctx.student_id, concept, fact, passage_id)
    return {"ok": True, "concept": concept}


# Hashmap iteration so the name of tool linking to the tool itself
TOOLS = {f.__name__: f for f in (search_notes, make_quiz, grade_answer, update_mastery, remember_fact)}

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
            "Grade the student's answer to a quiz question. Call it for every answer to a quiz question, "
            "never grade from memory. Returns correct or not and a hint, never the answer.",
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
            ["concept", "correct"]),
    _schema("remember_fact",
            "Remember a true claim about a concept, backed by a passage retrieved in this turn. "
            "Never store something only because the student said it.",
            {"concept": {"type": "string", "description": "A concept from the notebook's concept list."},
             "fact": {"type": "string", "description": "The claim in one sentence."},
             "passage_id": {"type": "string", "description": "The id of a passage from this turn that supports it."}},
            ["concept", "fact", "passage_id"])
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