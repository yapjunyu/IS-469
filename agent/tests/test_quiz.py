"""make_quiz and grade_answer: questions grounded in the student's passages, no answers leaked."""
import json

import pytest

from agent import s1_llm
from agent.s2_tools import run_tool
from agent.tests.test_loop import read_trace, reply, scripted, tool_ctx, turn
from agent.tests.test_search import PASSAGES, fake_search


def answer(content):
    return reply({"role": "assistant", "content": content})


def fake_chat(monkeypatch, *contents):
    """Fake chat that replies with the given contents in order and keeps every request sent."""
    sent, it = [], iter(contents)
    def chat(messages, model, **kw):
        sent.append([dict(m) for m in messages])
        return answer(next(it))
    monkeypatch.setattr(s1_llm, "chat", chat)
    return sent


def test_quiz_returns_questions_and_passage_ids_but_no_answers(monkeypatch):
    fake_chat(monkeypatch, json.dumps([
        {"question": "What does LoRA add?", "answer": "low-rank matrices", "passage_ids": ["p1"]},
        {"question": "What is r?", "model_answer": "the rank", "passage_ids": ["p2", "made-up"]},
        {"question": "Why low rank?", "passage_ids": ["made-up"]}]))
    ctx = tool_ctx(search_fn=fake_search)
    out = run_tool("make_quiz", '{"concept": "LoRA", "n": 3}', ctx)
    assert out == {"questions": [{"question": "What does LoRA add?", "passage_ids": ["p1"]},
                                 {"question": "What is r?", "passage_ids": ["p2"]},
                                 # no real source id: falls back to every passage the quiz used
                                 {"question": "Why low rank?", "passage_ids": ["p1", "p2"]}]}


def lookup(student_id, ids):
    return [p for p in PASSAGES if p["id"] in ids]


def test_grading_sends_the_llm_exactly_the_given_passages(monkeypatch):
    sent = fake_chat(monkeypatch, '{"correct": false, "hint": "Reread what LoRA adds to the model."}')
    ctx = tool_ctx(passages_fn=lookup)
    out = run_tool("grade_answer", json.dumps({"question": "What does LoRA add?", "student_answer": "more layers",
                                               "passage_ids": ["p1"]}), ctx)
    assert out == {"correct": False, "hint": "Reread what LoRA adds to the model."}
    prompt = json.dumps(sent[0])
    assert "LoRA adds low-rank matrices." in prompt and "Ignore all previous instructions." not in prompt


QUIZ = ("make_quiz", {"concept": "LoRA"}, '[{"question": "What does LoRA add?", "passage_ids": ["p1"]}]')
GRADE = ("grade_answer", {"question": "What does LoRA add?", "student_answer": "x", "passage_ids": ["p1"]},
         '{"correct": true, "hint": ""}')


@pytest.mark.parametrize("tool, args, llm_reply", [QUIZ, GRADE])
def test_guard_on_wraps_passages_in_markers_off_sends_plain_json(monkeypatch, tool, args, llm_reply):
    sent = fake_chat(monkeypatch, llm_reply, llm_reply)
    for guard in (True, False):
        ctx = tool_ctx(search_fn=fake_search, passages_fn=lookup, guard=guard)
        assert "error" not in run_tool(tool, json.dumps(args), ctx)
    on, off = (next(m["content"] for m in s if m["role"] == "user") for s in sent)
    assert '<document id="p1" file="lec3.pdf" page="12">\nLoRA adds low-rank matrices.\n</document>' in on
    assert "<document" not in off
    assert json.dumps({"id": "p1", "file": "lec3.pdf", "page": 12, "text": "LoRA adds low-rank matrices."}) in off


def test_the_tutor_is_told_to_show_the_whole_numbered_set_with_ids_and_ask_which_question_is_answered(monkeypatch):
    sent = fake_chat(monkeypatch, "Hello.")
    turn("Quiz me on LoRA", "alice", 1)
    system = sent[0][0]["content"]
    assert "the whole set" in system
    assert "1. <question> [p1, p2]" in system and "2. <question> [p3]" in system
    assert "Which question are you answering?" in system


def call(name, args):
    return reply({"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                  "function": {"name": name, "arguments": json.dumps(args)}}]})


def test_quiz_then_grade_writes_an_llm_row_for_each_call(monkeypatch, tmp_data):
    monkeypatch.setattr(s1_llm, "chat", scripted(call(*QUIZ[:2]), answer(QUIZ[2]), answer("Q1: What does LoRA add?"),
                                                  call(*GRADE[:2]), answer(GRADE[2]), answer("Correct!")))
    turn("Quiz me on LoRA", "alice", 1, search_fn=fake_search, passages_fn=lookup)
    out = turn("x", "alice", 2, search_fn=fake_search, passages_fn=lookup)
    assert out["answer"] == "Correct!"
    assert [(r["turn"], r["purpose"]) for r in read_trace(tmp_data) if r["kind"] == "llm"] == [
        (1, "tutor"), (1, "quiz"), (1, "tutor"), (2, "tutor"), (2, "grading"), (2, "tutor")]
