"""remember_fact and the memory-write check: a fact must cite a passage retrieved in this turn."""
import json

from agent import s1_llm
from agent.memory import learner
from agent.s2_tools import run_tool
from tests.agent.test_loop import tool_ctx, turn
from tests.agent.test_quiz import answer, call, fake_chat, lookup
from tests.agent.test_search import fake_search

FACT = {"concept": "LoRA", "fact": "LoRA adds low-rank matrices.", "passage_id": "p1"}


def remember(ctx, **changes):
    return run_tool("remember_fact", json.dumps({**FACT, **changes}), ctx)


def test_check_on_stores_a_fact_citing_a_passage_searched_this_turn():
    ctx = tool_ctx(search_fn=fake_search)
    run_tool("search_notes", '{"query": "LoRA"}', ctx)
    assert remember(ctx) == {"ok": True, "concept": "LoRA"}
    assert [(f["concept"], f["fact"], f["passage_id"]) for f in learner.read_all("alice")["facts"]] == \
        [("lora", "LoRA adds low-rank matrices.", "p1")]


def test_check_on_refuses_a_passage_not_retrieved_this_turn_and_stores_nothing():
    ctx = tool_ctx(search_fn=fake_search)
    run_tool("search_notes", '{"query": "LoRA"}', ctx)
    assert "error" in remember(ctx, passage_id="p9")
    assert "error" in remember(tool_ctx())  # p1 from an earlier turn's context does not count
    assert learner.read_all("alice")["facts"] == []


def test_check_off_stores_the_same_fact_with_nothing_retrieved():
    ctx = tool_ctx(memory_check=False)
    assert remember(ctx) == {"ok": True, "concept": "LoRA"}
    assert len(learner.read_all("alice")["facts"]) == 1


def test_passage_fetched_by_grading_this_turn_counts_as_retrieved(monkeypatch):
    fake_chat(monkeypatch, '{"correct": true, "hint": ""}')
    ctx = tool_ctx(passages_fn=lookup)
    run_tool("grade_answer", json.dumps({"question": "What does LoRA add?", "student_answer": "low-rank matrices",
                                         "passage_ids": ["p1"]}), ctx)
    assert remember(ctx) == {"ok": True, "concept": "LoRA"}


def test_unknown_concept_refused_and_nothing_stored():
    learner.add_concepts("alice", "notes.pdf", ["LoRA", "KV cache"])
    ctx = tool_ctx(memory_check=False)
    out = remember(ctx, concept="low-rank adaptation")
    assert "kv cache" in out["error"] and "lora" in out["error"]
    assert learner.read_all("alice")["facts"] == []


def test_run_turn_offers_remember_fact_and_passes_the_check_switch(monkeypatch):
    tools_sent = []
    replies = iter([call("remember_fact", FACT), answer("Noted.")] * 2)
    def chat(messages, model, tools=None, **kw):
        tools_sent.append([t["function"]["name"] for t in tools or []])
        return next(replies)
    monkeypatch.setattr(s1_llm, "chat", chat)
    turn("Remember that", "alice", 1)
    assert learner.read_all("alice")["facts"] == []  # check on by default: nothing searched, nothing stored
    turn("Remember that", "alice", 2, memory_check=False)
    assert len(learner.read_all("alice")["facts"]) == 1
    assert set(tools_sent[0]) == {"search_notes", "make_quiz", "grade_answer", "update_mastery", "remember_fact"}
