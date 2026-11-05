"""Tests with a fake LLM and fake search_notes: step cap, tool dispatch, trace lines."""
from agent import s1_llm, s3_loop
import json

SEARCH_CALL = {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
               "function": {"name": "search_notes", "arguments": '{"query": "LoRA"}'}}]}


def test_tool_then_answer(monkeypatch):
    replies = iter([SEARCH_CALL, {"role": "assistant", "content": "LoRA is ..."}])
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: {"message": next(replies)})
    out = s3_loop.run_turn("What is LoRA?", "alice", 1)
    assert out == {"answer": "LoRA is ...", "citations": [{"file": "stub.pdf", "page": 1}], "steps": 2}


def always_search(messages, model, tools=None, tool_choice="auto", **kw):
    """Fake chat that keeps calling search_notes until tools are switched off."""
    if tool_choice == "none":
        return {"message": {"role": "assistant", "content": "forced"}}
    return {"message": SEARCH_CALL}


def test_step_cap_forces_answer(monkeypatch):
    monkeypatch.setattr(s1_llm, "chat", always_search)
    out = s3_loop.run_turn("loop forever", "alice", 1, step_cap=2)
    assert out["answer"] == "forced" and out["steps"] == 3


def test_trace_line_per_tool_call(monkeypatch, tmp_data):
    replies = iter([SEARCH_CALL, {"role": "assistant", "content": "done"}])
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: {"message": next(replies)})
    s3_loop.run_turn("What is LoRA?", "alice", 1)
    lines = [json.loads(l) for l in (tmp_data / "trace.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["tool"] == "search_notes" and lines[0]["ok"]


def test_every_trace_row_carries_turn_number(monkeypatch, tmp_data):
    monkeypatch.setattr(s1_llm, "chat", always_search)
    s3_loop.run_turn("loop", "alice", 4, step_cap=2)
    lines = [json.loads(l) for l in (tmp_data / "trace.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(l["turn"] == 4 for l in lines)
