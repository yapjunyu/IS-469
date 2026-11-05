"""Tests with a fake LLM and fake search_notes: step cap, tool dispatch, trace lines."""
from agent import s3_loop
import json
import pytest

SEARCH_CALL = {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
               "function": {"name": "search_notes", "arguments": '{"query": "LoRA"}'}}]}


def test_tool_then_answer(monkeypatch):
    replies = iter([SEARCH_CALL, {"role": "assistant", "content": "LoRA is ..."}])
    monkeypatch.setattr(s3_loop, "chat", lambda *a, **k: {"message": next(replies)})
    out = s3_loop.run_turn("What is LoRA?", "alice")
    assert out == {"answer": "LoRA is ...", "citations": [{"file": "stub.pdf", "page": 1}], "steps": 2}


def test_step_cap_forces_answer(monkeypatch):
    def fake_chat(messages, model, tools=None, tool_choice="auto", **kw):
        if tool_choice == "none":
            return {"message": {"role": "assistant", "content": "forced"}}
        return {"message": SEARCH_CALL}
    monkeypatch.setattr(s3_loop, "chat", fake_chat)
    out = s3_loop.run_turn("loop forever", "alice", step_cap=2)
    assert out["answer"] == "forced" and out["steps"] == 3

@pytest.fixture(autouse=True)
def tmp_trace(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(s3_loop, "TRACE_PATH", path)
    return path

def test_trace_line_per_tool_call(monkeypatch, tmp_trace):
    replies = iter([SEARCH_CALL, {"role": "assistant", "content": "done"}])
    monkeypatch.setattr(s3_loop, "chat", lambda *a, **k: {"message": next(replies)})
    s3_loop.run_turn("What is LoRA?", "alice")
    lines = [json.loads(l) for l in tmp_trace.read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["tool"] == "search_notes" and lines[0]["ok"]