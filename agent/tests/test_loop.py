"""Tests with a fake LLM and fake search_notes: step cap, tool dispatch, trace rows."""
from agent import s1_llm, s3_loop
from agent.memory import learner
import json

SEARCH_CALL = {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
               "function": {"name": "search_notes", "arguments": '{"query": "LoRA"}'}}]}


def reply(message, tokens_in=10, tokens_out=5):
    """The plain-dict shape s1_llm.chat returns."""
    return {"message": message, "usage": {"in": tokens_in, "out": tokens_out}, "cached": False}


def scripted(*replies):
    """Fake chat that returns the given replies in order."""
    it = iter(replies)
    return lambda *a, **k: next(it)


def always_search(messages, model, tools=None, tool_choice="auto", **kw):
    """Fake chat that keeps calling search_notes until tools are switched off."""
    if tool_choice == "none":
        return reply({"role": "assistant", "content": "forced"})
    return reply(SEARCH_CALL)


# switches that add LLM calls a scripted fake would not answer; add each new one here
EXTRA_CALLS_OFF = {"faithfulness": False, "search_again": False}


def turn(*args, **switches):
    """run_turn with the extra-call checks off unless the test switches one on."""
    return s3_loop.run_turn(*args, **{**EXTRA_CALLS_OFF, **switches})


def read_trace(tmp_data):
    """Every row the turn wrote to the temp trace file, in order."""
    return [json.loads(line) for line in (tmp_data / "trace.jsonl").read_text().splitlines()]


def test_tool_then_answer(monkeypatch):
    monkeypatch.setattr(s1_llm, "chat", scripted(reply(SEARCH_CALL),
                                                  reply({"role": "assistant", "content": "LoRA is ..."})))
    out = turn("What is LoRA?", "alice", 1)
    assert out == {"answer": "LoRA is ...", "citations": [{"file": "stub.pdf", "page": 1}], "steps": 2,
                   "faithfulness": None, "warning": False}


def test_step_cap_forces_answer(monkeypatch):
    monkeypatch.setattr(s1_llm, "chat", always_search)
    out = turn("loop forever", "alice", 1, step_cap=2)
    assert out["answer"] == "forced" and out["steps"] == 3


def test_search_then_answer_trace_rows(monkeypatch, tmp_data):
    monkeypatch.setattr(s1_llm, "chat", scripted(reply(SEARCH_CALL),
                                                  reply({"role": "assistant", "content": "done"})))
    turn("What is LoRA?", "alice", 1)
    rows = read_trace(tmp_data)
    assert [r["kind"] for r in rows] == ["llm", "tool", "llm", "final"]
    assert all(list(r)[:5] == ["kind", "ts", "student", "turn", "step"] for r in rows)
    assert rows[1]["tool"] == "search_notes" and rows[1]["inputs"] == {"query": "LoRA"} and rows[1]["ok"]
    assert rows[-1]["duration_ms"] >= 0  # turn latency for step 22


def test_every_trace_row_carries_turn_number(monkeypatch, tmp_data):
    monkeypatch.setattr(s1_llm, "chat", always_search)
    turn("loop", "alice", 4, step_cap=2)
    assert all(r["turn"] == 4 for r in read_trace(tmp_data))


def test_tokens_only_on_llm_rows_and_summed_on_final(monkeypatch, tmp_data):
    monkeypatch.setattr(s1_llm, "chat", scripted(reply(SEARCH_CALL, 100, 20),
                                                  reply({"role": "assistant", "content": "done"}, 300, 40)))
    turn("What is LoRA?", "alice", 1)
    rows = read_trace(tmp_data)
    assert "tokens" not in rows[1]
    assert rows[-1]["tokens"] == {"in": 400, "out": 60}


def test_cap_hit_final_row_after_forced_answer(monkeypatch, tmp_data):
    monkeypatch.setattr(s1_llm, "chat", always_search)
    turn("loop", "alice", 1, step_cap=2)
    *_, forced, final = read_trace(tmp_data)
    assert forced["kind"] == "llm" and forced["purpose"] == "forced_answer"
    assert final["kind"] == "final" and final["cap_hit"] and final["steps"] == 3 and final["searches"] == 2


def test_bad_tool_json_traced_raw_and_turn_completes(monkeypatch, tmp_data):
    bad = {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
           "function": {"name": "search_notes", "arguments": '{"query": LoRA'}}]}
    monkeypatch.setattr(s1_llm, "chat", scripted(reply(bad), reply({"role": "assistant", "content": "ok"})))
    out = turn("What is LoRA?", "alice", 1)
    tool_row = read_trace(tmp_data)[1]
    assert out["answer"] == "ok"
    assert tool_row["inputs"] == '{"query": LoRA' and not tool_row["ok"] and tool_row["error"]


def test_system_prompt_lists_the_notebooks_concepts(monkeypatch):
    learner.add_concepts("alice", "notes.pdf", ["LoRA", "KV cache"])
    sent = []
    def chat(messages, model, **kw):
        sent.append(messages[0]["content"])
        return reply({"role": "assistant", "content": "hi"})
    monkeypatch.setattr(s1_llm, "chat", chat)
    turn("hello", "alice", 1)
    assert "lora" in sent[0] and "kv cache" in sent[0]
