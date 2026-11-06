"""Search-again: after each search one LLM call says whether the passages are enough, else rephrases, up to 3 searches."""
import json

from agent import s1_llm
from agent.tests.test_loop import SEARCH_CALL, read_trace, reply, turn

PASSAGES = [{"id": "p1", "file": "lec3.pdf", "page": 12, "text": "LoRA adds low-rank matrices."}]


def run(monkeypatch, *checks, **switches):
    """One search-then-answer turn; the check gives the (enough, query) replies in order. Returns the queries searched."""
    tutor = iter([reply(SEARCH_CALL), reply({"role": "assistant", "content": "done"})])
    checks = iter(checks)
    def chat(messages, model, tools=None, **kw):
        if tools:  # only the tutor's calls list tools
            return next(tutor)
        enough, query = next(checks)
        return reply({"role": "assistant", "content": json.dumps({"enough": enough, "query": query})}, 30, 6)
    monkeypatch.setattr(s1_llm, "chat", chat)
    queries = []
    def search(student_id, query):
        queries.append(query)
        return PASSAGES
    turn("What is LoRA?", "alice", 1, search_fn=search, **switches)
    return queries


def check_rows(tmp_data):
    return [r for r in read_trace(tmp_data) if r["kind"] == "llm" and r["purpose"] == "search_again"]


def test_not_enough_twice_searches_three_times_with_the_rephrased_queries(monkeypatch, tmp_data):
    queries = run(monkeypatch, (False, "LoRA low-rank adapters"), (False, "LoRA fine-tuning"), search_again=True)
    assert queries == ["LoRA", "LoRA low-rank adapters", "LoRA fine-tuning"]
    rows = check_rows(tmp_data)
    assert len(rows) == 2 and all(r["tokens"] == {"in": 30, "out": 6} for r in rows)  # no check after the 3rd search
    assert read_trace(tmp_data)[-1]["searches"] == 3
    tool_row = next(r for r in read_trace(tmp_data) if r["kind"] == "tool")
    assert '"LoRA low-rank adapters", "LoRA fine-tuning"' in tool_row["result_summary"]  # each attempt's query traced


def test_enough_on_the_first_check_searches_once(monkeypatch, tmp_data):
    queries = run(monkeypatch, (True, ""), search_again=True)
    assert queries == ["LoRA"] and len(check_rows(tmp_data)) == 1
    assert read_trace(tmp_data)[-1]["searches"] == 1


def test_switch_off_searches_once_with_no_check_call(monkeypatch, tmp_data):
    queries = run(monkeypatch, search_again=False)  # no check replies scripted: a check call would raise
    assert queries == ["LoRA"] and check_rows(tmp_data) == []
    assert read_trace(tmp_data)[-1]["searches"] == 1


def test_unreadable_check_keeps_the_first_result(monkeypatch, tmp_data):
    queries = run(monkeypatch, (None, None), search_again=True)  # {"enough": null}: not a clear "not enough"
    assert queries == ["LoRA"] and len(check_rows(tmp_data)) == 1
    assert read_trace(tmp_data)[-1]["searches"] == 1
