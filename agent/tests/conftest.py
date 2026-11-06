"""Shared fixtures: every agent test writes student databases, the trace and the LLM cache to a temp folder,
and any LLM call a test forgot to fake fails instead of reaching the provider."""
import pytest

from agent import s1_llm, s4_trace
from agent.memory import learner


@pytest.fixture(autouse=True)
def tmp_data(monkeypatch, tmp_path):
    monkeypatch.setattr(learner, "DB_DIR", tmp_path / "students")
    monkeypatch.setattr(s4_trace, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(s1_llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(s1_llm, "_get_client", lambda provider: pytest.fail("real LLM call: fake s1_llm.chat in this test"))
    return tmp_path
