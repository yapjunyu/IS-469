"""Shared fixtures: every agent test writes student databases, the trace and the LLM cache to a temp folder."""
import pytest

from agent import s1_llm, s4_trace
from agent.memory import learner


@pytest.fixture(autouse=True)
def tmp_data(monkeypatch, tmp_path):
    monkeypatch.setattr(learner, "DB_DIR", tmp_path / "students")
    monkeypatch.setattr(s4_trace, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(s1_llm, "CACHE_DIR", tmp_path / "llm_cache")
    return tmp_path
