"""Shared fixtures: every test writes student databases, the trace, the LLM cache and notebook files to a temp folder,
and any LLM call a test forgot to fake fails instead of reaching the provider."""
import pytest

from agent import s1_llm, s4_trace
from agent.memory import learner
from tutor import notebook
from tutor.retrieval import extract


@pytest.fixture(autouse=True)
def tmp_data(monkeypatch, tmp_path):
    monkeypatch.setattr(learner, "DB_DIR", tmp_path / "students")
    monkeypatch.setattr(s4_trace, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(s1_llm, "CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(s1_llm, "_get_client", lambda provider: pytest.fail("real LLM call: fake s1_llm.chat in this test"))
    monkeypatch.setattr(notebook, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(notebook, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(extract, "IMAGE_DIR", tmp_path / "images")
    return tmp_path
