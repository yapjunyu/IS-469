"""Shared fixtures: every test writes student databases, the trace, the LLM cache and notebook files to a temp folder,
and any LLM call a test forgot to fake fails instead of reaching the provider. The agent eval's LLM cache and results
go to the temp folder too. fake_models puts the notebook on fake models."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import s1_llm, s4_trace
from agent.memory import learner
from tutor import notebook
from tutor.retrieval import extract

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import agent_common as ac  # noqa: E402
from tests.test_notebook import FakeCrossEncoder, FakeSentenceTransformer  # noqa: E402
from tutor.retrieval import embed, search  # noqa: E402


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


@pytest.fixture(autouse=True)
def eval_out(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "CACHE", tmp_path / "llm_cache_eval")
    monkeypatch.setattr(ac, "OUT", tmp_path / "results_agent")


@pytest.fixture
def fake_models(monkeypatch):
    """Replaces the model library itself, so the real Embedder, HybridRerankIndex and their caches all run."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer, CrossEncoder=FakeCrossEncoder))
    monkeypatch.setattr(FakeSentenceTransformer, "loads", 0)
    monkeypatch.setattr(FakeCrossEncoder, "loads", 0)
    embed._load.cache_clear()
    search._cross_encoder.cache_clear()
    yield
    embed._load.cache_clear()  # so no fake model outlives this test
    search._cross_encoder.cache_clear()
