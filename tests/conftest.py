"""Tests outside agent/ get the same temp folders and real-LLM guard as the agent tests, and the agent eval's
LLM cache and results go to the temp folder too. fake_models puts the notebook on fake models."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.tests.conftest import tmp_data  # noqa: F401  (autouse: importing it applies it here)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import agent_common as ac  # noqa: E402
from test_notebook import FakeCrossEncoder, FakeSentenceTransformer  # noqa: E402
from tutor.retrieval import embed, search  # noqa: E402


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
