"""Tests outside agent/ get the same temp folders and real-LLM guard as the agent tests, and the agent eval's
LLM cache and results go to the temp folder too."""
import sys
from pathlib import Path

import pytest

from agent.tests.conftest import tmp_data  # noqa: F401  (autouse: importing it applies it here)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import agent_common as ac  # noqa: E402


@pytest.fixture(autouse=True)
def eval_out(monkeypatch, tmp_path):
    monkeypatch.setattr(ac, "CACHE", tmp_path / "llm_cache_eval")
    monkeypatch.setattr(ac, "OUT", tmp_path / "results_agent")
