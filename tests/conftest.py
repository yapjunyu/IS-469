"""Tests outside agent/ get the same temp folders and real-LLM guard as the agent tests."""
from agent.tests.conftest import tmp_data  # noqa: F401  (autouse: importing it applies it here)
