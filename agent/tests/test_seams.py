"""Every LLM call goes through s4_trace.traced_chat, so each one writes an llm row and counts toward the turn."""
from pathlib import Path

SOURCES = [p for p in Path(__file__).resolve().parents[1].rglob("*.py") if "tests" not in p.parts]


def test_no_module_holds_its_own_copy_of_chat():
    assert [p.name for p in SOURCES if "from agent.s1_llm import" in p.read_text(encoding="utf-8")] == []


def test_only_the_trace_module_calls_s1_llm_chat():
    assert [p.name for p in SOURCES
            if p.name != "s4_trace.py" and "s1_llm.chat(" in p.read_text(encoding="utf-8")] == []


def test_agent_code_says_passage_not_chunk():
    # GLOSSARY.md: the term is Passage; Chunk is the retrieval package's own pipeline stage
    hits = [f"{p.name}:{n}" for p in SOURCES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if "chunk" in line.lower() and "RENAME COLUMN chunk_id" not in line]
    assert hits == []
