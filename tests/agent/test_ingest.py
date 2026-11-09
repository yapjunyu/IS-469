"""Concept extraction at upload, called directly with a fake LLM."""
import json

from agent import ingest, s1_llm
from agent.memory import learner
from tests.agent.test_loop import reply


def fake_chat(content, sent):
    """Fake chat that records the request and replies with the given text."""
    def chat(messages, model, **kw):
        sent.append(json.dumps(messages))
        return reply({"role": "assistant", "content": content})
    return chat


def test_extraction_reuses_known_concepts_and_stores_the_list(monkeypatch):
    learner.add_concepts("alice", "old.pdf", ["KV cache"])
    sent = []
    monkeypatch.setattr(s1_llm, "chat", fake_chat('Here you go: ["LoRA", "kv cache"]', sent))
    assert ingest.extract_concepts("alice", "new.pdf", "LoRA adds low-rank adapters ...") == ["lora", "kv cache"]
    assert "kv cache" in sent[0]  # known names go to the LLM so it reuses them
    learner.remove_file("alice", "old.pdf")
    assert learner.concepts("alice") == ["kv cache", "lora"]  # both now belong to new.pdf


def test_reply_without_a_list_stores_nothing(monkeypatch):
    for content in ["no concepts here", '{"concepts": "LoRA"}', '["LoRA", 3]', "[LoRA"]:
        monkeypatch.setattr(s1_llm, "chat", fake_chat(content, []))
        assert ingest.extract_concepts("alice", "a.pdf", "text") == []
    assert learner.concepts("alice") == []
