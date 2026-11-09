import json

import numpy as np
import pytest
from agent import s1_llm
from agent.memory import learner, reflection
from agent.s2_tools import TOOL_SCHEMAS, run_tool
from tests.agent.test_loop import read_trace, tool_ctx, turn

def test_score_moves_with_answers():
    assert learner.update_mastery("alice", "LoRA", True) == 0.65      # 0.5 + 0.3 * (1 - 0.5)
    assert learner.update_mastery("alice", " lora ", False) == 0.455  # same concept, pulled down


def test_students_kept_apart():
    learner.update_mastery("alice", "LoRA", False)
    assert learner.weak_concepts("bob") == []
    assert learner.weak_concepts("alice") == [{"concept": "lora", "score": pytest.approx(0.35)}]


def test_wrong_belief_stored_and_deletable_fact_needs_source():
    learner.update_mastery("alice", "temperature", False, wrong_belief="temperature 0 means random")
    with pytest.raises(ValueError):
        learner.add_fact("alice", "temperature","temperature 0 is greedy", passage_id="")
    belief = learner.read_all("alice")["wrong_beliefs"][0]
    assert belief["belief"] == "temperature 0 means random"
    learner.delete("alice", "wrong_beliefs", belief["id"])
    assert learner.read_all("alice")["wrong_beliefs"] == []


def test_bad_student_id_rejected():
    with pytest.raises(ValueError):
        learner.weak_concepts("../bob")


def test_update_mastery_tool():
    ctx = tool_ctx()
    out = run_tool("update_mastery", '{"concept": "LoRA", "correct": false}', ctx)
    assert out == {"ok": True, "concept": "LoRA", "mastery": 0.35}
    assert "error" in run_tool("update_mastery", '{"concept": "LoRA", "correct": "false"}', ctx)

def test_concepts_merged_after_normalising():
    learner.add_concepts("alice", "notes.pdf", ["LoRA", " lora "])
    assert learner.concepts("alice") == ["lora"]


def test_removing_a_file_keeps_concepts_another_file_has():
    learner.add_concepts("alice", "a.pdf", ["LoRA", "KV cache"])
    learner.add_concepts("alice", "b.pdf", ["lora"])
    learner.remove_file("alice", "a.pdf")
    assert learner.concepts("alice") == ["lora"]
    assert "concepts" not in learner.read_all("alice")  # from the files, not the student: not on the memory page


@pytest.mark.parametrize("tool, args", [
    ("update_mastery", {"correct": True}),
    ("make_quiz", {"n": 2}),
])
def test_concept_tools_accept_only_listed_concepts(monkeypatch, tool, args):
    learner.add_concepts("alice", "notes.pdf", ["LoRA", "KV cache"])
    llm_calls = []
    def chat(messages, model, **kw):
        llm_calls.append(messages)
        return {"message": {"role": "assistant", "content": '[{"question": "Q?", "passage_ids": ["stub-1"]}]'},
                "usage": {"in": 1, "out": 1}, "cached": False}
    monkeypatch.setattr(s1_llm, "chat", chat)
    ctx = tool_ctx()
    bad = run_tool(tool, json.dumps({"concept": "low-rank adaptation", **args}), ctx)
    assert "kv cache" in bad["error"] and "lora" in bad["error"]
    assert llm_calls == []
    assert "error" not in run_tool(tool, json.dumps({"concept": " LoRA", **args}), ctx)


def test_no_topic_parameter_in_tool_schemas():
    assert all("topic" not in s["function"]["parameters"]["properties"] for s in TOOL_SCHEMAS)



def _system_prompt(monkeypatch, message="hello", **switches):
    """The system prompt the tutor's first LLM call gets."""
    sent = []
    def chat(messages, model, **kw):
        sent.append(messages[0]["content"])
        return {"message": {"role": "assistant", "content": "hi"}, "usage": {"in": 1, "out": 1}, "cached": False}
    monkeypatch.setattr(s1_llm, "chat", chat)
    turn(message, "alice", 1, **switches)
    return sent[0]


def _remembered_student():
    learner.add_concepts("alice", "notes.pdf", ["LoRA"])
    learner.update_mastery("alice", "LoRA", False, wrong_belief="LoRA retrains every weight")
    learner.add_session_note("alice", "Struggled with rank in LoRA.")


def test_memory_none_leaves_the_student_out_of_the_system_prompt(monkeypatch):
    _remembered_student()
    system = _system_prompt(monkeypatch, memory="none")
    assert "lora" in system  # the concept list describes the notebook, so it stays
    assert "0.35" not in system and "retrains every weight" not in system and "rank in LoRA" not in system


def test_memory_learner_puts_weak_spots_and_wrong_beliefs_in_the_system_prompt(monkeypatch):
    _remembered_student()
    system = _system_prompt(monkeypatch, memory="learner")
    assert "0.35" in system  # 0.5 + 0.3 * (0 - 0.5)
    assert "LoRA retrains every weight" in system


SESSION = [{"role": "user", "content": "Does LoRA retrain every weight?"},
           {"role": "assistant", "content": "No, only the low-rank adapters (notes.pdf, 3)."}]


def _reflect(monkeypatch, content):
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: {"message": {"role": "assistant", "content": content},
                                                         "usage": {"in": 7, "out": 3}, "cached": False})
    return reflection.end_session("alice", SESSION)


def test_ending_a_session_stores_one_session_note_and_traces_the_call(monkeypatch, tmp_data):
    _reflect(monkeypatch, '{"note": "Unsure which weights LoRA trains.", "weak_spots": []}')
    assert [n["note"] for n in learner.read_all("alice")["session_notes"]] == ["Unsure which weights LoRA trains."]
    rows = read_trace(tmp_data)
    assert [(r["kind"], r["purpose"]) for r in rows] == [("llm", "reflection")]
    assert rows[0]["tokens"] == {"in": 7, "out": 3}


def test_reflection_marks_its_listed_concepts_as_weak_spots(monkeypatch):
    learner.add_concepts("alice", "notes.pdf", ["LoRA", "KV cache"])
    _reflect(monkeypatch, '{"note": "Mixed up LoRA.", "weak_spots": ["LoRA", "not a concept"]}')
    assert learner.weak_concepts("alice") == [{"concept": "lora", "score": pytest.approx(0.35)}]  # 0.5 + 0.3 * (0 - 0.5)


def test_a_bad_reflection_reply_stores_nothing(monkeypatch):
    _reflect(monkeypatch, "Sorry, I can't.")
    assert learner.read_all("alice")["session_notes"] == []


def test_session_notes_on_the_memory_page_and_deletable():
    learner.add_session_note("alice", "Struggled with rank in LoRA.")
    note = learner.read_all("alice")["session_notes"][0]
    assert note["note"] == "Struggled with rank in LoRA."
    learner.delete("alice", "session_notes", note["id"])
    assert learner.read_all("alice")["session_notes"] == []


class FakeEmbedder:
    """Embedder-shaped: one axis per topic word, so a note is similar to a message on the same topic."""
    TOPICS = ("lora", "top-p", "cache")

    def _vec(self, text):
        return np.array([float(t in text.lower()) for t in self.TOPICS])

    def __call__(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def _vector_prompt(monkeypatch, message):
    return _system_prompt(monkeypatch, message, memory="vector", embedder=FakeEmbedder())


def test_memory_vector_puts_the_most_similar_session_notes_in_the_system_prompt(monkeypatch):
    for note in ("Struggled with rank in LoRA.", "Mixed up top-p and temperature.", "Unsure where LoRA adapters go."):
        learner.add_session_note("alice", note)
    system = _vector_prompt(monkeypatch, "How does LoRA work?")
    assert "rank in LoRA" in system and "LoRA adapters go" in system
    assert "top-p" not in system


def test_memory_vector_leaves_weak_spots_and_wrong_beliefs_out(monkeypatch):
    _remembered_student()
    system = _vector_prompt(monkeypatch, "How does LoRA work?")
    assert "rank in LoRA" in system  # the retrieved note
    assert "0.35" not in system and "retrains every weight" not in system


def test_memory_vector_with_no_session_notes_runs_the_turn(monkeypatch):
    system = _vector_prompt(monkeypatch, "How does LoRA work?")
    assert system == _system_prompt(monkeypatch, "How does LoRA work?", memory="none")
