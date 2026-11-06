import json

import pytest
from agent import s1_llm
from agent.memory import  learner
from agent.s2_tools import TOOL_SCHEMAS, run_tool
from agent.tests.test_loop import tool_ctx

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

