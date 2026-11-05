import pytest
from agent.memory import  learner
from agent.s2_tools import TurnContext, run_tool

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
        learner.add_fact("alice", "temperature","temperature 0 is greedy", chunk_id="")
    belief = learner.read_all("alice")["wrong_beliefs"][0]
    assert belief["belief"] == "temperature 0 means random"
    learner.delete("alice", "wrong_beliefs", belief["id"])
    assert learner.read_all("alice")["wrong_beliefs"] == []


def test_bad_student_id_rejected():
    with pytest.raises(ValueError):
        learner.weak_concepts("../bob")


def test_update_mastery_tool():
    ctx = TurnContext(student_id="alice")
    out = run_tool("update_mastery", '{"concept": "LoRA", "correct": false}', ctx)
    assert out == {"ok": True, "concept": "LoRA", "mastery": 0.35}
    assert "error" in run_tool("update_mastery", '{"concept": "LoRA", "correct": "false"}', ctx)