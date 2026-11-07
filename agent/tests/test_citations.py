"""run_turn's citations: the passages retrieved this turn that the final answer names, by ID."""
from pathlib import Path

from agent import s1_llm
from agent.tests.test_loop import SEARCH_CALL, reply, turn

PASSAGES = [{"id": "p1", "file": "lec3.pdf", "page": 12, "text": "LoRA adds low-rank matrices."},
            {"id": "p2", "file": "lec4.pdf", "page": 2, "text": "BM25 ranks by keyword."},
            {"id": "p3", "file": "lec5.pdf", "page": 7, "text": "Adapters are small."}]


def answer(text):
    return reply({"role": "assistant", "content": text})


def run(monkeypatch, *replies, **switches):
    """One turn with these LLM replies, searching PASSAGES. Returns run_turn's output and every request sent."""
    sent, it = [], iter(replies)

    def chat(messages, model, **kw):
        sent.append(list(messages))
        return next(it)

    monkeypatch.setattr(s1_llm, "chat", chat)
    return turn("What is LoRA?", "alice", 1, search_fn=lambda student, query: PASSAGES, **switches), sent


def cited(monkeypatch, text):
    return run(monkeypatch, reply(SEARCH_CALL), answer(text))[0]["citations"]


def test_the_tutor_is_told_to_cite_by_passage_id_in_square_brackets(monkeypatch):
    _, sent = run(monkeypatch, answer("Hello."))
    assert "[p1]" in sent[0][0]["content"]


def test_citations_are_the_ids_in_the_answer_in_order_of_first_appearance_without_repeats(monkeypatch):
    assert cited(monkeypatch, "BM25 ranks [p2]. LoRA adds [p1]. BM25 again [p2].") == ["p2", "p1"]


def test_several_ids_in_one_bracket_are_all_cited(monkeypatch):
    assert cited(monkeypatch, "Both say so [p3, p1].") == ["p3", "p1"]


def test_an_id_that_was_not_retrieved_this_turn_is_ignored(monkeypatch):
    assert cited(monkeypatch, "Made up [p9], real [p1].") == ["p1"]


def test_a_turn_with_no_search_cites_nothing_even_if_the_answer_names_an_id(monkeypatch):
    out, _ = run(monkeypatch, answer("From last turn [p1]."))
    assert out["citations"] == []


def test_a_retrieved_passage_the_answer_does_not_use_is_not_returned(monkeypatch):
    assert cited(monkeypatch, "LoRA adds [p1].") == ["p1"]
    assert cited(monkeypatch, "No citations here.") == []


def test_an_id_that_only_starts_with_a_retrieved_id_is_not_that_passage(monkeypatch):
    assert cited(monkeypatch, "Made up [p10].") == []


def test_the_answer_keeps_its_ids(monkeypatch):
    out, _ = run(monkeypatch, reply(SEARCH_CALL), answer("LoRA adds [p1]."))
    assert out["answer"] == "LoRA adds [p1]."


def test_a_turn_that_calls_make_quiz_returns_no_citations(monkeypatch):
    quiz_call = {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                 "function": {"name": "make_quiz", "arguments": '{"concept": "LoRA"}'}}]}
    out, _ = run(monkeypatch, reply(quiz_call), answer('[{"question": "What does LoRA add?", "passage_ids": ["p1"]}]'),
                 answer("1. What does LoRA add? [p1]"))
    assert out["answer"] == "1. What does LoRA add? [p1]" and out["citations"] == []


def test_the_walkthrough_notebook_keeps_no_output_in_the_old_shape():
    notebook = (Path(__file__).parents[1] / "agent_walkthrough.ipynb").read_text(encoding="utf-8")
    assert 'result[\\"citations\\"]' in notebook
    assert "CITATIONS: [{" not in notebook


def test_citations_come_from_the_regenerated_answer_not_the_one_the_judge_failed(monkeypatch):
    out, _ = run(monkeypatch, reply(SEARCH_CALL), answer("Wrong [p1]."),
                 answer('{"supported": false, "reason": "unstated"}'), answer("Right [p2]."),
                 answer('{"supported": true, "reason": "stated"}'), faithfulness=True)
    assert out["answer"] == "Right [p2]." and out["citations"] == ["p2"]
