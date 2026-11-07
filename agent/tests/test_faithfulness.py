"""The judge checks the final answer against every passage retrieved in the turn, before the student sees it."""
import json

from agent import s1_llm
from agent.memory import learner
from agent.s7_faithfulness import JUDGE_MODEL
from agent.tests.test_loop import SEARCH_CALL, read_trace, reply, turn

PASSAGES = [{"id": "p1", "file": "lec3.pdf", "page": 12, "text": "LoRA adds low-rank matrices."}]


def run(monkeypatch, *verdicts, **switches):
    """One search-then-answer turn; the judge gives the verdicts in order. Returns the output and judge requests."""
    tutor = iter([reply(SEARCH_CALL), reply({"role": "assistant", "content": "LoRA adds low-rank matrices."}),
                  reply({"role": "assistant", "content": "Second try."})])
    judged, verdicts = [], iter(verdicts)
    def chat(messages, model, **kw):
        if model != JUDGE_MODEL:
            return next(tutor)
        judged.append(messages)
        supported = next(verdicts)
        return reply({"role": "assistant", "content": json.dumps({"supported": supported, "reason": "why"})}, 50, 8)
    monkeypatch.setattr(s1_llm, "chat", chat)
    out = turn("What is LoRA?", "alice", 1, search_fn=lambda s, q: PASSAGES,
               passages_fn=lambda s, ids: [p for p in PASSAGES if p["id"] in ids], **{"faithfulness": True, **switches})
    return out, judged


def test_supported_answer_returned_unchanged_with_verdict_on_final_row(monkeypatch, tmp_data):
    out, judged = run(monkeypatch, True)
    rows = read_trace(tmp_data)
    assert out["answer"] == "LoRA adds low-rank matrices." and not out["warning"]
    assert rows[-1]["kind"] == "final" and rows[-1]["faithfulness"] == {"supported": True, "reason": "why"}
    judge_input = next(m["content"] for m in judged[0] if m["role"] == "user")
    assert '<document id="p1" file="lec3.pdf" page="12">' in judge_input  # the retrieved passage, not just the answer


def test_judge_calls_write_llm_rows(monkeypatch, tmp_data):
    run(monkeypatch, True)
    judge = [r for r in read_trace(tmp_data) if r["kind"] == "llm" and r["purpose"] == "judge"]
    assert len(judge) == 1 and judge[0]["model"] == JUDGE_MODEL and judge[0]["tokens"] == {"in": 50, "out": 8}


def test_unsupported_answer_regenerates_once_then_passes(monkeypatch, tmp_data):
    out, judged = run(monkeypatch, False, True)
    assert out["answer"] == "Second try." and not out["warning"] and len(judged) == 2


def test_second_fail_returns_the_answer_with_a_warning(monkeypatch, tmp_data):
    out, judged = run(monkeypatch, False, False)
    purposes = [r["purpose"] for r in read_trace(tmp_data) if r["kind"] == "llm"]
    assert out["answer"] == "Second try." and out["warning"]
    assert purposes == ["tutor", "tutor", "judge", "regenerate", "judge"]
    assert read_trace(tmp_data)[-1]["faithfulness"]["supported"] is False


def test_regenerate_can_search_before_its_answer_is_judged_again(monkeypatch, tmp_data):
    # the first answer came from memory with no passages; the regenerate searches, then answers
    tutor = iter([reply({"role": "assistant", "content": "LoRA is magic."}), reply(SEARCH_CALL),
                  reply({"role": "assistant", "content": "LoRA adds low-rank matrices. [p1]"})])
    judged, verdicts = [], iter([False, True])
    def chat(messages, model, **kw):
        if model != JUDGE_MODEL:
            return next(tutor)
        judged.append(next(m["content"] for m in messages if m["role"] == "user"))
        return reply({"role": "assistant", "content": json.dumps({"supported": next(verdicts), "reason": "why"})})
    monkeypatch.setattr(s1_llm, "chat", chat)
    out = turn("What is LoRA?", "alice", 1, search_fn=lambda s, q: PASSAGES, faithfulness=True)
    rows = read_trace(tmp_data)
    assert out["answer"] == "LoRA adds low-rank matrices. [p1]" and not out["warning"] and out["citations"] == ["p1"]
    assert [r.get("purpose", r["kind"]) for r in rows] == ["tutor", "judge", "regenerate", "tool", "tutor", "judge", "final"]
    assert "<document" not in judged[0] and '<document id="p1"' in judged[1]


def test_the_judge_reads_the_students_record(monkeypatch):
    learner.add_concepts("alice", "notes.pdf", ["LoRA"])
    learner.update_mastery("alice", "LoRA", False)  # 0.35: 0.5 - 0.3 * 0.5, a weak spot
    out, judged = run(monkeypatch, True)
    judge_input = next(m["content"] for m in judged[0] if m["role"] == "user")
    assert "lora 0.35" in judge_input


def test_switch_off_makes_no_judge_call(monkeypatch, tmp_data):
    out, judged = run(monkeypatch, faithfulness=False)
    assert judged == [] and not out["warning"] and read_trace(tmp_data)[-1]["faithfulness"] is None
