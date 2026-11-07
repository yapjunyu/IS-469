"""Reading pictures: every LLM call over passages receives their pictures, on by default."""
import base64
import json

import pytest

from agent import s1_llm
from agent.memory import learner
from agent.s2_tools import run_tool
from agent.tests.test_llm_client import fakes
from agent.tests.test_loop import SEARCH_CALL, read_trace, reply, tool_ctx, turn

ANSWER = {"role": "assistant", "content": "The figure shows a cat [p1]."}


@pytest.fixture
def figure(tmp_path):
    """figure("p1") is an image passage whose PNG file exists, with bytes unique to that id."""
    def make(passage_id, text=""):
        path = tmp_path / f"{passage_id}.png"
        path.write_bytes(f"png of {passage_id}".encode())
        return {"id": passage_id, "file": "lec3.pdf", "page": 12, "page_end": 12, "kind": "slide",
                "text": text, "image_path": str(path)}
    return make


def data_url(passage_id):
    return "data:image/png;base64," + base64.b64encode(f"png of {passage_id}".encode()).decode()


def pictures(message):
    """The picture URLs in one message, in order."""
    content = message.get("content")  # an assistant's tool call has none
    return [p["image_url"]["url"] for p in content if p["type"] == "image_url"] if isinstance(content, list) else []


def all_pictures(messages):
    return [url for m in messages for url in pictures(m)]


def text_of(message):
    """The text of one message, whether it holds pictures or not."""
    content = message["content"]
    return content if isinstance(content, str) else " ".join(p["text"] for p in content if p["type"] == "text")


def recording(monkeypatch, *replies):
    """Fake chat that returns the replies in order and records a copy of each request's messages."""
    sent, it = [], iter(replies)
    def chat(messages, model, **kw):
        sent.append(list(messages))
        return next(it)
    monkeypatch.setattr(s1_llm, "chat", chat)
    return sent


def search_returning(*results):
    """A search function that gives each result in turn."""
    it = iter(results)
    return lambda student, query: next(it)


def test_next_request_after_a_picture_has_it_named_as_material(monkeypatch, figure):
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER))
    out = turn("What does the figure show?", "alice", 1, search_fn=search_returning([figure("p1")]))
    assert out["answer"] == ANSWER["content"] and out["citations"] == ["p1"]
    assert all_pictures(sent[0]) == []
    last = sent[1][-1]
    assert last["role"] == "user" and pictures(last) == [data_url("p1")]
    words = text_of(last)
    assert "p1" in words and "Student-files material" in words


def test_picture_message_comes_after_every_tool_result_of_a_step(monkeypatch, figure):
    two_calls = {"role": "assistant", "tool_calls": [
        {"id": c, "type": "function", "function": {"name": "search_notes", "arguments": '{"query": "cat"}'}}
        for c in ("c1", "c2")]}
    sent = recording(monkeypatch, reply(two_calls), reply(ANSWER))
    turn("q", "alice", 1, search_fn=search_returning([figure("p1")], [figure("p2")]))
    # a user message between tool results is a provider 400: they must directly follow their tool calls
    assert [m["role"] for m in sent[1][-4:]] == ["assistant", "tool", "tool", "user"]
    assert pictures(sent[1][-1]) == [data_url("p1"), data_url("p2")]


def test_request_holds_the_newest_three_pictures(monkeypatch, figure):
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(SEARCH_CALL), reply(ANSWER))
    turn("q", "alice", 1, search_fn=search_returning([figure("p1"), figure("p2"), figure("p3")],
                                                     [figure("p4"), figure("p5")]))
    assert sorted(all_pictures(sent[2])) == sorted(data_url(p) for p in ("p3", "p4", "p5"))


def test_a_picture_found_twice_is_sent_once_and_counts_as_new(monkeypatch, figure):
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(SEARCH_CALL), reply(ANSWER))
    turn("q", "alice", 1, search_fn=search_returning([figure("p1"), figure("p2"), figure("p3")],
                                                     [figure("p4"), figure("p1")]))
    assert sorted(all_pictures(sent[2])) == sorted(data_url(p) for p in ("p3", "p4", "p1"))


def test_later_requests_keep_the_pictures_when_no_new_ones_arrive(monkeypatch, figure):
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(SEARCH_CALL), reply(ANSWER))
    turn("q", "alice", 1, search_fn=search_returning([figure("p1")], [{"id": "t1", "file": "a.pdf", "page": 1, "text": "x"}]))
    assert all_pictures(sent[2]) == [data_url("p1")]


def test_switch_off_sends_no_pictures_and_leaves_out_textless_passages(monkeypatch, figure):
    both = [figure("p1"), figure("p2", text="A cat has whiskers.")]
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply({"role": "assistant", "content": "Whiskers [p2]. [p1]"}))
    out = turn("q", "alice", 1, images=False, search_fn=search_returning(both))
    assert all_pictures(sent[1]) == []
    tool = next(m for m in sent[1] if m["role"] == "tool")
    assert "p2" in tool["content"] and "p1" not in tool["content"] and sent[1][-1]["role"] == "tool"
    assert out["citations"] == ["p2"]  # the tutor never received p1, so it was not retrieved


def test_passages_without_an_image_path_field_send_no_picture_message(monkeypatch):
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER))
    out = turn("q", "alice", 1)  # the default stub_search has no image_path field
    assert out["answer"] == ANSWER["content"] and sent[1][-1]["role"] == "tool"


def test_a_text_passage_with_image_path_none_sends_no_picture_message(monkeypatch):
    passage = {"id": "p1", "file": "a.pdf", "page": 1, "page_end": 1, "kind": "pdf", "text": "x", "image_path": None}
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER))
    turn("q", "alice", 1, search_fn=search_returning([passage]))
    assert sent[1][-1]["role"] == "tool"


def test_a_missing_picture_file_is_skipped_and_the_turn_still_answers(monkeypatch, figure, tmp_path):
    gone, kept = figure("p1"), figure("p2")
    (tmp_path / "p1.png").unlink()
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER))
    out = turn("q", "alice", 1, search_fn=search_returning([gone, kept]))
    assert out["answer"] == ANSWER["content"]
    assert all_pictures(sent[1]) == [data_url("p2")]
    assert "p1" not in text_of(sent[1][-1])


def test_a_textless_passage_whose_picture_file_is_gone_is_left_out_of_every_llm_request(monkeypatch, figure, tmp_path):
    gone, kept = figure("p1"), figure("p2", text="A cat has whiskers.")
    (tmp_path / "p1.png").unlink()
    verdict = {"role": "assistant", "content": '{"supported": true, "reason": "ok"}'}
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply({"role": "assistant", "content": "Whiskers [p2]. [p1]"}), reply(verdict))
    out = turn("q", "alice", 1, faithfulness=True, search_fn=search_returning([gone, kept]))
    tool = next(m for m in sent[1] if m["role"] == "tool")
    judge = text_of(sent[2][1])
    assert tool["content"].count("<document") == 1 and 'id="p2"' in tool["content"]
    assert judge.count("<document") == 1 and 'id="p2"' in judge
    assert out["citations"] == ["p2"]  # the tutor never received p1, so it was not retrieved


def test_a_passage_with_text_stays_when_its_picture_file_is_gone(monkeypatch, figure, tmp_path):
    gone = figure("p1", text="A cat has whiskers.")
    (tmp_path / "p1.png").unlink()
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER))
    turn("q", "alice", 1, search_fn=search_returning([gone]))
    assert "A cat has whiskers." in next(m for m in sent[1] if m["role"] == "tool")["content"]


def test_only_missing_picture_files_send_no_picture_message(monkeypatch, figure, tmp_path):
    gone = figure("p1")
    (tmp_path / "p1.png").unlink()
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER))
    turn("q", "alice", 1, search_fn=search_returning([gone]))
    assert sent[1][-1]["role"] == "tool"


def test_trace_answer_and_history_hold_no_image_data(monkeypatch, figure, tmp_data):
    recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER))
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    before = json.dumps(history)
    out = turn("q", "alice", 1, history=history, search_fn=search_returning([figure("p1")]))
    trace = (tmp_data / "trace.jsonl").read_text(encoding="utf-8")
    encoded = base64.b64encode(b"png of p1").decode()
    assert encoded not in trace and "base64" not in trace and ".png" not in trace  # not even the local path
    assert json.dumps(history) == before  # the caller's list gained nothing
    assert "base64" not in json.dumps(out)
    assert [r["kind"] for r in read_trace(tmp_data)] == ["llm", "tool", "llm", "final"]


def test_guard_off_tool_result_does_not_send_the_local_path(monkeypatch, figure):
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER))
    turn("q", "alice", 1, guard=False, search_fn=search_returning([figure("p1")]))
    tool = next(m for m in sent[1] if m["role"] == "tool")
    assert json.loads(tool["content"])["passages"] == [{"id": "p1", "file": "lec3.pdf", "page": 12, "text": ""}]
    assert pictures(sent[1][-1]) == [data_url("p1")]


def picture_in_only_user_message(messages):
    """ask's requests are one system message and one user message holding text and pictures."""
    assert [m["role"] for m in messages] == ["system", "user"]
    return pictures(messages[1])


def test_quiz_call_receives_the_picture(monkeypatch, figure):
    learner.add_concepts("alice", "lec3.pdf", ["cats"])
    sent = recording(monkeypatch, reply({"role": "assistant", "content": '[{"question": "q?", "passage_ids": ["p1"]}]'}))
    out = run_tool("make_quiz", '{"concept": "cats"}', tool_ctx(search_fn=search_returning([figure("p1")])))
    assert out["questions"][0]["passage_ids"] == ["p1"]
    assert picture_in_only_user_message(sent[0]) == [data_url("p1")]


def test_grading_call_receives_the_picture(monkeypatch, figure):
    sent = recording(monkeypatch, reply({"role": "assistant", "content": '{"correct": true, "hint": ""}'}))
    args = '{"question": "q", "student_answer": "a", "passage_ids": ["p1"]}'
    out = run_tool("grade_answer", args, tool_ctx(passages_fn=lambda s, ids: [figure("p1")]))
    assert out["correct"] is True
    assert picture_in_only_user_message(sent[0]) == [data_url("p1")]


def test_search_again_call_receives_the_picture(monkeypatch, figure):
    sent = recording(monkeypatch, reply({"role": "assistant", "content": '{"enough": true, "query": ""}'}))
    ctx = tool_ctx(search_again=True, search_fn=search_returning([figure("p1")]))
    run_tool("search_notes", '{"query": "cat"}', ctx)
    assert picture_in_only_user_message(sent[0]) == [data_url("p1")]


def test_judge_call_receives_the_pictures_the_tutor_last_saw(monkeypatch, figure):
    held = {p: figure(p) for p in ("p1", "p2", "p3", "p4")}
    verdict = {"role": "assistant", "content": '{"supported": true, "reason": "ok"}'}
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(SEARCH_CALL), reply(ANSWER), reply(verdict))
    out = turn("q", "alice", 1, faithfulness=True,
               search_fn=search_returning([held["p1"], held["p2"], held["p3"]], [held["p4"], held["p1"]]),
               passages_fn=lambda s, ids: [held[i] for i in ids])
    assert out["warning"] is False
    assert sorted(picture_in_only_user_message(sent[3])) == sorted(data_url(p) for p in ("p3", "p4", "p1"))
    assert sorted(all_pictures(sent[2])) == sorted(picture_in_only_user_message(sent[3]))


def test_tutor_and_judge_hold_the_same_pictures_after_a_search_then_a_grade_on_others(monkeypatch, figure):
    held = {p: figure(p) for p in ("p1", "p2", "p3", "p4", "p5")}
    grade = {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "grade_answer",
             "arguments": '{"question": "q", "student_answer": "a", "passage_ids": ["p4", "p5"]}'}}]}
    graded = {"role": "assistant", "content": '{"correct": false, "hint": "Reread."}'}
    verdict = {"role": "assistant", "content": '{"supported": true, "reason": "ok"}'}
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(grade), reply(graded), reply(ANSWER), reply(verdict))
    turn("q", "alice", 1, faithfulness=True, search_fn=search_returning([held["p1"], held["p2"], held["p3"]]),
         passages_fn=lambda s, ids: [held[i] for i in ids])
    newest_three = sorted(data_url(p) for p in ("p3", "p4", "p5"))
    tutor, judge = sent[3], sent[4]
    assert sorted(all_pictures(tutor)) == newest_three
    assert sorted(picture_in_only_user_message(judge)) == newest_three


def test_regenerate_call_still_holds_the_picture(monkeypatch, figure):
    fail = {"role": "assistant", "content": '{"supported": false, "reason": "no"}'}
    ok = {"role": "assistant", "content": '{"supported": true, "reason": "ok"}'}
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER), reply(fail), reply(ANSWER), reply(ok))
    turn("q", "alice", 1, faithfulness=True, search_fn=search_returning([figure("p1")]),
         passages_fn=lambda s, ids: [figure("p1")])
    assert all_pictures(sent[3]) == [data_url("p1")]


def test_switch_off_judge_gets_no_picture_and_no_textless_passage(monkeypatch, figure):
    verdict = {"role": "assistant", "content": '{"supported": true, "reason": "ok"}'}
    sent = recording(monkeypatch, reply(SEARCH_CALL), reply(ANSWER), reply(verdict))
    out = turn("q", "alice", 1, images=False, faithfulness=True,
               search_fn=search_returning([figure("p1"), figure("p2", text="Whiskers.")]))
    assert out["faithfulness"]["supported"] is True
    judge = sent[2]
    assert all_pictures(judge) == []
    assert judge[1]["content"].count("<document") == 1 and 'id="p2"' in judge[1]["content"]


def test_a_call_over_passages_keeps_the_newest_three_pictures(monkeypatch, figure):
    sent = recording(monkeypatch, reply({"role": "assistant", "content": "[]"}))
    run_tool("make_quiz", '{"concept": "cats"}',
             tool_ctx(search_fn=search_returning([figure(p) for p in ("p1", "p2", "p3", "p4")])))
    assert picture_in_only_user_message(sent[0]) == [data_url(p) for p in ("p2", "p3", "p4")]


def test_picture_bytes_are_part_of_the_llm_cache_key(monkeypatch, figure, tmp_path):
    clients = fakes(monkeypatch)
    passage = figure("p1")
    ask = lambda: run_tool("make_quiz", '{"concept": "cats"}', tool_ctx(search_fn=lambda s, q: [passage]))

    ask()
    assert len(clients["opencode"].requests) == 1
    assert pictures(clients["opencode"].requests[0]["messages"][1]) == [data_url("p1")]

    ask()  # same picture bytes: answered from the cache
    assert len(clients["opencode"].requests) == 1

    (tmp_path / "p1.png").write_bytes(b"a re-rendered slide")  # same path, different picture
    ask()
    assert len(clients["opencode"].requests) == 2
