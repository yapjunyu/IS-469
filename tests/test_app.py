"""The Streamlit app through AppTest: the LLM is scripted, the notebook's models are fakes, every file goes to a temp folder."""
import inspect
import json
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from agent import s1_llm, s3_loop
from agent.memory import learner
from agent.tests.test_loop import SEARCH_CALL, read_trace, reply
from test_notebook import fake_models, figure_pdf, make_pdf  # noqa: F401  (fake_models is autouse: importing it keeps real models out of this file)
from tutor import notebook
from tutor.retrieval.chunk import PASSAGE_ID

ROOT = Path(__file__).resolve().parents[1]


def open_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()


def buttons(at, label):
    return [b for b in at.button if b.label == label]


def enter_name(at, name):
    at.text_input[0].input(name)
    buttons(at, "Open my notebook")[0].click()
    return at.run()


def seed_file(student="jun-yu", name="notes.pdf"):
    """A file in the notebook without indexing it: enough for the app to list it and unlock chat."""
    folder = notebook.UPLOADS_DIR / student
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"%PDF")


def files_listed(at):
    return [t.value for t in at.sidebar.text]


# entering a notebook

def test_an_invalid_name_shows_a_clear_message_and_opens_no_notebook(tmp_data):
    at = enter_name(open_app(), "Jun!")
    assert "can't be used" in at.error[0].value
    assert not at.file_uploader and not at.chat_input
    assert not (tmp_data / "uploads").exists()


def test_a_name_reopens_its_notebook_whatever_the_capitals_and_spacing():
    seed_file("jun-yu", "notes.pdf")
    seed_file("bob", "bobs.pdf")
    for name in ["Jun Yu", "jun yu", "  JUN   yu "]:
        assert files_listed(enter_name(open_app(), name)) == ["notes.pdf"]


# uploading

def upload(at, tmp_path):
    content = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small").read_bytes()
    at.file_uploader[0].set_value(("notes.pdf", content, "application/pdf"))
    return at.run()


def test_an_uploaded_file_joins_the_list_and_unlocks_chat(tmp_path):
    at = enter_name(open_app(), "Jun Yu")
    assert not at.chat_input
    at = upload(at, tmp_path)
    assert files_listed(at) == ["notes.pdf"]
    assert "notes.pdf" in at.sidebar.success[0].value
    assert at.chat_input
    assert learner.concepts("jun-yu") == ["adapters", "lora"]
    assert [p["file"] for p in notebook.search("jun-yu", "LoRA")] == ["notes.pdf"]


def test_a_rejected_file_shows_why_and_stays_out_of_the_notebook():
    at = enter_name(open_app(), "Jun Yu")
    at.file_uploader[0].set_value(("bad.pdf", b"not a pdf", "application/pdf"))
    at.run()
    assert "could not be opened" in at.sidebar.error[0].value
    assert files_listed(at) == []
    assert not at.chat_input


def test_an_indexing_failure_names_the_file_is_not_retried_and_leaves_chat_working(monkeypatch, tmp_path):
    calls = []

    def fails(*args):
        calls.append(args)
        raise RuntimeError("embedder out of memory")

    monkeypatch.setattr(notebook, "add_file", fails)
    seed_file(name="held.pdf")
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    at.run()
    assert len(calls) == 1
    assert "notes.pdf" in at.sidebar.error[0].value and "embedder out of memory" in at.sidebar.error[0].value
    assert files_listed(at) == ["held.pdf"]
    assert at.chat_input


def test_a_file_whose_concepts_could_not_be_listed_says_so_and_how_to_retry(monkeypatch, tmp_path):
    def no_llm(*args, **kwargs):
        raise RuntimeError("rate limit reached")

    monkeypatch.setattr(s1_llm, "chat", no_llm)
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    notice = at.sidebar.warning[0].value
    assert "notes.pdf" in notice and "concepts" in notice and "re-upload" in notice
    assert files_listed(at) == ["notes.pdf"]
    assert at.chat_input


def test_a_rerun_does_not_index_the_same_file_again_and_keeps_its_message(monkeypatch, tmp_path):
    calls = []
    real = notebook.add_file
    monkeypatch.setattr(notebook, "add_file", lambda *a: calls.append(a) or real(*a))
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    at.run()
    assert len(calls) == 1
    assert "notes.pdf" in at.sidebar.success[0].value


def test_a_copy_under_another_name_names_the_file_already_holding_it(tmp_path):
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    content = (tmp_path / "notes.pdf").read_bytes()
    at.file_uploader[0].set_value(("copy.pdf", content, "application/pdf"))
    at.run()
    assert "notes.pdf" in at.sidebar.info[0].value
    assert files_listed(at) == ["notes.pdf"]


# replacing a file

def upload_new_version(at, tmp_path, *slides):
    (tmp_path / "new").mkdir(exist_ok=True)
    content = make_pdf(tmp_path / "new" / "notes.pdf", *slides).read_bytes()
    at.file_uploader[0].set_value(("notes.pdf", content, "application/pdf"))
    return at.run()


def test_a_new_version_says_it_replaced_the_file_and_the_tutor_finds_only_it(tmp_path):
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    at = upload_new_version(at, tmp_path, "BM25 ranks by keyword")
    assert "replaced" in at.sidebar.success[0].value and "notes.pdf" in at.sidebar.success[0].value
    assert files_listed(at) == ["notes.pdf"]
    assert [p["text"] for p in notebook.search("jun-yu", "LoRA")] == ["BM25 ranks by keyword"]


def test_a_new_version_whose_concepts_could_not_be_listed_says_it_replaced_the_file_and_how_to_retry(monkeypatch, tmp_path):
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)

    def no_llm(*args, **kwargs):
        raise RuntimeError("rate limit reached")

    monkeypatch.setattr(s1_llm, "chat", no_llm)
    notice = upload_new_version(at, tmp_path, "BM25 ranks by keyword").sidebar.warning[0].value
    assert "replaced" in notice and "concepts" in notice and "re-upload" in notice


def test_a_rejected_new_version_shows_why_and_the_old_file_stays(tmp_path):
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    at.file_uploader[0].set_value(("notes.pdf", b"not a pdf", "application/pdf"))
    at.run()
    assert "could not be opened" in at.sidebar.error[0].value
    assert files_listed(at) == ["notes.pdf"]
    assert [p["text"] for p in notebook.search("jun-yu", "LoRA")] == ["LoRA adapters are small"]
    assert at.chat_input


def a_pdf(tmp_path, name, *slides):
    return name, make_pdf(tmp_path / "built.pdf", *slides).read_bytes(), "application/pdf"


def in_the_uploader_together(at, *files):
    at.file_uploader[0].set_value(list(files))  # the real uploader keeps every upload, so older ones are handed back too
    return at.run()


def test_a_new_version_clears_the_older_notices_about_that_file_and_keeps_its_own(tmp_path):
    old, new = a_pdf(tmp_path, "notes.pdf", "LoRA adapters are small"), a_pdf(tmp_path, "notes.pdf", "BM25 ranks by keyword")
    copy = ("copy.pdf", old[1], "application/pdf")
    at = in_the_uploader_together(enter_name(open_app(), "Jun Yu"), old, copy, new)
    for _ in range(2):  # the run that indexes them, and a later one
        assert len(at.sidebar.success) == 1 and "replaced" in at.sidebar.success[0].value
        assert not at.sidebar.info
        at.run()


def test_a_third_version_leaves_one_notice(tmp_path):
    versions = [a_pdf(tmp_path, "notes.pdf", text) for text in ("LoRA one", "LoRA two", "LoRA three")]
    at = in_the_uploader_together(enter_name(open_app(), "Jun Yu"), *versions)
    assert len(at.sidebar.success) == 1 and "replaced" in at.sidebar.success[0].value


def test_a_new_version_leaves_notices_about_other_files(tmp_path):
    other = a_pdf(tmp_path, "other.pdf", "An agent calls tools")
    old, new = a_pdf(tmp_path, "notes.pdf", "LoRA adapters are small"), a_pdf(tmp_path, "notes.pdf", "BM25 ranks by keyword")
    at = in_the_uploader_together(enter_name(open_app(), "Jun Yu"), other, old, new)
    assert ["other.pdf" in s.value for s in at.sidebar.success] == [True, False]


def test_a_new_version_differing_only_in_capitals_clears_the_notices_and_removing_it_clears_its_own(tmp_path):
    old, new = a_pdf(tmp_path, "notes.pdf", "LoRA adapters are small"), a_pdf(tmp_path, "Notes.PDF", "BM25 ranks by keyword")
    copy = ("copy.pdf", old[1], "application/pdf")
    at = in_the_uploader_together(enter_name(open_app(), "Jun Yu"), old, copy, new)
    assert len(at.sidebar.success) == 1 and "replaced" in at.sidebar.success[0].value
    assert not at.sidebar.info
    at = remove(at, "Notes.PDF")
    assert not at.sidebar.success and not at.sidebar.info
    assert files_listed(at) == []


# removing a file

def remove(at, name="notes.pdf"):
    buttons(at, f"Remove {name}")[0].click()
    return at.run()


def test_a_remove_button_takes_the_file_out_of_the_list_and_the_notebook(tmp_path):
    seed_file(name="kept.pdf")
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    assert files_listed(at) == ["kept.pdf", "notes.pdf"]
    at = remove(at)
    assert files_listed(at) == ["kept.pdf"]
    assert notebook.files("jun-yu") == ["kept.pdf"]
    assert notebook.search("jun-yu", "LoRA") == []
    assert learner.concepts("jun-yu") == []
    assert not buttons(at, "Remove notes.pdf")
    assert at.chat_input  # another file is still there


def test_removing_the_last_file_locks_chat_again(tmp_path):
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    assert at.chat_input
    at = remove(at)
    assert files_listed(at) == []
    assert not at.chat_input
    assert "upload" in at.info[0].value.lower()


def test_an_upload_still_in_the_uploader_is_not_indexed_again_and_stops_saying_it_was_added(monkeypatch, tmp_path):
    calls, real = [], notebook.add_file
    monkeypatch.setattr(notebook, "add_file", lambda *a: calls.append(a) or real(*a))
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    assert at.sidebar.success
    at = remove(at)
    assert at.file_uploader[0].value  # it is still in the widget, which hands it back on every rerun
    for _ in range(2):
        assert not at.sidebar.success
        assert files_listed(at) == []
        assert len(calls) == 1
        at.run()


def test_a_copy_still_in_the_uploader_stops_naming_the_removed_file_and_is_not_indexed(tmp_path):
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    at.file_uploader[0].set_value(("copy.pdf", (tmp_path / "notes.pdf").read_bytes(), "application/pdf"))
    at.run()
    assert "notes.pdf" in at.sidebar.info[0].value
    at = remove(at)
    assert not at.sidebar.info
    assert files_listed(at) == []


def test_uploading_a_removed_file_again_indexes_it_again(tmp_path):
    at = remove(upload(enter_name(open_app(), "Jun Yu"), tmp_path))
    at = upload(at, tmp_path)  # a new upload has a new file_id, whatever its content
    assert "added" in at.sidebar.success[0].value
    assert files_listed(at) == ["notes.pdf"]
    assert [p["text"] for p in notebook.search("jun-yu", "LoRA")] == ["LoRA adapters are small"]
    assert at.chat_input


def test_a_rejected_upload_keeps_its_message_when_an_unrelated_file_is_removed():
    seed_file(name="held.pdf")
    at = enter_name(open_app(), "Jun Yu")
    at.file_uploader[0].set_value(("bad.pdf", b"not a pdf", "application/pdf"))
    at.run()
    at = remove(at, "held.pdf")
    assert "could not be opened" in at.sidebar.error[0].value


def test_a_removal_that_fails_says_so_and_the_file_stays_listed(monkeypatch):
    def fails(*args):
        raise OSError("file in use")

    monkeypatch.setattr(notebook, "remove_file", fails)
    seed_file()
    at = remove(enter_name(open_app(), "Jun Yu"))
    assert "notes.pdf" in at.sidebar.error[0].value and "file in use" in at.sidebar.error[0].value
    assert files_listed(at) == ["notes.pdf"]
    assert at.chat_input


# the chat screen

def test_with_no_files_chat_is_locked_with_a_prompt_to_upload():
    at = enter_name(open_app(), "Jun Yu")
    assert not at.chat_input
    assert "upload" in at.info[0].value.lower()


def test_a_notice_says_the_tutor_is_only_as_accurate_as_the_files():
    assert any("only as accurate as your files" in c.value for c in open_app().caption)
    seed_file()
    assert any("only as accurate as your files" in c.value for c in enter_name(open_app(), "Jun Yu").caption)


def test_no_screen_offers_an_experiment_switch_and_run_turn_gets_only_the_learner_memory(monkeypatch):
    # every run_turn option but what the app must hand it, read from the signature so a new switch is covered too
    switches = set(list(inspect.signature(s3_loop.run_turn).parameters)[3:]) - {"history", "search_fn", "passages_fn"}
    assert "faithfulness" in switches

    def labels(at):
        kinds = ("button", "checkbox", "toggle", "radio", "selectbox", "multiselect", "select_slider", "slider",
                 "number_input", "text_input", "text_area", "file_uploader")
        return [w.label.lower().replace(" ", "_") for kind in kinds for w in getattr(at, kind)]

    def names_a_switch(at):  # not "memory": the Memory view is a real screen, and the call below pins that switch
        return [l for l in labels(at) if any(s in l for s in switches - {"memory"})]

    received = []
    monkeypatch.setattr(s3_loop, "run_turn", lambda *a, **k: received.append(k) or {"answer": "Fine.", "warning": False})
    at = open_app()
    assert labels(at) and not names_a_switch(at)
    seed_file()
    at = say(enter_name(at, "Jun Yu"), "What is LoRA?")
    assert labels(at) and not names_a_switch(at)
    assert {k: v for k, v in received[0].items() if k in switches} == {"memory": "learner"}


# a turn

def script_llm(monkeypatch, *tutor_replies, supported=True, checks=None):
    """Fake s1_llm.chat. Calls with tools are the tutor's: each gets the next reply, a plain string as the answer,
    an Exception raised, a function called with the messages for the reply. Calls without are the checks: enough passages,
    `supported` from the judge, and any fields in `checks`: grading reads its own, and a quiz reads the list inside them.
    The judge and search-again stay on, so they're answered here, because the app exposes no switches to turn them off.
    Returns the messages each tutor call was sent."""
    replies, sent = iter(tutor_replies), []

    def chat(messages, model, tools=None, **kwargs):
        if not tools:
            return reply({"role": "assistant", "content": json.dumps({"enough": True, "supported": supported, "reason": "r", **(checks or {})})})
        sent.append(list(messages))
        next_reply = next(replies)
        if callable(next_reply):
            next_reply = next_reply(messages)
        if isinstance(next_reply, Exception):
            raise next_reply
        return reply(next_reply if isinstance(next_reply, dict) else {"role": "assistant", "content": next_reply})

    monkeypatch.setattr(s1_llm, "chat", chat)
    return sent


def chat_ready():
    seed_file()
    return enter_name(open_app(), "Jun Yu")


def say(at, message):
    at.chat_input[0].set_value(message)
    return at.run()


def resend(at):
    buttons(at, "Resend")[0].click()
    return at.run()


def shown(at):
    return [(m.name, m.markdown[0].value) for m in at.chat_message]


def history_sent(sent_messages):
    """What a tutor call was sent, without the system prompt."""
    return [(m["role"], m["content"]) for m in sent_messages[1:]]


def final_turns(tmp_data):
    return [r["turn"] for r in read_trace(tmp_data) if r["kind"] == "final"]


def test_a_message_reaches_the_tutor_and_the_answer_is_shown(monkeypatch):
    sent = script_llm(monkeypatch, "LoRA is a small adapter.")
    at = say(chat_ready(), "What is LoRA?")
    assert history_sent(sent[0]) == [("user", "What is LoRA?")]
    assert shown(at) == [("user", "What is LoRA?"), ("assistant", "LoRA is a small adapter.")]
    assert not at.warning and not at.error


def test_the_tutor_searches_the_students_real_notebook(monkeypatch, tmp_path):
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path)
    sent = script_llm(monkeypatch, SEARCH_CALL, "They are small.")
    say(at, "What is LoRA?")
    assert "LoRA adapters are small" in sent[1][-1]["content"]


def test_a_second_turn_gets_the_first_in_its_history_and_the_next_turn_number(monkeypatch, tmp_data):
    sent = script_llm(monkeypatch, "First answer.", "Second answer.")
    at = say(say(chat_ready(), "First question"), "Second question")
    assert history_sent(sent[1]) == [("user", "First question"), ("assistant", "First answer."), ("user", "Second question")]
    assert shown(at)[-1] == ("assistant", "Second answer.") and len(shown(at)) == 4
    assert final_turns(tmp_data) == [1, 2]


def test_an_answer_that_still_fails_the_judge_shows_a_warning(monkeypatch):
    script_llm(monkeypatch, "Unsupported claim.", "Still unsupported.", supported=False)
    at = say(chat_ready(), "What is LoRA?")
    assert shown(at)[-1] == ("assistant", "Still unsupported.")
    assert "Warning" in at.chat_message[1].warning[0].value


def test_a_warning_stays_on_its_own_answer_in_later_turns(monkeypatch):
    script_llm(monkeypatch, "Unsupported claim.", "Still unsupported.", supported=False)
    at = say(chat_ready(), "What is LoRA?")
    script_llm(monkeypatch, "Fine answer.")
    at = say(at, "And BM25?")
    assert [len(m.warning) for m in at.chat_message] == [0, 1, 0, 0]


# citations

LECTURE = {"id": "aaaaaaaaaaaa:0001", "file": "Lecture3.pdf", "page": 12, "page_end": 12, "kind": "page",
           "text": "LoRA adds low-rank matrices.", "image_path": None}
SLIDES = {"id": "bbbbbbbbbbbb:0002", "file": "Week5.pptx", "page": 4, "page_end": 6, "kind": "slide",
          "text": "BM25 ranks by keyword.", "image_path": None}
UNUSED = {"id": "cccccccccccc:0003", "file": "Lecture3.pdf", "page": 13, "page_end": 13, "kind": "page",
          "text": "Not used by the answer.", "image_path": None}
RAW_ANSWER = f"BM25 ranks [{SLIDES['id']}]. LoRA adds matrices [{LECTURE['id']}]. BM25 again [{SLIDES['id']}]."
QUIZ_CALL = {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
             "function": {"name": "make_quiz", "arguments": '{"concept": "LoRA"}'}}]}


def fake_notes(monkeypatch):
    """The notebook holds their files, and its search and lookup return these three passages. Tutor replies are scripted by the test."""
    passages = [LECTURE, SLIDES, UNUSED]
    seed_file("jun-yu", "Lecture3.pdf")
    seed_file("jun-yu", "Week5.pptx")
    monkeypatch.setattr(notebook, "search", lambda student, query: passages)
    monkeypatch.setattr(notebook, "get_passages", lambda student, ids: [p for i in ids for p in passages if p["id"] == i])


def test_an_answer_shows_numbered_markers_with_no_passage_ids(monkeypatch):
    fake_notes(monkeypatch)
    script_llm(monkeypatch, SEARCH_CALL, RAW_ANSWER)
    at = say(chat_ready(), "What is LoRA?")
    assert shown(at)[-1] == ("assistant", "BM25 ranks [1]. LoRA adds matrices [2]. BM25 again [1].")


def test_the_app_lists_only_the_cited_passages_by_label_in_marker_order(monkeypatch):
    fake_notes(monkeypatch)
    script_llm(monkeypatch, SEARCH_CALL, RAW_ANSWER)
    at = say(chat_ready(), "What is LoRA?")
    assert [e.label for e in at.chat_message[1].expander] == ["[1] Week5.pptx, slides 4–6", "[2] Lecture3.pdf, page 12"]


def test_opening_a_citation_shows_its_text(monkeypatch):
    fake_notes(monkeypatch)
    script_llm(monkeypatch, SEARCH_CALL, RAW_ANSWER)
    first, second = say(chat_ready(), "What is LoRA?").chat_message[1].expander
    assert first.text[0].value == "BM25 ranks by keyword."
    assert second.text[0].value == "LoRA adds low-rank matrices."
    assert not first.image and not second.image


def test_opening_a_citation_with_a_picture_shows_the_picture(monkeypatch, tmp_data):
    notebook.add_file("jun-yu", figure_pdf(tmp_data / "paper.pdf", "LoRA adapters, see the figure"))
    [picture] = [p for p in notebook.search("jun-yu", "LoRA") if p["image_path"]]
    script_llm(monkeypatch, SEARCH_CALL, f"See the figure [{picture['id']}].")
    at = say(enter_name(open_app(), "Jun Yu"), "What is LoRA?")
    [citation] = at.chat_message[1].expander
    assert citation.label == "[1] paper.pdf, page 1"
    assert len(citation.image) == 1


def test_removing_a_cited_file_takes_its_citation_away_and_the_chat_still_renders(monkeypatch, tmp_data):
    notebook.add_file("jun-yu", figure_pdf(tmp_data / "paper.pdf", "LoRA adapters, see the figure"))
    seed_file("jun-yu", "other.pdf")  # so removing paper.pdf doesn't lock the chat
    [picture] = [p for p in notebook.search("jun-yu", "LoRA") if p["image_path"]]
    script_llm(monkeypatch, SEARCH_CALL, f"See the figure [{picture['id']}].")
    at = say(enter_name(open_app(), "Jun Yu"), "What is LoRA?")
    assert at.chat_message[1].expander
    at = remove(at, "paper.pdf")
    assert not at.exception and not at.chat_message[1].expander
    assert shown(at)[-1] == ("assistant", "See the figure [1].")


def test_a_citation_left_after_another_file_goes_keeps_its_number(monkeypatch):
    fake_notes(monkeypatch)
    script_llm(monkeypatch, SEARCH_CALL, RAW_ANSWER)
    at = say(chat_ready(), "What is LoRA?")
    (notebook.UPLOADS_DIR / "jun-yu" / "Week5.pptx").unlink()
    at = at.run()
    assert [e.label for e in at.chat_message[1].expander] == ["[2] Lecture3.pdf, page 12"]


def test_a_cited_picture_lost_to_a_new_version_of_its_file_leaves_the_citation_without_it(monkeypatch, tmp_data):
    notebook.add_file("jun-yu", figure_pdf(tmp_data / "paper.pdf", "LoRA adapters, see the figure"))
    [picture] = [p for p in notebook.search("jun-yu", "LoRA") if p["image_path"]]
    script_llm(monkeypatch, SEARCH_CALL, f"See the figure [{picture['id']}].")
    at = say(enter_name(open_app(), "Jun Yu"), "What is LoRA?")
    notebook.add_file("jun-yu", figure_pdf(tmp_data / "paper.pdf", "LoRA adapters, a new version"))
    at = at.run()
    assert not at.exception and not Path(picture["image_path"]).exists()
    [citation] = at.chat_message[1].expander
    assert citation.label == "[1] paper.pdf, page 1" and not citation.image


def test_the_next_turn_gets_the_raw_ids_in_its_history(monkeypatch):
    fake_notes(monkeypatch)
    sent = script_llm(monkeypatch, SEARCH_CALL, RAW_ANSWER, "Later answer.")
    say(say(chat_ready(), "What is LoRA?"), "Later question")
    assert history_sent(sent[2]) == [("user", "What is LoRA?"), ("assistant", RAW_ANSWER), ("user", "Later question")]


def test_a_quiz_shows_no_passage_ids_and_no_citations_and_the_next_turn_still_gets_the_ids(monkeypatch):
    fake_notes(monkeypatch)
    raw_quiz = f"1. What does LoRA add? [{LECTURE['id']}, {UNUSED['id']}]\nWhich one are you answering?"
    sent = script_llm(monkeypatch, QUIZ_CALL, raw_quiz, "Not quite.")
    at = say(chat_ready(), "Quiz me on LoRA")
    assert shown(at)[-1] == ("assistant", "1. What does LoRA add?\nWhich one are you answering?")
    assert not at.chat_message[1].expander
    say(at, "Question 1: tiny matrices")
    assert history_sent(sent[2])[1] == ("assistant", raw_quiz)


def test_each_answer_keeps_its_own_citations_in_later_turns(monkeypatch):
    fake_notes(monkeypatch)
    script_llm(monkeypatch, SEARCH_CALL, RAW_ANSWER, SEARCH_CALL, f"Only this [{UNUSED['id']}].")
    at = say(say(chat_ready(), "What is LoRA?"), "And the rest?")
    assert shown(at)[1][1] == "BM25 ranks [1]. LoRA adds matrices [2]. BM25 again [1]."
    assert len(at.chat_message[1].expander) == 2
    assert [e.label for e in at.chat_message[3].expander] == ["[1] Lecture3.pdf, page 13"]


def test_a_student_message_is_shown_as_typed(monkeypatch):
    script_llm(monkeypatch, "Fine.")
    message = f"What is {LECTURE['id']}?"
    assert shown(say(chat_ready(), message))[0] == ("user", message)


# the quiz

def picker(at):
    return at.sidebar.selectbox[0], buttons(at, "Start quiz")[0]


def test_the_picker_lists_the_notebooks_concepts():
    seed_file()
    learner.add_concepts("jun-yu", "notes.pdf", ["LoRA", "KV cache"])
    concept, start = picker(enter_name(open_app(), "Jun Yu"))
    assert concept.options == ["kv cache", "lora"]
    assert not concept.disabled and not start.disabled


def test_the_picker_is_unavailable_when_the_notebook_has_no_concepts():
    seed_file()
    concept, start = picker(enter_name(open_app(), "Jun Yu"))
    assert concept.options == []
    assert concept.disabled and start.disabled


def start_quiz(at, concept):
    at.sidebar.selectbox[0].select(concept)
    buttons(at, "Start quiz")[0].click()
    return at.run()


def test_start_quiz_sends_a_normal_message_and_the_agent_calls_make_quiz(monkeypatch, tmp_data):
    fake_notes(monkeypatch)
    learner.add_concepts("jun-yu", "Lecture3.pdf", ["LoRA", "BM25"])
    sent = script_llm(monkeypatch, QUIZ_CALL, "1. What does LoRA add?")
    at = start_quiz(chat_ready(), "lora")
    assert history_sent(sent[0]) == [("user", "Quiz me on lora")]
    assert shown(at) == [("user", "Quiz me on lora"), ("assistant", "1. What does LoRA add?")]
    assert [r["inputs"] for r in read_trace(tmp_data) if r["kind"] == "tool" and r["tool"] == "make_quiz"] == [{"concept": "LoRA"}]


def test_a_rerun_after_start_quiz_does_not_send_it_again(monkeypatch, tmp_data):
    fake_notes(monkeypatch)
    learner.add_concepts("jun-yu", "Lecture3.pdf", ["LoRA"])
    script_llm(monkeypatch, QUIZ_CALL, "1. What does LoRA add?")
    start_quiz(chat_ready(), "lora").run()
    assert final_turns(tmp_data) == [1]


def grade_call_from_history(messages):
    """The tutor's grading call on question 1, with the passage IDs it reads after that question in last turn's answer."""
    ids = re.findall(PASSAGE_ID, messages[-2]["content"].splitlines()[0])
    return {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "grade_answer", "arguments": json.dumps(
        {"question": "What does LoRA add?", "student_answer": "tiny matrices", "passage_ids": ids})}}]}


def test_a_numbered_set_shows_without_ids_then_a_wrong_answer_is_graded_on_them_and_shows_the_hint(monkeypatch, tmp_data):
    fake_notes(monkeypatch)
    learner.add_concepts("jun-yu", "Lecture3.pdf", ["LoRA"])
    hint = "Reread what the lecture says LoRA adds."
    raw_quiz = (f"1. What does LoRA add? [{LECTURE['id']}, {UNUSED['id']}]\n"
                f"2. What does BM25 rank by? [{SLIDES['id']}]\nWhich question are you answering?")
    sent = script_llm(monkeypatch, QUIZ_CALL, raw_quiz, grade_call_from_history, f"Not quite. {hint}",
                      checks={"correct": False, "hint": hint,
                              "questions": [{"question": "What does LoRA add?", "passage_ids": [LECTURE["id"], UNUSED["id"]]},
                                            {"question": "What does BM25 rank by?", "passage_ids": [SLIDES["id"]]}]})
    at = start_quiz(chat_ready(), "lora")
    assert shown(at)[-1] == ("assistant", "1. What does LoRA add?\n2. What does BM25 rank by?\nWhich question are you answering?")
    assert not at.chat_message[1].expander

    at = say(at, "Question 1: tiny matrices")
    assert history_sent(sent[2])[1] == ("assistant", raw_quiz)
    [graded] = [r for r in read_trace(tmp_data) if r["kind"] == "tool" and r["tool"] == "grade_answer"]
    assert graded["ok"] and graded["inputs"]["passage_ids"] == [LECTURE["id"], UNUSED["id"]]
    assert hint in sent[3][-1]["content"]  # the tool's result reached the tutor
    assert shown(at)[-1] == ("assistant", f"Not quite. {hint}")


def test_a_hint_that_names_a_passage_id_offers_no_citation_to_open(monkeypatch, tmp_data):
    fake_notes(monkeypatch)
    learner.add_concepts("jun-yu", "Lecture3.pdf", ["LoRA"])
    raw_quiz = f"1. What does LoRA add? [{LECTURE['id']}]\nWhich question are you answering?"
    script_llm(monkeypatch, QUIZ_CALL, raw_quiz, grade_call_from_history, f"Not quite. Reread this [{LECTURE['id']}].",
               checks={"correct": False, "hint": "Reread.",
                       "questions": [{"question": "What does LoRA add?", "passage_ids": [LECTURE["id"]]}]})
    at = say(start_quiz(chat_ready(), "lora"), "Question 1: tiny matrices")
    assert shown(at)[-1] == ("assistant", "Not quite. Reread this.")
    assert not at.chat_message[3].expander


# a failed turn

def test_a_failed_turn_shows_the_error_and_keeps_the_message_with_a_resend_button(monkeypatch):
    script_llm(monkeypatch, RuntimeError("rate limit reached"))
    at = say(chat_ready(), "What is LoRA?")
    assert "rate limit reached" in at.error[0].value
    assert shown(at) == [("user", "What is LoRA?")]
    assert buttons(at, "Resend")


def test_resending_runs_the_failed_message_as_the_same_turn_and_keeps_it_out_of_history(monkeypatch, tmp_data):
    sent = script_llm(monkeypatch, RuntimeError("rate limit reached"), "LoRA is small.", "Later answer.")
    at = resend(say(chat_ready(), "What is LoRA?"))
    assert not at.error and not buttons(at, "Resend")
    assert shown(at) == [("user", "What is LoRA?"), ("assistant", "LoRA is small.")]
    assert history_sent(sent[1]) == [("user", "What is LoRA?")]
    assert final_turns(tmp_data) == [1]
    say(at, "Later question")
    assert final_turns(tmp_data) == [1, 2]
    assert history_sent(sent[2]) == [("user", "What is LoRA?"), ("assistant", "LoRA is small."), ("user", "Later question")]


def test_typing_something_else_after_a_failure_replaces_the_failed_message(monkeypatch, tmp_data):
    sent = script_llm(monkeypatch, RuntimeError("rate limit reached"), "Other answer.")
    at = say(say(chat_ready(), "What is LoRA?"), "Something else")
    assert history_sent(sent[1]) == [("user", "Something else")]
    assert shown(at) == [("user", "Something else"), ("assistant", "Other answer.")]
    assert final_turns(tmp_data) == [1]


def test_a_failed_resend_keeps_the_message_and_the_turn_number(monkeypatch, tmp_data):
    script_llm(monkeypatch, RuntimeError("first failure"), RuntimeError("second failure"), "Worked.")
    at = resend(say(chat_ready(), "What is LoRA?"))
    assert "second failure" in at.error[0].value
    assert shown(at) == [("user", "What is LoRA?")]
    at = resend(at)
    assert shown(at)[-1] == ("assistant", "Worked.")
    assert final_turns(tmp_data) == [1]


# the Memory view

def open_view(at, view):
    at.sidebar.radio[0].set_value(view)
    return at.run()


def test_a_sidebar_control_switches_views_and_the_chat_survives_the_trip(monkeypatch):
    sent = script_llm(monkeypatch, "LoRA is small.", "Later answer.")
    at = say(chat_ready(), "What is LoRA?")
    at = open_view(at, "Memory")
    assert not at.chat_input and not shown(at)
    at = open_view(at, "Chat")
    assert at.chat_input
    assert shown(at) == [("user", "What is LoRA?"), ("assistant", "LoRA is small.")]
    say(at, "Later question")
    assert history_sent(sent[1]) == [("user", "What is LoRA?"), ("assistant", "LoRA is small."), ("user", "Later question")]


def remember_one_of_each(student="jun-yu"):
    learner.update_mastery(student, "LoRA", correct=True, wrong_belief="LoRA retrains the whole model")
    learner.add_fact(student, "LoRA", "LoRA adds small matrices", "abc-1")
    learner.add_session_note(student, "Struggled with LoRA rank")


def memory_rows(at):
    return [t.value for t in at.main.text]


def test_the_view_shows_mastery_wrong_beliefs_facts_and_session_notes():
    remember_one_of_each()
    at = open_view(chat_ready(), "Memory")
    assert [s.value for s in at.main.subheader] == ["Mastery scores", "Wrong beliefs", "Facts", "Session notes"]
    assert memory_rows(at) == ["lora: 0.65",  # 0.5 + 0.3 * (1 - 0.5)
                               "lora: LoRA retrains the whole model",
                               "lora: LoRA adds small matrices",
                               "Struggled with LoRA rank"]


def delete_row(at, row_text):
    """Click the delete control beside the row showing this text."""
    row = memory_rows(at).index(row_text)
    at.main.button[row].click()
    return at.run()


def test_each_row_has_a_delete_control_and_a_deleted_row_leaves_the_store_and_the_screen():
    remember_one_of_each()
    at = open_view(chat_ready(), "Memory")
    assert [b.label for b in at.main.button] == ["Delete"] * 4
    expected = {"lora: 0.65": "mastery", "lora: LoRA retrains the whole model": "wrong_beliefs",
                "lora: LoRA adds small matrices": "facts", "Struggled with LoRA rank": "session_notes"}
    for row_text, table in expected.items():
        assert learner.read_all("jun-yu")[table]
        at = delete_row(at, row_text)
        assert learner.read_all("jun-yu")[table] == []
        assert row_text not in memory_rows(at)
    assert not at.main.button


def test_no_control_edits_an_entry():
    remember_one_of_each()
    at = open_view(chat_ready(), "Memory")
    for kind in ("text_input", "text_area", "number_input", "slider", "select_slider", "selectbox", "multiselect",
                 "checkbox", "toggle", "radio", "chat_input"):
        assert not getattr(at.main, kind), kind
    assert {b.label for b in at.main.button} == {"Delete"}


def test_a_deleted_wrong_belief_is_absent_from_the_system_prompt_on_the_next_turn(monkeypatch):
    sent = script_llm(monkeypatch, "First.", "Second.")
    learner.update_mastery("jun-yu", "LoRA", correct=False, wrong_belief="LoRA retrains the whole model")
    learner.update_mastery("jun-yu", "KV cache", correct=False, wrong_belief="KV cache stores weights")
    at = say(chat_ready(), "What is LoRA?")
    assert "LoRA retrains the whole model" in sent[0][0]["content"]
    at = delete_row(open_view(at, "Memory"), "lora: LoRA retrains the whole model")
    say(open_view(at, "Chat"), "And now?")
    assert "LoRA retrains the whole model" not in sent[1][0]["content"]
    assert "KV cache stores weights" in sent[1][0]["content"]  # the prompt still carries the belief that wasn't deleted


def test_another_students_entries_never_appear_and_a_delete_leaves_them_alone():
    remember_one_of_each("bob")  # the same row ids as jun-yu's, as each student has their own database
    learner.update_mastery("bob", "bobs concept", correct=True)
    learner.add_session_note("jun-yu", "Jun Yu's own note")
    at = open_view(chat_ready(), "Memory")
    assert memory_rows(at) == ["Jun Yu's own note"]
    bobs = learner.read_all("bob")
    at = delete_row(at, "Jun Yu's own note")
    assert memory_rows(at) == []
    assert learner.read_all("bob") == bobs


def test_the_view_is_reachable_with_no_files_because_memory_can_outlive_them():
    remember_one_of_each()
    at = open_view(enter_name(open_app(), "Jun Yu"), "Memory")
    assert "Struggled with LoRA rank" in memory_rows(at)
    assert not at.info  # the prompt to upload locks chat, not this view


def test_a_delete_that_fails_says_so_and_the_row_stays(monkeypatch):
    remember_one_of_each()
    at = open_view(chat_ready(), "Memory")

    def locked(*args):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(learner, "delete", locked)
    at = delete_row(at, "Struggled with LoRA rank")
    assert not at.exception
    assert "database is locked" in at.error[0].value
    assert "Struggled with LoRA rank" in memory_rows(at)


def test_start_quiz_is_unavailable_in_the_memory_view_where_it_could_not_show_a_reply():
    learner.add_concepts("jun-yu", "notes.pdf", ["LoRA"])
    at = chat_ready()
    assert not buttons(at, "Start quiz")[0].disabled
    assert buttons(open_view(at, "Memory"), "Start quiz")[0].disabled


# running it

def test_the_readme_documents_the_launch_command_and_streamlit_is_pinned():
    assert "streamlit run app.py" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"^streamlit==\d+\.\d+\.\d+$", (ROOT / "requirements.txt").read_text(), re.M)
