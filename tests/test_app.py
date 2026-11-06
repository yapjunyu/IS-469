"""The Streamlit app through AppTest: the LLM is scripted, the notebook's models are fakes, every file goes to a temp folder."""
import json
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from agent import s1_llm
from agent.memory import learner
from agent.tests.test_loop import SEARCH_CALL, read_trace, reply
from test_notebook import fake_models, make_pdf  # noqa: F401  (fake_models is autouse: importing it keeps real models out of this file)
from tutor import notebook

ROOT = Path(__file__).resolve().parents[1]


def open_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()


def enter_name(at, name):
    at.text_input[0].input(name)
    at.button[0].click()
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

def upload(at, tmp_path, *slides, name="notes.pdf"):
    content = make_pdf(tmp_path / name, *slides).read_bytes()
    at.file_uploader[0].set_value((name, content, "application/pdf"))
    return at.run()


def test_an_uploaded_file_joins_the_list_and_unlocks_chat(tmp_path):
    at = enter_name(open_app(), "Jun Yu")
    assert not at.chat_input
    at = upload(at, tmp_path, "LoRA adapters are small")
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


def test_a_rerun_does_not_index_the_same_file_again_and_keeps_its_message(monkeypatch, tmp_path):
    calls = []
    real = notebook.add_file
    monkeypatch.setattr(notebook, "add_file", lambda *a: calls.append(a) or real(*a))
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path, "LoRA adapters are small")
    at.run()
    assert len(calls) == 1
    assert "notes.pdf" in at.sidebar.success[0].value


def test_a_copy_under_another_name_names_the_file_already_holding_it(tmp_path):
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path, "LoRA adapters are small")
    content = (tmp_path / "notes.pdf").read_bytes()
    at.file_uploader[0].set_value(("copy.pdf", content, "application/pdf"))
    at.run()
    assert "notes.pdf" in at.sidebar.info[0].value
    assert files_listed(at) == ["notes.pdf"]


# the chat screen

def test_with_no_files_chat_is_locked_with_a_prompt_to_upload():
    at = enter_name(open_app(), "Jun Yu")
    assert not at.chat_input
    assert "upload" in at.info[0].value.lower()


def test_a_notice_says_the_tutor_is_only_as_accurate_as_the_files():
    assert any("only as accurate as your files" in c.value for c in open_app().caption)
    seed_file()
    assert any("only as accurate as your files" in c.value for c in enter_name(open_app(), "Jun Yu").caption)


def test_no_screen_offers_a_switch():
    def switches(at):
        return [*at.checkbox, *at.toggle, *at.radio, *at.selectbox, *at.multiselect]
    at = open_app()
    assert not switches(at)
    seed_file()
    assert not switches(enter_name(at, "Jun Yu"))


# a turn

def script_llm(monkeypatch, *tutor_replies, supported=True):
    """Fake s1_llm.chat. Calls with tools are the tutor's: each gets the next reply, a plain string as the answer,
    an Exception raised. Calls without are the checks: enough passages, and `supported` from the judge.
    Returns the messages each tutor call was sent."""
    replies, sent = iter(tutor_replies), []

    def chat(messages, model, tools=None, **kwargs):
        if not tools:
            return reply({"role": "assistant", "content": json.dumps({"enough": True, "supported": supported, "reason": "r"})})
        sent.append(list(messages))
        next_reply = next(replies)
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
    at = upload(enter_name(open_app(), "Jun Yu"), tmp_path, "LoRA adapters are small")
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


def test_a_warning_stays_on_its_own_answer_as_the_chat_goes_on(monkeypatch):
    script_llm(monkeypatch, "Unsupported claim.", "Still unsupported.", supported=False)
    at = say(chat_ready(), "What is LoRA?")
    script_llm(monkeypatch, "Fine answer.")
    at = say(at, "And BM25?")
    assert [len(m.warning) for m in at.chat_message] == [0, 1, 0, 0]


# a failed turn

def test_a_failed_turn_shows_the_error_and_keeps_the_message_with_a_resend_button(monkeypatch):
    script_llm(monkeypatch, RuntimeError("rate limit reached"))
    at = say(chat_ready(), "What is LoRA?")
    assert "rate limit reached" in at.error[0].value
    assert shown(at) == [("user", "What is LoRA?")]
    assert [b.label for b in at.button] == ["Resend"]


def test_resending_runs_the_failed_message_as_the_same_turn_and_keeps_it_out_of_history(monkeypatch, tmp_data):
    sent = script_llm(monkeypatch, RuntimeError("rate limit reached"), "LoRA is small.", "Later answer.")
    at = say(chat_ready(), "What is LoRA?")
    at.button[0].click()
    at.run()
    assert not at.error and not at.button
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
    at = say(chat_ready(), "What is LoRA?")
    at.button[0].click()
    at.run()
    assert "second failure" in at.error[0].value
    assert shown(at) == [("user", "What is LoRA?")]
    at.button[0].click()
    at.run()
    assert shown(at)[-1] == ("assistant", "Worked.")
    assert final_turns(tmp_data) == [1]


# running it

def test_the_readme_documents_the_launch_command_and_streamlit_is_pinned():
    assert "streamlit run app.py" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"^streamlit==\d+\.\d+\.\d+$", (ROOT / "requirements.txt").read_text(), re.M)
