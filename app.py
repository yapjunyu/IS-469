"""The Study Tutor app: enter a name, upload files, chat with the tutor. Start it from the repo root with `streamlit run app.py`."""
import sys
import tempfile
from pathlib import Path

import streamlit as st

SRC = str(Path(__file__).resolve().parent / "src")
# shortcut: puts src on the path because the tutor package isn't installed, drop it once it is
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from agent.s3_loop import run_turn  # noqa: E402
from tutor import notebook  # noqa: E402
from tutor.retrieval.extract import FileRejected  # noqa: E402

WARNING = "Warning: this answer may not be supported by your files, so don't rely on it."


def index_upload(student, upload):
    """Add one uploaded file to the notebook. Returns the st function to show the outcome with, its text,
    and the notebook file that text is about, or None when it's about the upload alone."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(upload.name).name  # add_file keeps this name, so only the base name
        path.write_bytes(upload.getvalue())
        try:
            added = notebook.add_file(student, path)
        except FileRejected as e:
            return st.error, str(e), None
        except Exception as e:  # the embedder or store failing: cached like any outcome, so a rerun doesn't retry and die before chat
            return st.error, f"{upload.name} couldn't be indexed: {e}. Remove it from the uploader and add it again to retry.", None
    file = added["file"]
    if added["status"] == "already_held":
        return st.info, f"{upload.name} has the same content as {file}, already in your notebook.", file
    if not added["concepts_found"]:
        return st.warning, f"{file} {added['status']}, but its concepts couldn't be listed. Removing and re-uploading it retries.", file
    if added["status"] == "replaced":
        return st.success, f"{file} replaced with the new version.", file
    return st.success, f"{file} added to your notebook.", file


state = st.session_state

st.title("Study Tutor")
st.caption("The tutor is only as accurate as your files, so check anything important against them.")

if "student" not in state:
    with st.sidebar.form("name"):
        name = st.text_input("Your name")
        entered = st.form_submit_button("Open my notebook")
    if entered:
        try:
            state.student = notebook.student_id(name)
        except ValueError as e:
            st.sidebar.error(str(e))
        else:
            st.rerun()
    st.info("Enter your name in the sidebar to open your notebook.")
    st.stop()

student = state.student
# file_id -> outcome, as Streamlit hands back every uploaded file on every rerun. None once its file is removed: the entry
# stays, so an upload still in the uploader isn't indexed again, but says nothing. Uploading it again gets a new file_id.
state.setdefault("upload_outcomes", {})

with st.sidebar:
    st.write(f"Notebook: {student}")
    for upload in st.file_uploader("Add .pptx or PDF files", type=["pptx", "pdf"], accept_multiple_files=True):
        if upload.file_id not in state.upload_outcomes:
            with st.spinner(f"Indexing {upload.name}...", show_time=True):
                state.upload_outcomes[upload.file_id] = index_upload(student, upload)
        if outcome := state.upload_outcomes[upload.file_id]:
            show, text, _ = outcome
            show(text)
    st.subheader("Your files")
    for file in notebook.files(student):
        st.text(file)
        if st.button(f"Remove {file}"):
            try:
                notebook.remove_file(student, file)
            except Exception as e:  # the store or a locked file failing: it stays listed, and removing it again finishes the job
                st.error(f"{file} couldn't be removed: {e}")
            else:
                state.upload_outcomes = {i: None if o and o[2] == file else o for i, o in state.upload_outcomes.items()}
                st.rerun()

if not notebook.files(student):
    st.info("Upload a .pptx or PDF in the sidebar to start chatting.")
    st.stop()

state.setdefault("messages", [])  # the messages shown, each {"role", "content"}, and "warning" on an answer
state.setdefault("turn", 1)
state.setdefault("failed", None)  # the turn that failed last, {"message", "error"}; neither messages nor turn counted it

for entry in state.messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("warning"):
            st.warning(WARNING)

# chat_input pins itself to the bottom of the page, so calling it first only means a new message is known before drawing
message = st.chat_input("Ask about your notes", submit_mode="disable") or state.pop("resend", None)
if message:
    state.failed = None
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"), st.spinner("The tutor is working...", show_time=True):
        history = [{"role": entry["role"], "content": entry["content"]} for entry in state.messages]
        try:
            out = run_turn(message, student, state.turn, history=history, search_fn=notebook.search,
                           passages_fn=notebook.get_passages, memory="learner")
        except Exception as e:  # a turn can fail in many ways (rate limit, missing key), and the student can resend
            state.failed = {"message": message, "error": str(e)}
        else:
            state.messages += [{"role": "user", "content": message},
                               {"role": "assistant", "content": out["answer"], "warning": out["warning"]}]
            state.turn += 1
    st.rerun()

if state.failed:
    with st.chat_message("user"):
        st.markdown(state.failed["message"])
    st.error(f"The tutor couldn't answer: {state.failed['error']}")
    if st.button("Resend"):
        state.resend = state.failed["message"]
        st.rerun()
