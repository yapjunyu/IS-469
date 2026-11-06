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
    """Add one uploaded file to the notebook. Returns the st function to show the outcome with, and its text."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(upload.name).name  # add_file keeps this name, so only the base name
        path.write_bytes(upload.getvalue())
        try:
            added = notebook.add_file(student, path)
        except FileRejected as e:
            return "error", str(e)
    if added["status"] == "already_held":
        return "info", f"{upload.name} has the same content as {added['file']}, already in your notebook."
    return "success", f"{added['file']} added to your notebook."


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
state.setdefault("uploads", {})  # file_id -> outcome, as Streamlit hands back every uploaded file on every rerun

with st.sidebar:
    st.write(f"Notebook: {student}")
    for upload in st.file_uploader("Add .pptx or PDF files", type=["pptx", "pdf"], accept_multiple_files=True):
        if upload.file_id not in state.uploads:
            with st.spinner(f"Indexing {upload.name}...", show_time=True):
                state.uploads[upload.file_id] = index_upload(student, upload)
        show, text = state.uploads[upload.file_id]
        getattr(st, show)(text)
    st.subheader("Your files")
    for file in notebook.files(student):
        st.text(file)

if not notebook.files(student):
    st.info("Upload a .pptx or PDF in the sidebar to start chatting.")
    st.stop()

state.setdefault("chat", [])  # the messages shown, each {"role", "content"}, and "warning" on an answer
state.setdefault("turn", 1)
state.setdefault("failed", None)  # the turn that failed last, {"message", "error"}; neither chat nor turn counted it

for entry in state.chat:
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
        history = [{"role": entry["role"], "content": entry["content"]} for entry in state.chat]
        try:
            out = run_turn(message, student, state.turn, history=history, search_fn=notebook.search,
                           passages_fn=notebook.get_passages, memory="learner")
        except Exception as e:  # a turn can fail in many ways (rate limit, missing key), and the student can resend
            state.failed = {"message": message, "error": str(e)}
        else:
            state.chat += [{"role": "user", "content": message},
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
