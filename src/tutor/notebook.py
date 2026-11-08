"""One student's notebook: the only way the app and other scripts reach retrieval. No Streamlit in here.

    student = student_id("Jun Yu")          # 'jun-yu'
    add_file(student, "Lecture3.pdf")       # {"status": "added", "file": "Lecture3.pdf", "concepts_found": True}
                                            # status "replaced": same name, new content; "already_held": same content
    files(student)                          # ['Lecture3.pdf']
    remove_file(student, "Lecture3.pdf")    # its passages, concepts, facts, saved copy and pictures go; mastery scores stay
    search(student, "What is LoRA?")        # passages: id, file, page, page_end, kind, text, image_path
    get_passages(student, ids)              # the passages with those ids, same shape

search and get_passages are what run_turn takes as search_fn and passages_fn.
"""
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# shortcut: puts the repo root on the path because the agent package sits beside src, drop it once both are installed as packages
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ingest import extract_concepts  # noqa: E402
from agent.memory import learner  # noqa: E402
from tutor.pipeline import EMBEDDER, STORE_TAG, search_index, store_pages  # noqa: E402
from tutor.generator import K  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import FileRejected, extract, file_sha256  # noqa: E402
import tutor.retrieval.extract as extraction  # noqa: E402  (its IMAGE_DIR; `extract` here is the function)
from tutor.retrieval.search import ChromaIndex  # noqa: E402

# shortcut: hard-coded paths, move to the team's config file once it exists
UPLOADS_DIR = ROOT / "data" / "uploads"
CHROMA_DIR = ROOT / "data" / "notebook_chroma"
# shortcut: hard-coded, move to the team's config file once it exists
MAX_ID_CHARS = 64  # far above a real name, and keeps the file, folder and Chroma names built from it within their limits


def student_id(name: str) -> str:
    """'  Jun   Yu ' -> 'jun-yu'. Raises ValueError, with a message to show the student, on a name that can't be used."""
    student = "-".join(name.lower().split())
    if not learner.STUDENT_ID.fullmatch(student):
        raise ValueError(f"The name {name!r} can't be used: use only letters, numbers, spaces, - and _.")
    if len(student) > MAX_ID_CHARS:
        raise ValueError(f"That name is too long: use at most {MAX_ID_CHARS} characters.")
    return student


def _store(student: str) -> ChromaIndex:
    # STORE_TAG in front and ChromaIndex's embedder suffix behind make any student ID a valid Chroma name, and one the shared corpus can't have.
    # A folder each: with two collections in one folder, Chroma 1.5.9 fails about 1 in 100 searches ("Nothing found on disk") until the next write.
    return ChromaIndex(Embedder(EMBEDDER), tag=f"{STORE_TAG}-{student}", path=CHROMA_DIR / student)


def add_file(student: str, path) -> dict:
    """Make a file searchable in this student's notebook and list its concepts.

    New content under a name already in the notebook replaces that file, as remove_file would remove it.
    Returns {"status": "added", "replaced" or "already_held", "file": its name in the notebook, "concepts_found": True
    or False, or None when the file was already held, as nothing new was read}.
    Raises FileRejected, with a message to show the student, and then changes nothing.
    """
    student, path = student_id(student), Path(path)
    # images=False saves no pictures, so a file rejected here leaves none behind
    if not clean_pages(extract(path, images=False)):
        raise FileRejected(f"{path.name} has no text to read. If it is a scanned PDF, upload a version with selectable text.")
    digest, folder = file_sha256(path), UPLOADS_DIR / student
    held = next((name for name in files(student) if file_sha256(folder / name) == digest), None)
    if held:
        return {"status": "already_held", "file": held, "concepts_found": None}
    # any capitals, as Windows sees Notes.pdf and notes.pdf as one file
    replacing = next((name for name in files(student) if name.casefold() == path.name.casefold()), None)

    pages = clean_pages(extract(path, images=True))
    store_pages(_store(student), pages, digest)  # stores nothing if a crash before the save below already stored this file
    if replacing:  # only now, so a file that can't be indexed leaves the old one searchable
        remove_file(student, replacing)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, folder / path.name)  # saved last, so a saved file is always searchable
    try:
        names = extract_concepts(student, path.name, "\n\n".join(p.text for p in pages))
    except Exception:  # wraps a provider call that can fail in many ways, and the file is already searchable
        # shortcut: re-adding can't retry a failed extraction because the file is now held, only removing then re-adding does, add a retry control if students need one
        names = []
    return {"status": "replaced" if replacing else "added", "file": path.name, "concepts_found": bool(names)}


def remove_file(student: str, filename: str) -> None:
    """Remove a file and everything from it: passages, concepts, facts, the saved copy and its pictures.

    Mastery scores stay, as the panel only lists current concepts and they return if the file is added again.
    A name not in the notebook changes nothing.
    """
    student = student_id(student)
    if filename not in files(student):  # this check is what keeps a name from a UI control inside the student's folder
        return
    saved = UPLOADS_DIR / student / filename
    digest = file_sha256(saved)
    _store(student).remove_file(digest)
    learner.remove_file(student, filename)
    for fact in learner.read_all(student)["facts"]:
        if fact["passage_id"].startswith(digest[:12]):
            learner.delete(student, "facts", fact["id"])
    # pictures are stored per file fingerprint and shared, so they stay while any other saved file has this content
    # shortcut: hashes every student's saved files on each removal, keep a fingerprint per file if that gets slow
    pictures = extraction.IMAGE_DIR / digest[:12]
    if pictures.exists() and not any(p != saved and file_sha256(p) == digest for p in UPLOADS_DIR.glob("*/*")):
        shutil.rmtree(pictures)
    saved.unlink()  # last, so after a failure part-way the file is still listed and removing it again finishes the job


def files(student: str) -> list[str]:
    """The names of the files in this student's notebook."""
    return sorted(p.name for p in (UPLOADS_DIR / student_id(student)).glob("*"))


def search(student: str, query: str) -> list[dict]:
    """The top K passages for the query, from this student's files only."""
    student = student_id(student)
    if not files(student):  # before _store, so an empty notebook loads no models
        return []
    # shortcut: rebuilds the index on every search so it always covers the current files, keep one per student if a big notebook makes this slow
    return [_passage(p) for p in search_index(_store(student)).search(query, K)]


def get_passages(student: str, ids: list[str]) -> list[dict]:
    """The passages with these IDs, in the order asked. An ID not in this notebook is left out.
    A slide comes back with its text and PNG merged, the same as search returns it."""
    student = student_id(student)
    if not files(student):
        return []
    chunks = _store(student).chunks
    held = {c.id: c for c in chunks}
    slide_text = {(c.file, c.page): c.text for c in chunks if c.kind == "slide" and c.text}
    slide_png = {(c.file, c.page): c.image_path for c in chunks if c.kind == "slide" and c.image_path}

    found = []
    for i in ids:
        if i not in held:
            continue
        c = held[i]
        if c.kind == "slide":  # fill in the other half, like SlideMergeIndex does
            key = (c.file, c.page)
            c = replace(c, text=c.text or slide_text.get(key, ""), image_path=c.image_path or slide_png.get(key))
        found.append(_passage(c))
    return found


def _passage(passage) -> dict:
    return {k: getattr(passage, k) for k in ("id", "file", "page", "page_end", "kind", "text", "image_path")}
