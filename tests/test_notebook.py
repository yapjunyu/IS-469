"""The notebook module: real Chroma in a temp folder, a generated PDF, fake models and a scripted LLM."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import chromadb
import numpy as np
import pymupdf
import pytest
from pptx import Presentation

from agent import s1_llm
from agent.memory import learner
from agent.tests.test_loop import SEARCH_CALL, reply, turn
from tutor import notebook
from tutor.retrieval import embed, search
from tutor.retrieval.extract import FileRejected

WORDS = ["lora", "bm25", "agent"]
CONCEPTS = reply({"role": "assistant", "content": '["LoRA", "Adapters"]'})
PICTURE = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), False)


def _meaning(text):
    return text.lower().replace("low-rank", "lora")  # the fakes' idea of two words meaning the same


class FakeSentenceTransformer:
    """Fake embedding model: one dimension per keyword, so a text points towards the keywords it mentions."""
    loads = 0

    def __init__(self, *args, **kwargs):
        type(self).loads += 1

    def encode(self, inputs, **kwargs):
        # a picture has no words, so it points equally at every keyword
        return np.array([[_meaning(i).count(w) + 0.01 if isinstance(i, str) else 1.0 for w in WORDS] for i in inputs])


class FakeCrossEncoder:
    """Fake reranker: a passage scores one point per question word it contains."""
    loads = 0

    def __init__(self, *args, **kwargs):
        type(self).loads += 1

    def predict(self, pairs):
        return np.array([sum(w in _meaning(text) for w in _meaning(query).split()) for query, text in pairs], dtype=float)


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    """Replaces the model library itself, so the real Embedder, HybridRerankIndex and their caches all run."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer, CrossEncoder=FakeCrossEncoder))
    monkeypatch.setattr(FakeSentenceTransformer, "loads", 0)
    monkeypatch.setattr(FakeCrossEncoder, "loads", 0)
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: CONCEPTS)
    embed._load.cache_clear()
    search._cross_encoder.cache_clear()
    yield
    embed._load.cache_clear()  # so no fake model outlives this test
    search._cross_encoder.cache_clear()


def make_pdf(path, *slides):
    """A landscape PDF, which extract treats as slides: one passage per page, with these texts."""
    with pymupdf.open() as doc:
        for text in slides:
            doc.new_page(width=792, height=612).insert_text((72, 72), text)
        doc.save(path)
    return path


# student_id

@pytest.mark.parametrize("name", ["Jun Yu", "jun yu", "  Jun   Yu ", "JUN\tYU"])
def test_capitals_and_spacing_reach_the_same_notebook(name):
    assert notebook.student_id(name) == "jun-yu"


@pytest.mark.parametrize("name", ["Jun!", "../bob", "a/b", "Zoë", "", "   ", "x" * 65])
def test_a_name_that_cannot_be_used_raises_a_clear_error(name):
    with pytest.raises(ValueError, match="name"):
        notebook.student_id(name)


# add_file and search

def test_added_file_is_reported_and_its_concepts_join_the_concept_list(tmp_path):
    out =notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    assert out == {"status": "added", "file": "notes.pdf", "concepts_found": True}
    assert learner.concepts("alice") == ["adapters", "lora"]


def test_search_returns_passages_with_their_source_and_no_picture(tmp_path):
    path = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small")
    notebook.add_file("alice", path)
    [found] = notebook.search("alice", "What is LoRA?")
    assert found == {"id": f"{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}:0000", "file": "notes.pdf",
                     "page": 1, "page_end": 1, "kind": "slide", "text": "LoRA adapters are small", "image_path": None}


def test_search_returns_the_top_3(tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA one", "LoRA two", "LoRA three", "LoRA four"))
    assert len(notebook.search("alice", "LoRA")) == 3


def test_a_figure_comes_back_with_its_saved_picture(tmp_data):
    path = tmp_data / "paper.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()  # portrait, so a document page: its figure is cropped to its own PNG
        page.insert_text((72, 72), "LoRA adapters, see the figure")
        page.insert_image(pymupdf.Rect(72, 100, 272, 300), pixmap=PICTURE)
        doc.save(path)
    notebook.add_file("alice", path)
    [picture] = [p for p in notebook.search("alice", "LoRA") if p["image_path"]]
    assert picture["text"] == "" and picture["page"] == 1
    assert Path(picture["image_path"]).is_relative_to(tmp_data / "images") and Path(picture["image_path"]).exists()


def test_a_file_added_for_alice_never_reaches_bob(tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    notebook.add_file("bob", make_pdf(tmp_path / "other.pdf", "BM25 ranks by keyword"))
    alices = notebook.search("alice", "LoRA")
    assert {p["file"] for p in alices} == {"notes.pdf"}
    assert {p["file"] for p in notebook.search("bob", "LoRA")} == {"other.pdf"}
    assert notebook.get_passages("bob", [p["id"] for p in alices]) == []
    assert notebook.files("bob") == ["other.pdf"]


def test_a_student_with_no_files_finds_nothing_and_loads_no_models():
    assert notebook.search("alice", "LoRA") == []
    assert notebook.get_passages("alice", ["abc:0000"]) == []
    assert notebook.files("alice") == []
    assert FakeSentenceTransformer.loads == 0 and FakeCrossEncoder.loads == 0


def test_a_second_file_is_found_at_once_by_meaning(tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "first.pdf", "BM25 ranks by keyword"))
    notebook.search("alice", "BM25")  # the notebook has been searched before the second file arrives
    notebook.add_file("alice", make_pdf(tmp_path / "second.pdf", "LoRA adapters are small"))
    assert notebook.search("alice", "low-rank")[0]["file"] == "second.pdf"  # "low-rank" is nowhere in its text


def test_a_second_file_is_found_at_once_by_keyword(tmp_path):
    # 25 balanced slides fill meaning search's 20 candidates, since the question points the same way they do.
    # The second file's slide points away, so only keyword search can bring it to the reranker.
    slides = [f"LoRA BM25 agent overview part {n}" for n in range(1, 26)]
    notebook.add_file("alice", make_pdf(tmp_path / "first.pdf", *slides))
    notebook.search("alice", "zyxel")
    notebook.add_file("alice", make_pdf(tmp_path / "second.pdf", "LoRA zyxel"))
    assert notebook.search("alice", "zyxel")[0]["file"] == "second.pdf"


def test_get_passages_returns_those_ids_in_the_same_shape(tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small", "BM25 ranks by keyword"))
    lora, bm25 = notebook.search("alice", "LoRA")
    assert notebook.get_passages("alice", [bm25["id"], "nonsense:0000"]) == [bm25]
    assert notebook.get_passages("alice", [bm25["id"], lora["id"]]) == [bm25, lora]


def test_files_lists_the_notebooks_files_by_base_name(tmp_data, tmp_path):
    (tmp_path / "week1").mkdir()
    notebook.add_file("alice", make_pdf(tmp_path / "week1" / "b.pdf", "LoRA adapters are small"))
    notebook.add_file("alice", make_pdf(tmp_path / "a.pdf", "BM25 ranks by keyword"))
    assert notebook.files("alice") == ["a.pdf", "b.pdf"]
    assert (tmp_data / "uploads" / "alice" / "b.pdf").read_bytes() == (tmp_path / "week1" / "b.pdf").read_bytes()


# rejected files

def scanned_pdf(path):
    with pymupdf.open() as doc:
        doc.new_page(width=792, height=612).insert_image(pymupdf.Rect(0, 0, 792, 612), pixmap=PICTURE)
        doc.new_page().insert_image(pymupdf.Rect(0, 0, 400, 400), pixmap=PICTURE)
        doc.save(path)


def blank_pdf(path):
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path)


def page_numbers_only_pdf(path):
    make_pdf(path, "1", "2")  # cleaning removes these, which leaves no text


def blank_pptx(path):
    deck = Presentation()
    deck.slides.add_slide(deck.slide_layouts[6])
    deck.save(path)


def corrupt_pdf(path):
    path.write_text("not really a PDF")


def unsupported_type(path):
    path.write_text("LoRA adapters are small")


@pytest.mark.parametrize("build, name", [(scanned_pdf, "bad.pdf"), (blank_pdf, "bad.pdf"), (page_numbers_only_pdf, "bad.pdf"),
                                         (blank_pptx, "bad.pptx"), (corrupt_pdf, "bad.pdf"), (unsupported_type, "bad.txt")])
def test_a_rejected_file_raises_and_stores_nothing(build, name, monkeypatch, tmp_data, tmp_path):
    build(tmp_path / name)
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: pytest.fail("a rejected file needs no concepts"))
    with pytest.raises(FileRejected, match="bad"):
        notebook.add_file("alice", tmp_path / name)
    assert notebook.files("alice") == []
    assert learner.concepts("alice") == []
    assert not list((tmp_data / "uploads").rglob("*")) and not list((tmp_data / "images").rglob("*"))
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: CONCEPTS)
    notebook.add_file("alice", make_pdf(tmp_path / "good.pdf", "LoRA one", "LoRA two", "LoRA three", "LoRA four"))
    assert {p["file"] for p in notebook.search("alice", "LoRA")} == {"good.pdf"}  # no passages of the rejected file


def test_a_file_that_fails_to_index_is_not_listed(monkeypatch, tmp_path):
    def fail(self, inputs, **kwargs):
        raise RuntimeError("out of memory")
    with monkeypatch.context() as broken:
        broken.setattr(FakeSentenceTransformer, "encode", fail)
        with pytest.raises(RuntimeError):
            notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    assert notebook.files("alice") == []
    assert notebook.add_file("alice", tmp_path / "notes.pdf")["status"] == "added"


# concept extraction failing

def _rate_limited(*a, **k):
    raise RuntimeError("rate limited")


@pytest.mark.parametrize("failure", [_rate_limited, lambda *a, **k: reply({"role": "assistant", "content": "I found no list"})])
def test_if_concept_extraction_fails_the_file_stays_searchable_and_the_result_says_so(failure, monkeypatch, tmp_path):
    monkeypatch.setattr(s1_llm, "chat", failure)
    out = notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    assert out == {"status": "added", "file": "notes.pdf", "concepts_found": False}
    assert learner.concepts("alice") == []
    assert notebook.files("alice") == ["notes.pdf"]
    assert notebook.search("alice", "LoRA")[0]["text"] == "LoRA adapters are small"


# adding the same file again

def test_identical_content_is_recognised_and_adds_no_passages(monkeypatch, tmp_path):
    path = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small")
    notebook.add_file("alice", path)
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: pytest.fail("nothing new to read"))
    assert notebook.add_file("alice", path) == {"status": "already_held", "file": "notes.pdf", "concepts_found": None}
    renamed = tmp_path / "renamed.pdf"
    renamed.write_bytes(path.read_bytes())
    assert notebook.add_file("alice", renamed) == {"status": "already_held", "file": "notes.pdf", "concepts_found": None}
    assert notebook.files("alice") == ["notes.pdf"]
    assert len(notebook.search("alice", "LoRA")) == 1


def test_adding_the_saved_copy_itself_is_harmless(tmp_data, tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    saved = tmp_data / "uploads" / "alice" / "notes.pdf"
    assert notebook.add_file("alice", saved)["status"] == "already_held"
    assert saved.read_bytes() == (tmp_path / "notes.pdf").read_bytes()


def test_a_saved_copy_not_yet_indexed_gets_indexed(tmp_data):
    saved = tmp_data / "uploads" / "alice" / "notes.pdf"
    saved.parent.mkdir(parents=True)
    make_pdf(saved, "LoRA adapters are small")
    assert notebook.add_file("alice", saved)["status"] == "added"
    assert notebook.search("alice", "LoRA")[0]["text"] == "LoRA adapters are small"


def test_same_name_with_new_content_is_turned_away_until_replacing_exists(tmp_data, tmp_path):
    old = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small").read_bytes()
    notebook.add_file("alice", tmp_path / "notes.pdf")
    (tmp_path / "new").mkdir()
    with pytest.raises(FileRejected, match="notes.pdf"):
        notebook.add_file("alice", make_pdf(tmp_path / "new" / "notes.pdf", "BM25 ranks by keyword"))
    assert [p["text"] for p in notebook.search("alice", "BM25")] == ["LoRA adapters are small"]
    assert (tmp_data / "uploads" / "alice" / "notes.pdf").read_bytes() == old


# student IDs reaching storage

@pytest.mark.parametrize("name", ["a", "-a-", "_", "x" * 64])
def test_any_id_student_id_allows_gets_a_working_store(name, tmp_path):
    notebook.add_file(name, make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    assert notebook.search(name, "LoRA")[0]["file"] == "notes.pdf"
    assert learner.concepts(name) == ["adapters", "lora"]


def test_an_id_that_could_reach_another_notebook_is_refused(tmp_data, tmp_path):
    path = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small")
    with pytest.raises(ValueError, match="name"):
        notebook.add_file("../bob", path)
    with pytest.raises(ValueError, match="name"):
        notebook.files("../bob")
    assert not list(tmp_data.glob("uploads*"))


def test_each_student_has_one_collection_in_their_own_folder_named_apart_from_the_shared_corpus(tmp_data, tmp_path):
    # one collection per folder because two in one folder made Chroma 1.5.9 fail about 1 search in 100
    for student in ("alice", "bob"):
        notebook.add_file(student, make_pdf(tmp_path / f"{student}.pdf", "LoRA adapters are small"))
        client = chromadb.PersistentClient(path=str(tmp_data / "chroma" / student))
        assert [c.name for c in client.list_collections()] == [f"slide-recursive200-images-{student}_gemma2"]


# models

def test_the_embedder_and_reranker_load_once_however_much_is_added_and_searched(tmp_path):
    for student, name in [("alice", "a.pdf"), ("alice", "b.pdf"), ("bob", "c.pdf")]:
        notebook.add_file(student, make_pdf(tmp_path / name, f"LoRA adapters in {name}"))
        notebook.search(student, "LoRA")
        notebook.get_passages(student, ["x:0000"])
    assert FakeSentenceTransformer.loads == 1
    assert FakeCrossEncoder.loads == 1


# run_turn

def test_a_turn_through_the_notebook_gives_the_llm_a_real_passage(monkeypatch, tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    replies = iter([reply(SEARCH_CALL), reply({"role": "assistant", "content": "LoRA adapters are small."}),
                    reply({"role": "assistant", "content": '{"supported": true, "reason": "stated"}'})])
    sent = []

    def chat(messages, model, **kw):
        sent.append(json.dumps(messages))
        return next(replies)

    monkeypatch.setattr(s1_llm, "chat", chat)
    out = turn("What is LoRA?", "alice", 1, search_fn=notebook.search, passages_fn=notebook.get_passages, faithfulness=True)
    assert out["citations"] == [{"file": "notes.pdf", "page": 1}]
    assert "LoRA adapters are small" in sent[1]  # the tool message, from search
    assert "LoRA adapters are small" in sent[2]  # the judge's passages, from get_passages
    assert out["warning"] is False


# no Streamlit

def test_the_module_imports_without_streamlit_and_with_only_src_on_the_path():
    code = "import sys; sys.modules['streamlit'] = None; import tutor.notebook"  # None makes any import of it fail
    subprocess.run([sys.executable, "-c", code], cwd=Path(notebook.__file__).parents[1],
                   env={**os.environ, "PYTHONPATH": ""}, check=True)
