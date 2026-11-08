"""The notebook module: real Chroma in a temp folder, a generated PDF, fake models and a scripted LLM."""
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
from tutor.pipeline import EMBEDDER, STORE_TAG
from tutor.retrieval import embed, extract, search
from tutor.retrieval.extract import FileRejected, file_sha256

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


def test_search_returns_passages_with_their_file_page_and_text(tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    [found] = notebook.search("alice", "What is LoRA?")
    assert found.pop("id")
    assert found == {"file": "notes.pdf", "page": 1, "page_end": 1, "kind": "slide",
                     "text": "LoRA adapters are small", "image_path": None}


def test_search_returns_the_top_3(tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA one", "LoRA two", "LoRA three", "LoRA four"))
    assert len(notebook.search("alice", "LoRA")) == 3


def test_a_figure_comes_back_with_its_saved_picture(tmp_data):
    notebook.add_file("alice", figure_pdf(tmp_data / "paper.pdf", "LoRA adapters, see the figure"))
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


def test_a_crash_between_indexing_and_saving_is_put_right_by_adding_again(tmp_data, tmp_path):
    path = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small")
    notebook.add_file("alice", path)
    (tmp_data / "uploads" / "alice" / "notes.pdf").unlink()  # passages stored, file not saved: what the crash leaves
    assert notebook.add_file("alice", path)["status"] == "added"
    assert notebook.files("alice") == ["notes.pdf"]
    assert len(notebook.search("alice", "LoRA")) == 1


# removing a file

def concepts_reply(*names):
    return lambda *a, **k: reply({"role": "assistant", "content": json.dumps(names)})


def figure_pdf(path, text):
    """A portrait PDF, so a document page: its figure is cropped to its own saved picture."""
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        page.insert_image(pymupdf.Rect(72, 100, 272, 300), pixmap=PICTURE)
        doc.save(path)
    return path


def saved_pictures(path):
    """The pictures saved for the file at this path, which live in a folder named after its fingerprint."""
    return sorted((extract.IMAGE_DIR / file_sha256(path)[:12]).glob("*"))


def test_a_removed_file_is_gone_from_the_list_search_and_get_passages_and_its_saved_copy_is_deleted(tmp_data, tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "gone.pdf", "LoRA adapters are small"))
    notebook.add_file("alice", make_pdf(tmp_path / "kept.pdf", "BM25 ranks by keyword"))
    gone = next(p for p in notebook.search("alice", "LoRA") if p["file"] == "gone.pdf")
    notebook.remove_file("alice", "gone.pdf")
    assert notebook.files("alice") == ["kept.pdf"]
    assert {p["file"] for p in notebook.search("alice", "LoRA")} == {"kept.pdf"}
    assert notebook.get_passages("alice", [gone["id"]]) == []
    assert not (tmp_data / "uploads" / "alice" / "gone.pdf").exists()


def test_removing_a_file_drops_concepts_only_it_had(monkeypatch, tmp_path):
    monkeypatch.setattr(s1_llm, "chat", concepts_reply("LoRA", "Shared"))
    notebook.add_file("alice", make_pdf(tmp_path / "gone.pdf", "LoRA adapters are small"))
    monkeypatch.setattr(s1_llm, "chat", concepts_reply("BM25", "Shared"))
    notebook.add_file("alice", make_pdf(tmp_path / "kept.pdf", "BM25 ranks by keyword"))
    notebook.remove_file("alice", "gone.pdf")
    assert learner.concepts("alice") == ["bm25", "shared"]


def test_removing_a_file_drops_facts_citing_it_keeps_other_facts_and_keeps_mastery(tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "gone.pdf", "LoRA adapters are small"))
    notebook.add_file("alice", make_pdf(tmp_path / "kept.pdf", "BM25 ranks by keyword"))
    [gone] = [p for p in notebook.search("alice", "LoRA") if p["file"] == "gone.pdf"]
    [kept] = [p for p in notebook.search("alice", "BM25") if p["file"] == "kept.pdf"]
    learner.add_fact("alice", "lora", "LoRA adapters are small", gone["id"])
    learner.add_fact("alice", "bm25", "BM25 ranks by keyword", kept["id"])
    learner.update_mastery("alice", "lora", correct=True)
    notebook.remove_file("alice", "gone.pdf")
    memory = learner.read_all("alice")
    assert [f["passage_id"] for f in memory["facts"]] == [kept["id"]]
    assert [(m["concept"], m["score"]) for m in memory["mastery"]] == [("lora", 0.65)]  # 0.5 + 0.3 * (1 - 0.5)


def test_removing_a_file_deletes_its_pictures_and_no_other_files(tmp_path):
    gone = figure_pdf(tmp_path / "gone.pdf", "LoRA adapters, see the figure")
    kept = figure_pdf(tmp_path / "kept.pdf", "BM25 ranks, see the figure")
    notebook.add_file("alice", gone)
    notebook.add_file("alice", kept)
    assert saved_pictures(gone) and saved_pictures(kept)
    notebook.remove_file("alice", "gone.pdf")
    assert not saved_pictures(gone)
    assert saved_pictures(kept)


def test_pictures_stay_until_the_last_notebook_holding_the_file_removes_it(tmp_path):
    path = figure_pdf(tmp_path / "notes.pdf", "LoRA adapters, see the figure")
    notebook.add_file("alice", path)
    notebook.add_file("bob", path)
    notebook.remove_file("alice", "notes.pdf")
    [picture] = [p for p in notebook.search("bob", "LoRA") if p["image_path"]]
    assert Path(picture["image_path"]).exists()  # bob's search still reaches it
    notebook.remove_file("bob", "notes.pdf")
    assert not saved_pictures(path)


def test_pictures_stay_if_another_notebook_holds_the_same_file_under_another_name(tmp_path):
    path = figure_pdf(tmp_path / "notes.pdf", "LoRA adapters, see the figure")
    renamed = tmp_path / "renamed.pdf"
    renamed.write_bytes(path.read_bytes())
    notebook.add_file("alice", path)
    notebook.add_file("bob", renamed)
    notebook.remove_file("alice", "notes.pdf")
    assert saved_pictures(path)


@pytest.mark.parametrize("name", ["ghost.pdf", "../bob/notes.pdf", r"..\bob\notes.pdf", "../../outside.pdf", "..", ".", ""])
def test_removing_a_name_not_in_the_notebook_touches_nothing(name, tmp_data, tmp_path):
    path = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small")
    outside = tmp_data / "outside.pdf"
    outside.write_bytes(b"not the students")
    notebook.add_file("alice", path)
    notebook.add_file("bob", path)
    before = sorted(p for p in tmp_data.rglob("*") if "chroma" not in p.parts)
    notebook.remove_file("alice", name)
    assert sorted(p for p in tmp_data.rglob("*") if "chroma" not in p.parts) == before
    assert notebook.files("alice") == notebook.files("bob") == ["notes.pdf"]
    assert learner.concepts("alice") == learner.concepts("bob") == ["adapters", "lora"]
    assert notebook.search("alice", "LoRA")[0]["file"] == notebook.search("bob", "LoRA")[0]["file"] == "notes.pdf"


def test_removing_the_last_file_empties_the_notebook_and_adding_it_again_brings_back_its_score(tmp_path):
    path = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small")
    notebook.add_file("alice", path)
    learner.update_mastery("alice", "lora", correct=True)
    notebook.remove_file("alice", "notes.pdf")
    assert notebook.files("alice") == [] and notebook.search("alice", "LoRA") == []
    assert learner.concepts("alice") == []
    assert notebook.add_file("alice", path)["status"] == "added"
    assert notebook.search("alice", "LoRA")[0]["text"] == "LoRA adapters are small"
    assert learner.concepts("alice") == ["adapters", "lora"]
    assert [m["concept"] for m in learner.read_all("alice")["mastery"]] == ["lora"]


def tutor_prompt(monkeypatch, student):
    """The system prompt a turn with the app's memory switch sends the tutor."""
    sent = []

    def chat(messages, model, **kw):
        sent.append(messages[0]["content"])
        return reply({"role": "assistant", "content": "Fine."})

    monkeypatch.setattr(s1_llm, "chat", chat)
    turn("Hello", student, 1, search_fn=notebook.search, passages_fn=notebook.get_passages, memory="learner")
    return sent[0]


def test_a_removed_files_concepts_leave_the_weak_spots_to_a_current_one_and_return_with_the_file(monkeypatch, tmp_path):
    gone = make_pdf(tmp_path / "gone.pdf", "LoRA adapters are small")
    removed = ["alpha", "beta", "gamma"]
    monkeypatch.setattr(s1_llm, "chat", concepts_reply(*removed))
    notebook.add_file("alice", gone)
    monkeypatch.setattr(s1_llm, "chat", concepts_reply("delta"))
    notebook.add_file("alice", make_pdf(tmp_path / "kept.pdf", "BM25 ranks by keyword"))
    for concept in removed:
        for _ in range(3):
            learner.update_mastery("alice", concept, correct=False)  # 0.5 * 0.7 ** 3 = 0.17
    learner.update_mastery("alice", "delta", correct=False)  # 0.5 * 0.7 = 0.35

    prompt = tutor_prompt(monkeypatch, "alice")
    assert all(f"{c} 0.17" in prompt for c in removed) and "delta 0.35" not in prompt  # the limit of 3 crowds it out

    notebook.remove_file("alice", "gone.pdf")
    prompt = tutor_prompt(monkeypatch, "alice")
    assert not any(c in prompt for c in removed) and "delta 0.35" in prompt
    assert {m["concept"] for m in learner.read_all("alice")["mastery"]} == {*removed, "delta"}  # the scores stay

    monkeypatch.setattr(s1_llm, "chat", concepts_reply(*removed))
    notebook.add_file("alice", gone)
    prompt = tutor_prompt(monkeypatch, "alice")
    assert all(f"{c} 0.17" in prompt for c in removed) and "delta 0.35" not in prompt


def test_removing_then_adding_retries_concepts_that_could_not_be_listed(monkeypatch, tmp_path):
    path = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small")
    monkeypatch.setattr(s1_llm, "chat", _rate_limited)
    assert notebook.add_file("alice", path)["concepts_found"] is False
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: CONCEPTS)
    notebook.remove_file("alice", "notes.pdf")
    assert notebook.add_file("alice", path)["concepts_found"] is True
    assert learner.concepts("alice") == ["adapters", "lora"]


# replacing a file

def replacement(tmp_path, name="notes.pdf"):
    """Where to build the new upload: its own folder, as the name it shares with the old file is what makes it a replacement."""
    (tmp_path / "new").mkdir()
    return tmp_path / "new" / name


def test_same_name_with_new_content_replaces_the_old_file(tmp_data, tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    old = notebook.search("alice", "LoRA")[0]["id"]
    new = make_pdf(replacement(tmp_path), "BM25 ranks by keyword")
    assert notebook.add_file("alice", new) == {"status": "replaced", "file": "notes.pdf", "concepts_found": True}
    assert notebook.files("alice") == ["notes.pdf"]
    assert [p["text"] for p in notebook.search("alice", "LoRA")] == ["BM25 ranks by keyword"]  # the only passage left
    assert notebook.get_passages("alice", [old]) == []
    assert (tmp_data / "uploads" / "alice" / "notes.pdf").read_bytes() == new.read_bytes()


def test_a_replacement_swaps_the_files_concepts_and_keeps_another_files_and_keeps_mastery(monkeypatch, tmp_path):
    monkeypatch.setattr(s1_llm, "chat", concepts_reply("LoRA", "Shared"))
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    monkeypatch.setattr(s1_llm, "chat", concepts_reply("Agents", "Shared"))
    notebook.add_file("alice", make_pdf(tmp_path / "other.pdf", "An agent calls tools"))
    learner.update_mastery("alice", "lora", correct=True)
    monkeypatch.setattr(s1_llm, "chat", concepts_reply("BM25"))
    notebook.add_file("alice", make_pdf(replacement(tmp_path), "BM25 ranks by keyword"))
    assert learner.concepts("alice") == ["agents", "bm25", "shared"]
    assert [m["concept"] for m in learner.read_all("alice")["mastery"]] == ["lora"]


def test_a_replacement_drops_facts_citing_the_old_file(tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    learner.add_fact("alice", "lora", "LoRA adapters are small", notebook.search("alice", "LoRA")[0]["id"])
    notebook.add_file("alice", make_pdf(replacement(tmp_path), "BM25 ranks by keyword"))
    assert learner.read_all("alice")["facts"] == []


def test_a_replacement_whose_concepts_could_not_be_listed_says_so(monkeypatch, tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    monkeypatch.setattr(s1_llm, "chat", _rate_limited)
    out = notebook.add_file("alice", make_pdf(replacement(tmp_path), "BM25 ranks by keyword"))
    assert out == {"status": "replaced", "file": "notes.pdf", "concepts_found": False}
    assert learner.concepts("alice") == []


def test_a_replacement_deletes_the_old_files_pictures_and_keeps_the_new_ones(tmp_path):
    old = figure_pdf(tmp_path / "notes.pdf", "LoRA adapters, see the figure")
    notebook.add_file("alice", old)
    new = figure_pdf(replacement(tmp_path), "BM25 ranks, see the figure")
    notebook.add_file("alice", new)
    assert not saved_pictures(old)
    assert saved_pictures(new)
    [picture] = [p for p in notebook.search("alice", "BM25") if p["image_path"]]
    assert Path(picture["image_path"]).exists()


def test_a_replacement_keeps_the_old_pictures_while_another_notebook_holds_the_old_file(tmp_path):
    old = figure_pdf(tmp_path / "notes.pdf", "LoRA adapters, see the figure")
    notebook.add_file("alice", old)
    notebook.add_file("bob", old)
    notebook.add_file("alice", figure_pdf(replacement(tmp_path), "BM25 ranks, see the figure"))
    assert saved_pictures(old)
    [picture] = [p for p in notebook.search("bob", "LoRA") if p["image_path"]]
    assert Path(picture["image_path"]).exists()


@pytest.mark.parametrize("build", [scanned_pdf, blank_pdf, corrupt_pdf])
def test_a_rejected_replacement_changes_nothing_including_pictures(build, monkeypatch, tmp_data, tmp_path):
    old = figure_pdf(tmp_path / "notes.pdf", "LoRA adapters, see the figure")
    notebook.add_file("alice", old)
    learner.add_fact("alice", "lora", "LoRA adapters are small", notebook.search("alice", "LoRA")[0]["id"])
    passages, memory = notebook.search("alice", "LoRA"), learner.read_all("alice")

    def on_disk():
        return sorted(tmp_data.glob("images/**/*")) + sorted(tmp_data.glob("uploads/**/*"))

    before = on_disk()
    bad = replacement(tmp_path)
    build(bad)
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: pytest.fail("a rejected file needs no concepts"))
    with pytest.raises(FileRejected, match="notes.pdf"):
        notebook.add_file("alice", bad)
    assert notebook.search("alice", "LoRA") == passages
    assert learner.read_all("alice") == memory and learner.concepts("alice") == ["adapters", "lora"]
    assert on_disk() == before
    assert (tmp_data / "uploads" / "alice" / "notes.pdf").read_bytes() == old.read_bytes()


def test_a_replacement_that_fails_to_index_leaves_the_old_file_searchable(monkeypatch, tmp_data, tmp_path):
    old = make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small")
    notebook.add_file("alice", old)
    new = make_pdf(replacement(tmp_path), "BM25 ranks by keyword")

    def fail(self, inputs, **kwargs):
        raise RuntimeError("out of memory")

    with monkeypatch.context() as broken:
        broken.setattr(FakeSentenceTransformer, "encode", fail)
        with pytest.raises(RuntimeError):
            notebook.add_file("alice", new)
    assert [p["text"] for p in notebook.search("alice", "LoRA")] == ["LoRA adapters are small"]
    assert learner.concepts("alice") == ["adapters", "lora"]
    assert (tmp_data / "uploads" / "alice" / "notes.pdf").read_bytes() == old.read_bytes()
    assert notebook.add_file("alice", new)["status"] == "replaced"


def test_a_name_differing_only_in_capitals_replaces_the_old_file_and_its_concepts(monkeypatch, tmp_path):
    # Windows sees these as one file, so without this the old name's concepts would never go
    monkeypatch.setattr(s1_llm, "chat", concepts_reply("LoRA"))
    notebook.add_file("alice", make_pdf(tmp_path / "notes.pdf", "LoRA adapters are small"))
    monkeypatch.setattr(s1_llm, "chat", concepts_reply("BM25"))
    out = notebook.add_file("alice", make_pdf(replacement(tmp_path, "Notes.PDF"), "BM25 ranks by keyword"))
    assert out == {"status": "replaced", "file": "Notes.PDF", "concepts_found": True}
    assert notebook.files("alice") == ["Notes.PDF"]
    assert learner.concepts("alice") == ["bm25"]


def test_a_files_name_with_another_files_content_changes_nothing_and_names_the_holder(monkeypatch, tmp_path):
    notebook.add_file("alice", make_pdf(tmp_path / "a.pdf", "LoRA adapters are small"))
    b = make_pdf(tmp_path / "b.pdf", "BM25 ranks by keyword")
    notebook.add_file("alice", b)
    copy = replacement(tmp_path, "a.pdf")
    copy.write_bytes(b.read_bytes())
    monkeypatch.setattr(s1_llm, "chat", lambda *a, **k: pytest.fail("nothing new to read"))
    assert notebook.add_file("alice", copy) == {"status": "already_held", "file": "b.pdf", "concepts_found": None}
    assert notebook.files("alice") == ["a.pdf", "b.pdf"]
    assert {p["file"] for p in notebook.search("alice", "LoRA")} == {"a.pdf", "b.pdf"}


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
        assert [c.name for c in client.list_collections()] == [f"{STORE_TAG}-{student}_{EMBEDDER}"]


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
    replies = iter([reply(SEARCH_CALL), reply({"role": "assistant", "content": "They are tiny."}),
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


def slide_figure_pdf(path, text):
    """A landscape PDF, so a slide: its text and a whole-slide PNG are stored as two chunks."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=792, height=612)
        page.insert_text((72, 72), text)
        page.insert_image(pymupdf.Rect(72, 100, 272, 300), pixmap=PICTURE)
        doc.save(path)
    return path


def test_a_slide_comes_back_as_one_passage_with_its_text_and_picture(tmp_data):
    notebook.add_file("alice", slide_figure_pdf(tmp_data / "deck.pdf", "LoRA adapters are small"))
    [found] = notebook.search("alice", "LoRA")
    assert found["kind"] == "slide" and found["text"] == "LoRA adapters are small"
    assert Path(found["image_path"]).exists()


def test_get_passages_returns_a_slide_merged_like_search(tmp_data):
    notebook.add_file("alice", slide_figure_pdf(tmp_data / "deck.pdf", "LoRA adapters are small"))
    [found] = notebook.search("alice", "LoRA")
    assert notebook.get_passages("alice", [found["id"]]) == [found]
