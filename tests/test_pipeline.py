import numpy as np
import pymupdf

from tutor.pipeline import CHUNKING, EMBEDDER, K, ask, build_index
from tutor.retrieval.types import Passage


class FakeIndex:
    def __init__(self):
        self.asked_for = None

    def search(self, query, k=5):
        self.asked_for = k
        return [Passage(id="h:1", file="deck.pptx", page=1, page_end=1, text="text", kind="slide", score=0.5)]


def test_champion_settings():
    assert CHUNKING == {"slide": {"slide_method": "slide"},
                        "page": {"method": "recursive", "chunk_size": 200, "chunk_overlap": 30}}
    assert EMBEDDER == "gemma2" and K == 3


def test_ask_searches_the_top_3_and_answers_with_citations():
    index = FakeIndex()
    result = ask("q", index, llm=lambda prompt: "It says so [1].")
    assert index.asked_for == 3
    assert result.citations() == ["deck.pptx, slide 1"]


class FakeEmbed:
    def __init__(self):
        self.batches = 0

    def __call__(self, texts):
        self.batches += 1
        return np.array([[t.lower().count("lora") + 0.01, 0.01] for t in texts])

    def embed_query(self, text):
        return np.array([text.lower().count("lora") + 0.01, 0.01])


class FakeReranker:
    def predict(self, pairs):
        return np.array([float("lora" in text.lower()) for _, text in pairs])


def test_build_index_embeds_each_file_once_and_finds_its_slides(tmp_path):
    with pymupdf.open() as doc:
        for text in ("BM25 ranks by keyword", "LoRA adapters are small"):
            doc.new_page(width=792, height=612).insert_text((72, 72), text)
        doc.save(tmp_path / "deck.pdf")
    embed = FakeEmbed()
    for _ in range(2):  # the second run finds the file stored and embeds nothing
        index = build_index([tmp_path / "deck.pdf"], embed=embed, path=tmp_path / "chroma", reranker=FakeReranker())
    assert embed.batches == 1
    top = index.search("LoRA", 1)[0]
    assert (top.file, top.page, top.text) == ("deck.pdf", 2, "LoRA adapters are small")
