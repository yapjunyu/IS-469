from tutor.pipeline import CHUNKING, EMBEDDER, K, ask
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
