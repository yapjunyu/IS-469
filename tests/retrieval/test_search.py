import numpy as np
import pytest

from tutor.retrieval.search import BM25Index, DenseIndex, HybridIndex, RerankIndex, make_index, rrf
from tutor.retrieval.types import Chunk, Passage

WORDS = ["lora", "bm25", "agent"]


class FakeEmbed:
    """One dimension per keyword: a text points towards the keywords it mentions."""

    def __call__(self, texts):
        return np.array([[t.lower().count(w) + 0.01 for w in WORDS] for t in texts])

    def embed_query(self, text):
        return self([text])[0]


fake_embed = FakeEmbed()


def chunk(n, text):
    return Chunk(id=f"h:{n}", file="deck.pptx", page=n, page_end=n, text=text, kind="slide")


def passage(n):
    return Passage(id=f"h:{n}", file="deck.pptx", page=n, page_end=n, text="", kind="slide")


CHUNKS = [chunk(1, "BM25 keyword search"), chunk(2, "LoRA adapters"), chunk(3, "Agent loop"),
          chunk(4, "HyDE drafts a hypothetical answer")]


# dense

def test_dense_returns_closest_chunks_first_with_scores():
    results = DenseIndex(CHUNKS, fake_embed).search("what is LoRA", k=2)
    assert results[0].page == 2
    assert len(results) == 2
    assert results[0].score > results[1].score


def test_empty_index_returns_nothing():
    assert DenseIndex([], fake_embed).search("anything") == []
    assert BM25Index([]).search("anything") == []


# bm25

def test_bm25_finds_exact_rare_word_regardless_of_case():
    [top] = BM25Index(CHUNKS).search("What is HYDE?", k=1)
    assert top.page == 4


# rrf / hybrid

def test_rrf_rewards_chunks_ranked_high_in_both_lists():
    dense = [passage(1), passage(2), passage(3)]
    bm25 = [passage(4), passage(2), passage(1)]
    fused = rrf([dense, bm25], k=4)
    assert [p.page for p in fused[:2]] == [1, 2]  # in both lists, near the top
    assert {p.page for p in fused} == {1, 2, 3, 4}
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_hybrid_finds_what_either_method_finds():
    index = HybridIndex(CHUNKS, fake_embed, candidates=4)
    pages = {p.page for p in index.search("LoRA and HyDE", k=2)}
    assert pages == {2, 4}  # LoRA via dense (and bm25), HyDE via bm25 only


# re-rank

class FakeCrossEncoder:
    """Scores a (question, passage) pair by how many question words the passage shares."""

    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


def test_rerank_reorders_first_stage_candidates(monkeypatch):
    monkeypatch.setattr("tutor.retrieval.search._load_cross_encoder", lambda name: FakeCrossEncoder())
    index = RerankIndex(DenseIndex(CHUNKS, fake_embed), candidates=4)
    [top] = index.search("agent loop", k=1)
    assert top.page == 3


def test_make_index_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown search mode"):
        make_index("magic", CHUNKS, fake_embed)
