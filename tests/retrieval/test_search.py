import numpy as np
import pytest

from tutor.retrieval.search import (BM25Index, ChromaIndex, ChromaTextOnly, DenseIndex, HybridIndex,
                                    HybridRerankIndex, SlideMergeIndex, make_index, rrf)
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


def test_make_index_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown search mode"):
        make_index("magic", CHUNKS, fake_embed)


# chroma: dense search saved to disk

class CountingEmbed(FakeEmbed):
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return super().__call__(texts)


def test_chroma_matches_dense_search(tmp_path):
    store = ChromaIndex(fake_embed, path=tmp_path)
    store.add_file(CHUNKS)
    # top 2 only: chunks 1 and 3 tie for third place, and the two indexes may break the tie differently
    dense = DenseIndex(CHUNKS, fake_embed).search("what is LoRA", k=2)
    saved = store.search("what is LoRA", k=2)
    assert [p.page for p in saved] == [p.page for p in dense]
    assert saved[0].score == pytest.approx(dense[0].score, abs=1e-4)


def test_chroma_keeps_vectors_between_runs_and_skips_stored_files(tmp_path):
    first = CountingEmbed()
    ChromaIndex(first, path=tmp_path).add_file(CHUNKS)
    assert first.calls == 1

    second = CountingEmbed()
    reopened = ChromaIndex(second, path=tmp_path)
    reopened.add_file(CHUNKS)  # same file again: nothing is embedded
    assert second.calls == 0
    assert reopened.search("agent loop", k=1)[0].page == 3


def test_chroma_remove_file(tmp_path):
    store = ChromaIndex(fake_embed, path=tmp_path)
    store.add_file(CHUNKS)
    store.remove_file("h")
    assert store.search("LoRA") == []
    assert store.chunks == []


def test_chroma_returns_the_chunks_it_was_given(tmp_path):
    store = ChromaIndex(fake_embed, path=tmp_path)
    store.add_file(CHUNKS)
    assert sorted(store.chunks, key=lambda c: c.page) == CHUNKS  # image_path comes back as None


def test_chroma_skips_image_chunks_for_text_only_models(tmp_path):
    image = Chunk(id="h:9", file="deck.pptx", page=9, page_end=9, text="", kind="slide", image_path="x.png")
    store = ChromaIndex(fake_embed, path=tmp_path)
    store.add_file(CHUNKS + [image])
    assert len(store.chunks) == len(CHUNKS)


def test_hybrid_can_use_chroma_as_its_dense_half(tmp_path):
    store = ChromaIndex(fake_embed, path=tmp_path)
    store.add_file(CHUNKS)
    index = HybridIndex(store.chunks, fake_embed, candidates=4, dense=store)
    assert {p.page for p in index.search("LoRA and HyDE", k=2)} == {2, 4}


# hybrid_rerank
class FakeReranker:
    """Scores a chunk by how often it mentions "agent", so it can overturn the first ranking."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return np.array([text.lower().count("agent") for _, text in pairs], dtype=float)


class FixedDense:
    """A dense ranking set by hand, with a picture second."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k=5):
        return [Passage(id=c.id, file=c.file, page=c.page, page_end=c.page_end, text=c.text, kind=c.kind,
                        image_path=c.image_path, score=0.5) for c in self.chunks[:k]]


PICTURE = Chunk(id="h:9", file="deck.pptx", page=9, page_end=9, text="", kind="slide", image_path="fig.png")


def test_rerank_reorders_text_and_a_picture_keeps_its_dense_position():
    dense = FixedDense([CHUNKS[0], PICTURE, CHUNKS[1], CHUNKS[2]])
    index = HybridRerankIndex(CHUNKS + [PICTURE], fake_embed, dense=dense, model=FakeReranker())
    found = index.search("lora adapters", k=3)
    assert found[1].image_path == "fig.png"
    assert found[0].page == 3  # the reranker lifts the agent chunk over what hybrid ranked first


def test_rerank_scores_each_question_once():
    model = FakeReranker()
    index = HybridRerankIndex(CHUNKS, fake_embed, model=model)
    index.search("lora", k=3)
    index.search("lora", k=5)
    assert model.calls == 1


def test_chroma_text_only_skips_pictures(tmp_path):
    class Embed(FakeEmbed):
        supports_images = True
        key = "fake"

        def embed_images(self, paths):
            return np.ones((len(paths), len(WORDS)))

    store = ChromaIndex(Embed(), tag="t", path=tmp_path)
    store.add_file(CHUNKS + [PICTURE])
    found = ChromaTextOnly(store).search("lora", k=5)
    assert found and not any(p.image_path for p in found)


class FixedIndex:
    def __init__(self, *passages):
        self.passages = list(passages)

    def search(self, query, k=5):
        return self.passages[:k]


def hit(n, page, kind="slide", text="", image=None):
    return Passage(id=f"h:{n}", file="deck.pptx", page=page, page_end=page, text=text, kind=kind, image_path=image)


def test_slide_text_and_png_merge_into_one_result_but_document_chunks_do_not():
    index = SlideMergeIndex(FixedIndex(hit(1, 26, text="t26"), hit(2, 26, image="26.png"),
                                       hit(3, 12, "page", text="p12"), hit(4, 12, "page", image="12.png")))
    found = index.search("q", 3)
    assert [(p.page, p.text, p.image_path) for p in found] == [
        (26, "t26", "26.png"), (12, "p12", None), (12, "", "12.png")]


def test_merging_still_returns_k_results():
    index = SlideMergeIndex(FixedIndex(hit(1, 1, text="a"), hit(2, 1, image="1.png"),
                                       hit(3, 2, text="b"), hit(4, 2, image="2.png"),
                                       hit(5, 3, text="c"), hit(6, 4, text="d")))
    assert [p.page for p in index.search("q", 3)] == [1, 2, 3]
