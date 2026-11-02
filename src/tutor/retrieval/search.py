"""Search: find the chunks that best answer a question (plan step 7).

Four methods, compared on the same gold questions:
  dense          meaning: embeddings + cosine similarity (Topic 8, slides 13-15)
  bm25           exact words: BM25 keyword scores, built on TF-IDF (Topic 1; slide 35)
  hybrid         run dense and BM25, merge the two rankings with RRF (slide 35)
  hybrid_rerank  hybrid top 20, re-ordered by a cross-encoder (slides 28-32)

Every index has the same method, search(query, k) -> list[Passage], so they can be
swapped with one setting (see make_index).
"""
import re
from dataclasses import asdict, replace
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import EmbedFn
from .types import Chunk, Passage

SEARCH_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # the course lab's re-ranker


def make_index(mode: str, chunks: list[Chunk], embed: EmbedFn, candidates: int = 20):
    """Build the search index for one of SEARCH_MODES."""
    if mode == "dense":
        return DenseIndex(chunks, embed)
    if mode == "bm25":
        return BM25Index(chunks)
    if mode == "hybrid":
        return HybridIndex(chunks, embed, candidates)
    if mode == "hybrid_rerank":
        return RerankIndex(HybridIndex(chunks, embed, candidates), candidates)
    raise ValueError(f"unknown search mode: {mode!r}")


# dense: meaning
class DenseIndex:
    """Holds one vector per chunk and returns the chunks closest to the question."""

    def __init__(self, chunks: list[Chunk], embed: EmbedFn):
        self.chunks = chunks
        self.embed = embed
        self.vectors = _normalise(embed([c.text for c in chunks])) if chunks else np.zeros((0, 1))

    def search(self, query: str, k: int = 5) -> list[Passage]:
        if not self.chunks:
            return []
        q = _normalise([self.embed.embed_query(query)])[0]  # the question, with the model's query prompt
        scores = self.vectors @ q  # cosine similarity with every chunk at once
        top = np.argsort(-scores)[:k]
        return [Passage(**asdict(self.chunks[i]), score=float(scores[i])) for i in top]


def _normalise(vectors) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)


# bm25: exact words
_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase words and numbers: 'What is HyDE?' -> ['what', 'is', 'hyde']."""
    return _WORD.findall(text.lower())


class BM25Index:
    """Scores chunks by the question's words they contain, rare words counting most."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int = 5) -> list[Passage]:
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(_tokenize(query))  # one score per chunk
        top = np.argsort(-scores)[:k]
        return [Passage(**asdict(self.chunks[i]), score=float(scores[i])) for i in top]


# hybrid: dense + bm25, merged with Reciprocal Rank Fusion
def rrf(ranked_lists: list[list[Passage]], k: int = 5, rrf_k: int = 60) -> list[Passage]:
    """Merge rankings by position, ignoring their raw scores (which are on different scales).

    Each chunk gets 1/(rrf_k + rank) from every list it appears in, so chunks
    near the top of either list, or of both, rise to the top.
    """
    totals: dict[str, float] = {}
    by_id: dict[str, Passage] = {}
    for ranked in ranked_lists:
        for rank, p in enumerate(ranked, start=1):
            totals[p.id] = totals.get(p.id, 0.0) + 1 / (rrf_k + rank)
            by_id[p.id] = p
    best = sorted(totals, key=totals.get, reverse=True)[:k]
    return [replace(by_id[i], score=totals[i]) for i in best]


class HybridIndex:
    """Runs dense and BM25 search, then merges their rankings with RRF."""

    def __init__(self, chunks: list[Chunk], embed: EmbedFn, candidates: int = 20):
        self.dense = DenseIndex(chunks, embed)
        self.bm25 = BM25Index(chunks)
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[Passage]:
        return rrf([self.dense.search(query, self.candidates),
                    self.bm25.search(query, self.candidates)], k)


# re-rank: a cross-encoder re-orders the first stage's candidates
class RerankIndex:
    """Two-phase search: recall `candidates` chunks cheaply, then a cross-encoder
    reads the question together with each one and re-orders them."""

    def __init__(self, first_stage, candidates: int = 20, model_name: str = RERANK_MODEL):
        self.first_stage = first_stage
        self.candidates = candidates
        self.model = _load_cross_encoder(model_name)

    def search(self, query: str, k: int = 5) -> list[Passage]:
        pool = self.first_stage.search(query, self.candidates)  # phase 1: recall wide
        if not pool:
            return []
        scores = np.asarray(self.model.predict([(query, p.text) for p in pool]))  # phase 2: read each pair
        order = np.argsort(-scores)[:k]
        return [replace(pool[i], score=float(scores[i])) for i in order]


@lru_cache(maxsize=None)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)
