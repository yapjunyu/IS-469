"""Search: find the chunks that best answer a question (plan step 7).

Three methods, compared on the same gold questions:
  dense          meaning: embeddings + cosine similarity (Topic 8, slides 13-15)
  bm25           exact words: BM25 keyword scores, built on TF-IDF (Topic 1; slide 35)
  hybrid         run dense and BM25, merge the two rankings with RRF (slide 35)

Every index has the same method, search(query, k) -> list[Passage], so they can be
swapped with one setting (see make_index).
"""
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from tutor.retrieval.types import Chunk, Passage

# Takes a list of texts and returns one embedding vector per text (one row each).
EmbedFn = Callable[[list[str]], np.ndarray]

SEARCH_MODES = ("dense", "bm25", "hybrid")


def make_index(mode: str, chunks: list[Chunk], embed: EmbedFn, candidates: int = 20):
    """Build the search index for one of SEARCH_MODES."""
    if mode == "dense":
        return DenseIndex(chunks, embed)
    if mode == "bm25":
        return BM25Index(chunks)
    if mode == "hybrid":
        return HybridIndex(chunks, embed, candidates)
    raise ValueError(f"unknown search mode: {mode!r}")


# dense: meaning
class DenseIndex:
    """Holds one vector per chunk and returns the chunks closest to the question."""

    def __init__(self, chunks: list[Chunk], embed: EmbedFn):
        chunks = _embeddable(chunks, embed)
        self.chunks = chunks
        self.embed = embed
        self.vectors = _normalise(_embed_chunks(chunks, embed)) if chunks else np.zeros((0, 1))

    def search(self, query: str, k: int = 5) -> list[Passage]:
        if not self.chunks:
            return []
        q = _normalise([self.embed.embed_query(query)])[0]  # the question, with the model's query prompt
        scores = self.vectors @ q  # cosine similarity with every chunk at once
        top = np.argsort(-scores)[:k]
        return [Passage(**asdict(self.chunks[i]), score=float(scores[i])) for i in top]


# dense, saved to disk: the vector store of the RAG pipeline (Topic 8 pipeline slide)
CHROMA_BATCH = 500  # chunks sent to Chroma per call
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[3] / "data" / "chroma"


class ChromaIndex:
    """Dense search like DenseIndex, but the vectors live in a Chroma database on disk.

    A file is embedded once: add_file skips a file that is already stored, so a
    later run (or a student's next visit) loads instead of re-embedding. Update a
    file by remove_file then add_file.

    There is one collection per embedding model (their vectors differ in length) and
    per `tag`. Use a new tag when the chunking settings change, since the stored
    chunks were cut with the old ones.

    Usage:   store = ChromaIndex(embed, tag="fixed-200")
             store.add_file(chunks)
             store.search("what is LoRA?")
             HybridIndex(store.chunks, embed, dense=store)   # keyword + meaning
    """

    def __init__(self, embed, tag: str = "default", path=DEFAULT_CHROMA_DIR):
        import chromadb  # imported here: slow, and only this index needs it
        self.embed = embed
        client = chromadb.PersistentClient(path=str(path))
        # cosine, so a distance d means a similarity of 1 - d, like DenseIndex scores
        self.collection = client.get_or_create_collection(
            f"{tag}_{getattr(embed, 'key', 'embed')}", configuration={"hnsw": {"space": "cosine"}})

    def has_file(self, file_hash: str) -> bool:
        found = self.collection.get(where={"file_hash": file_hash[:12]}, limit=1)
        return bool(found["ids"])

    def add_file(self, chunks: list[Chunk]) -> None:
        """Embed and store one file's chunks, unless that file is already stored."""
        chunks = _embeddable(chunks, self.embed)
        if not chunks or self.has_file(_file_key(chunks[0])):
            return
        vectors = _normalise(_embed_chunks(chunks, self.embed))
        for start in range(0, len(chunks), CHROMA_BATCH):
            part = chunks[start:start + CHROMA_BATCH]
            self.collection.add(
                ids=[c.id for c in part],
                embeddings=vectors[start:start + CHROMA_BATCH].tolist(),
                documents=[c.text for c in part],
                # Chroma rejects None in metadata, so a missing image is stored as ""
                metadatas=[{"file": c.file, "page": c.page, "page_end": c.page_end, "kind": c.kind,
                            "image_path": c.image_path or "", "file_hash": _file_key(c)}
                           for c in part])

    def remove_file(self, file_hash: str) -> None:
        self.collection.delete(where={"file_hash": file_hash[:12]})

    @property
    def chunks(self) -> list[Chunk]:
        """Every stored chunk, e.g. to build a BM25 index over the same chunks."""
        rows = self.collection.get(include=["documents", "metadatas"])
        return [_stored_chunk(i, text, meta)
                for i, text, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])]

    def search(self, query: str, k: int = 5) -> list[Passage]:
        stored = self.collection.count()
        if stored == 0:
            return []
        q = _normalise([self.embed.embed_query(query)])[0]
        found = self.collection.query(query_embeddings=[q.tolist()], n_results=min(k, stored),
                                      include=["documents", "metadatas", "distances"])
        rows = zip(found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0])
        return [Passage(**asdict(_stored_chunk(i, text, meta)), score=1 - distance)
                for i, text, meta, distance in rows]


def _file_key(chunk: Chunk) -> str:
    """The file fingerprint at the start of a chunk id ('a1b2c3d4e5f6:0007' -> 'a1b2c3d4e5f6')."""
    return chunk.id.split(":")[0]


def _stored_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(id=chunk_id, file=meta["file"], page=meta["page"], page_end=meta["page_end"],
                 text=text, kind=meta["kind"], image_path=meta["image_path"] or None)


def _embeddable(chunks: list[Chunk], embed) -> list[Chunk]:
    """The chunks this embedder can embed: text-only models never see image chunks."""
    if getattr(embed, "supports_images", False):
        return chunks
    return [c for c in chunks if not c.image_path]


def _embed_chunks(chunks: list[Chunk], embed) -> np.ndarray:
    """One vector per chunk, in order: text chunks embedded as text, image chunks as
    images. Both land in the same vector space, so one question can match either."""
    vectors = [None] * len(chunks)
    text_ids = [i for i, c in enumerate(chunks) if not c.image_path]
    image_ids = [i for i, c in enumerate(chunks) if c.image_path]
    if text_ids:
        for i, v in zip(text_ids, embed([chunks[i].text for i in text_ids])):
            vectors[i] = v
    if image_ids:
        for i, v in zip(image_ids, embed.embed_images([chunks[i].image_path for i in image_ids])):
            vectors[i] = v
    return np.array(vectors)


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
        chunks = [c for c in chunks if c.text]  # image chunks have no words to match
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

    def __init__(self, chunks: list[Chunk], embed: EmbedFn, candidates: int = 20, dense=None):
        # `dense` lets a saved index (ChromaIndex) stand in for the in-memory one
        self.dense = dense if dense is not None else DenseIndex(chunks, embed)
        self.bm25 = BM25Index(chunks)
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[Passage]:
        return rrf([self.dense.search(query, self.candidates),
                    self.bm25.search(query, self.candidates)], k)
