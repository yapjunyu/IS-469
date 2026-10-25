"""Cut cleaned pages into passages (plan step 3).

Four methods, chosen with `method` (documents) or `slide_method` (slide decks):
  slide      one chunk per slide (slide decks only)
  fixed      every `chunk_size` words, with overlap (the baseline)
  recursive  pack whole paragraphs, then sentences, up to `chunk_size` words
  semantic   cut where neighbouring sentences change topic, up to `chunk_size` words

chunk_size and chunk_overlap are counted in words.
"""
import re
from typing import Callable

import numpy as np

from .types import Chunk, PageRecord

# Takes a list of texts and returns one embedding vector per text (one row each).
EmbedFn = Callable[[list[str]], np.ndarray]


def _chunk_id(file_hash: str, n: int) -> str:
    """e.g. 'a1b2c3d4e5f6:0007': the file's fingerprint plus the chunk's position."""
    return f"{file_hash[:12]}:{n:04d}"


def chunk_pages(pages: list[PageRecord], file_hash: str, chunk_size: int = 200,
                chunk_overlap: int = 30, method: str = "fixed", slide_method: str = "slide",
                embed: EmbedFn | None = None, semantic_percentile: float = 85) -> list[Chunk]:
    """Cut one file's cleaned pages into chunks.

    Documents use `method`; slide decks use `slide_method`, which can also be any
    document method (e.g. "recursive" packs small neighbouring slides together).
    Semantic chunking needs `embed`, and ignores chunk_overlap.
    """
    if not pages:
        return []

    if pages[0].kind == "slide":
        if slide_method == "slide":
            return _chunk_slides(pages, file_hash)
        method = slide_method

    if method in ("fixed", "recursive") and not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_size")
    if method == "fixed":
        return _chunk_fixed(pages, file_hash, chunk_size, chunk_overlap)
    if method == "recursive":
        return _chunk_recursive(pages, file_hash, chunk_size, chunk_overlap)
    if method == "semantic":
        if embed is None:
            raise ValueError("semantic chunking needs an embed function")
        return _chunk_semantic(pages, file_hash, chunk_size, embed, semantic_percentile)
    raise ValueError(f"unknown chunking method: {method!r}")


# one chunk per slide
def _chunk_slides(pages: list[PageRecord], file_hash: str) -> list[Chunk]:
    """A slide is usually one idea, so each slide becomes exactly one chunk."""
    chunks = []
    for n, p in enumerate(pages):
        chunks.append(
            Chunk(
                id=_chunk_id(file_hash, n),
                file=p.file,
                page=p.page,
                page_end=p.page,
                text=p.text,
                kind="slide",
            )
        )
    return chunks


# fixed window (try at different sizes) + overlap (used as baseline)
def _doc_chunk(pages, file_hash, n, text, first_page, last_page) -> Chunk:
    """Build a chunk that may span several pages (or slides)."""
    return Chunk(id=_chunk_id(file_hash, n), file=pages[0].file, page=first_page,
                 page_end=last_page, text=text, kind=pages[0].kind)


def _chunk_fixed(pages, file_hash, size, overlap) -> list[Chunk]:
    """Cut every `size` words, each window starting `overlap` words before the last one ended."""
    words = [(word, p.page) for p in pages for word in p.text.split()]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        text = " ".join(word for word, _ in window)
        chunks.append(_doc_chunk(pages, file_hash, len(chunks), text,
                                 first_page=window[0][1], last_page=window[-1][1]))
        if start + size >= len(words):
            break
    return chunks


# recursive chunking
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def _split_to_fit(text: str, size: int) -> list[str]:
    """Split one paragraph into pieces of at most `size` words: whole paragraph if it
    fits, otherwise sentences, otherwise fixed word windows."""
    if len(text.split()) <= size:
        return [text.strip()] if text.strip() else []
    pieces = []
    for sentence in _SENTENCE_END.split(text):
        words = sentence.split()
        if not words:
            continue
        if len(words) <= size:
            pieces.append(" ".join(words))
        else:
            for i in range(0, len(words), size):
                pieces.append(" ".join(words[i:i + size]))
    return pieces


def _chunk_recursive(pages, file_hash, size, overlap) -> list[Chunk]:
    """Pack whole paragraphs (or sentences) into chunks of at most `size` words."""
    # 1. Split: every piece fits within `size`, and remembers its page.
    pieces = []  # (text, page)
    for p in pages:
        for para in re.split(r"\n\s*\n", p.text):
            for piece in _split_to_fit(para, size):
                pieces.append((piece, p.page))

    # 2. Pack: add pieces until the next one won't fit, then start a new chunk.
    chunks = []
    current = []
    for piece in pieces:
        if current and _n_words(current) + _n_words([piece]) > size:
            chunks.append(_pieces_to_chunk(pages, file_hash, len(chunks), current))
            current = _overlap_tail(current, overlap)
            while current and _n_words(current) + _n_words([piece]) > size:
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append(_pieces_to_chunk(pages, file_hash, len(chunks), current))
    return chunks


def _n_words(pieces) -> int:
    return sum(len(text.split()) for text, _ in pieces)


def _overlap_tail(pieces, overlap):
    """The last few whole pieces of a chunk, totalling at most `overlap` words."""
    tail = []
    for piece in reversed(pieces):
        if _n_words(tail) + _n_words([piece]) > overlap:
            break
        tail.insert(0, piece)
    return tail


def _pieces_to_chunk(pages, file_hash, n, pieces) -> Chunk:
    text = " ".join(text for text, _ in pieces)
    return _doc_chunk(pages, file_hash, n, text,
                      first_page=pieces[0][1], last_page=pieces[-1][1])


# semantic chunking (Topic 8, slide 22: semantic segmentation)
# Bullet lines on slides rarely end in a full stop, so line breaks also end a sentence.
_SENTENCE_OR_LINE = re.compile(r"\n+|(?<=[.!?])\s+")

def _sentences(pages, size) -> list[tuple[str, int]]:
    """Every sentence (or bullet line) with its page; any longer than `size` words is cut."""
    sentences = []
    for p in pages:
        for sentence in _SENTENCE_OR_LINE.split(p.text):
            words = sentence.split()
            for i in range(0, len(words), size):
                sentences.append((" ".join(words[i:i + size]), p.page))
    return sentences


def _chunk_semantic(pages, file_hash, size, embed: EmbedFn, percentile: float) -> list[Chunk]:
    """Start a new chunk where the next sentence's meaning shifts sharply from the
    current one, or when the chunk would exceed `size` words.

    A "sharp shift" is a gap bigger than `percentile` percent of all the gaps in
    this file, so the cut-off adapts to each document. A chunk is never cut for a
    topic shift before it holds a fifth of `size`, so stray lines like "01" don't
    become chunks of their own.
    """
    min_words = size // 5
    sentences = _sentences(pages, size)
    if not sentences:
        return []

    # 1. Embed every sentence; normalise so a dot product is the cosine similarity.
    vectors = np.asarray(embed([text for text, _ in sentences]), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)

    # 2. gaps[i] = how different sentence i+1 is from sentence i (0 = same meaning).
    gaps = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = np.percentile(gaps, percentile) if len(gaps) else 0.0

    # 3. Walk through the sentences, cutting at topic shifts or when the chunk is full.
    chunks = []
    current = [sentences[0]]
    for i, sentence in enumerate(sentences[1:]):
        topic_shift = gaps[i] > threshold and _n_words(current) >= min_words
        too_big = _n_words(current) + _n_words([sentence]) > size
        if topic_shift or too_big:
            chunks.append(_pieces_to_chunk(pages, file_hash, len(chunks), current))
            current = []
        current.append(sentence)
    chunks.append(_pieces_to_chunk(pages, file_hash, len(chunks), current))
    return chunks
