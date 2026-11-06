"""
Three methods, chosen with `method` (documents) or `slide_method` (slide decks):
  slide      one chunk per slide 
  fixed      every `chunk_size` words, with overlap (the baseline)
  recursive  pack whole paragraphs, then sentences, up to `chunk_size` words
"""
import re
from tutor.retrieval.types import Chunk, PageRecord

def _chunk_id(file_hash: str, n: int) -> str:
    """e.g. 'a1b2c3d4e5f6:0007': the file's fingerprint plus the chunk's position."""
    return f"{file_hash[:12]}:{n:04d}"

def chunk_pages(pages: list[PageRecord], file_hash: str, chunk_size: int = 200,
                chunk_overlap: int = 30, method: str = "fixed",
                slide_method: str = "slide") -> list[Chunk]:
    text_pages = [p for p in pages if p.text.strip()]  
    chunks = _chunk_text(text_pages, file_hash, chunk_size, chunk_overlap, method, slide_method)
    return chunks + _chunk_images(pages, file_hash, start=len(chunks))


# one chunk per page image
def _chunk_images(pages: list[PageRecord], file_hash: str, start: int) -> list[Chunk]:
    """Each saved image becomes one extra chunk that is embedded as an image, so a
    question can match a figure that the page's text never describes."""
    figures = [(p, path) for p in pages for path in p.image_paths]
    return [Chunk(id=_chunk_id(file_hash, start + i), file=p.file, page=p.page,
                  page_end=p.page, text="", kind=p.kind, image_path=path)
            for i, (p, path) in enumerate(figures)]


def _chunk_text(pages, file_hash, chunk_size, chunk_overlap, method, slide_method) -> list[Chunk]:
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
    """The last `overlap` words of a chunk: whole pieces first, then the end of the piece before
    them, so the overlap matches fixed chunking even when the last paragraph is long."""
    tail = []
    for piece in reversed(pieces):
        left = overlap - _n_words(tail)
        if left <= 0:
            break
        words = piece[0].split()
        if len(words) > left:
            tail.insert(0, (" ".join(words[-left:]), piece[1]))
            break
        tail.insert(0, piece)
    return tail


def _pieces_to_chunk(pages, file_hash, n, pieces) -> Chunk:
    text = " ".join(text for text, _ in pieces)
    return _doc_chunk(pages, file_hash, n, text,
                      first_page=pieces[0][1], last_page=pieces[-1][1])
