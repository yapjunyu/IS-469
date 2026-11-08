"""The final RAG pipeline: the champion from each evaluation stage, in one place.

Run it from any folder. The first run embeds the corpus, and later runs skip this. Needs OPENCODE_API_KEY.

    .venv/bin/python src/tutor/pipeline.py "What is cloud computing, and which key problems does it solve?"
    .venv/bin/python src/tutor/pipeline.py      # keeps asking until you press Enter on an empty line

Or use it from code, with src on the Python path:

    index = build_index(corpus_files())         # each file is embedded once, then saved to disk
    result = ask("What is LoRA?", index)
    print(result.readable())                    # the answer, with its citations

Why each setting won: eval/results_60_40/SUMMARY_60_40.md.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:  # lets this file run as a script, as well as being imported
    sys.path.insert(0, str(ROOT / "src"))

from tutor.generator import GENERATOR_MODEL, K, Answer, answer, make_llm  # noqa: E402, F401  (GENERATOR_MODEL and K are champions too)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import (DEFAULT_CHROMA_DIR, ChromaIndex, ChromaTextOnly,  # noqa: E402
                                    HybridRerankIndex, SlideMergeIndex)
from tutor.retrieval.types import PageRecord  # noqa: E402

# chunk_pages arguments per page kind: one chunk per slide; documents recursive 200 words, 30-word overlap
CHUNKING = {"slide": dict(slide_method="slide"), "page": dict(method="recursive", chunk_size=200, chunk_overlap=30)}
EMBEDDER = "gemma2"  # Gemma 2, with image chunks. Search is hybrid + rerank: see HybridRerankIndex.
STORE_TAG = "slide-recursive200-images"  # use a new tag whenever CHUNKING changes, see ChromaIndex

def corpus_files() -> list[Path]:
    files = [p for p in (ROOT / "corpora").rglob("*")
             if p.suffix.lower() in (".pdf", ".pptx") and not p.name.startswith("~$")]
    return sorted(files, key=lambda p: p.name)


def store_pages(store: ChromaIndex, pages: list[PageRecord], digest: str) -> None:
    """Chunk one file's cleaned pages and store them. `digest` is the file's fingerprint."""
    store.add_file(chunk_pages(pages, digest, **CHUNKING[pages[0].kind]))


def search_index(store: ChromaIndex, reranker=None) -> SlideMergeIndex:
    """The chosen search method over everything in `store`, with a slide's text and PNG merged into one result."""
    return SlideMergeIndex(HybridRerankIndex(store.chunks, store.embed, dense=store,
                                             text_dense=ChromaTextOnly(store), model=reranker))


def build_index(files: list[Path], embed=None, path=DEFAULT_CHROMA_DIR, reranker=None) -> SlideMergeIndex:
    """Extract, clean, chunk and embed any file that isn't stored yet, and return the search index."""
    embed = embed or Embedder(EMBEDDER)
    store = ChromaIndex(embed, tag=STORE_TAG, path=path)
    for file in files:
        digest = file_sha256(file)
        if not store.has_file(digest):
            store_pages(store, clean_pages(extract(file, images=True)), digest)
    return search_index(store, reranker)


def ask(question: str, index, llm=None) -> Answer:
    """Search `index` for the top K chunks and answer from them, with citations."""
    return answer(question, index, llm or make_llm(), k=K)


def show(question: str, index) -> None:
    result = ask(question, index)
    print(f"\n{result.readable()}\n")
    for figure in result.cited_figures():  # a terminal can't draw a picture, so give its file to open
        print(f"Figure {figure.citation()}: {figure.image_path}")


def main():
    index = build_index(corpus_files())
    if sys.argv[1:]:
        show(" ".join(sys.argv[1:]), index)
        return
    while question := input("Question: ").strip():
        show(question, index)


if __name__ == "__main__":
    main()
