"""The final RAG pipeline: the champion from each evaluation stage, in one place.

Run it from any folder. The first run embeds the corpus, and later runs skip this. Needs OPENROUTER_API_KEY.

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

from tutor.rag import GENERATOR_MODEL, K, Answer, answer, make_llm  # noqa: E402, F401  (GENERATOR_MODEL and K are champions too)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DEFAULT_CHROMA_DIR, ChromaIndex, ChromaTextOnly, HybridRerankIndex  # noqa: E402

# chunk_pages arguments per page kind: one chunk per slide; documents recursive 200 words, 30-word overlap
CHUNKING = {"slide": dict(slide_method="slide"), "page": dict(method="recursive", chunk_size=200, chunk_overlap=30)}
EMBEDDER = "gemma2"  # Gemma 2, with image chunks. Search is hybrid + rerank: see HybridRerankIndex.
STORE_TAG = "slide-recursive200-images"  # use a new tag whenever CHUNKING changes, see ChromaIndex
# Every corpus file except the two duplicate decks: the older Topic 5-6 copy and the Topic 8 .pptx
# (their content repeats Topic6_Evaluation.pdf and Topic8_AgenticRAG_MITB.pdf).
SKIP = {"Topic5-6_Inference_Evaluation_IS469.pdf", "Topic8_AgenticRAG_MITB.pptx"}


def corpus_files() -> list[Path]:
    files = [p for p in (ROOT / "corpora").rglob("*")
             if p.suffix.lower() in (".pdf", ".pptx") and not p.name.startswith("~$") and p.name not in SKIP]
    return sorted(files, key=lambda p: p.name)


def build_index(files: list[Path], embed=None, path=DEFAULT_CHROMA_DIR, reranker=None) -> HybridRerankIndex:
    """Extract, clean, chunk and embed any file that isn't stored yet, and return the search index."""
    embed = embed or Embedder(EMBEDDER)
    store = ChromaIndex(embed, tag=STORE_TAG, path=path)
    for file in files:
        digest = file_sha256(file)
        if not store.has_file(digest):
            pages = clean_pages(extract(file, images=True))
            store.add_file(chunk_pages(pages, digest, **CHUNKING[pages[0].kind]))
    return HybridRerankIndex(store.chunks, embed, dense=store, text_dense=ChromaTextOnly(store), model=reranker)


def ask(question: str, index, llm=None) -> Answer:
    """Search `index` for the top K chunks and answer from them, with citations."""
    return answer(question, index, llm or make_llm(), k=K)


def show(question: str, index) -> None:
    result = ask(question, index)
    print(f"\n{result.readable()}\n")
    print("Cited:", "; ".join(result.citations()) or "none")
    read = [p.citation() for p in result.sources + result.figures]
    print("Retrieved:", "; ".join(dict.fromkeys(read)) or "none", "\n")


def main():
    index = build_index(corpus_files())
    if sys.argv[1:]:
        show(" ".join(sys.argv[1:]), index)
        return
    while question := input("Question: ").strip():
        show(question, index)


if __name__ == "__main__":
    main()
