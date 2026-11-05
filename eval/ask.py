"""Ask the tutor a question, using the best setup from the retrieval comparison.

    chunking   the step 1 winners (chunking.csv)
    embedder   the step 2 winner (embedder_*.csv), with its image setting
    search     dense, top 3 pages
    answer     rag.py (Gemma 4 through Google's API, needs GOOGLE_API_KEY)

The first run loads the corpus into Chroma, reusing vectors saved by the eval runs. Later runs skip this.

    .venv/bin/python eval/ask.py "What happens in Kafka if a broker goes down?"
    .venv/bin/python eval/ask.py          # keeps asking until you press Enter on an empty line
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import run_baseline as rb  # noqa: E402  (corpus_files)
import run_embedder as re_  # noqa: E402  (chunking, winner, CachedImageEmbed)
from tutor.rag import answer, make_llm  # noqa: E402
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import ChromaIndex  # noqa: E402

STORE_TAG = "best-slide-fixed-images"  # new tag whenever the chunking settings change, see ChromaIndex
K_PAGES = 3
CANDIDATES = 50  # ranked chunks to walk through to find the top pages, as in run_chunking.top_pages


class TopPages:
    """Search that returns the chunks of the first k distinct pages, which is how the
    evaluation counted k: a page's text and image chunks are one page, not two results."""

    def __init__(self, index):
        self.index = index

    def search(self, query: str, k: int = K_PAGES) -> list:
        found = self.index.search(query, CANDIDATES)
        pages = []
        for p in found:
            for page in range(p.page, p.page_end + 1):
                if (p.file, page) not in pages and len(pages) < k:
                    pages.append((p.file, page))
        return [p for p in found if any((p.file, page) in pages for page in range(p.page, p.page_end + 1))]


def index_files(store: ChromaIndex, files: list[Path], images: bool = True) -> None:
    """Embed any file that isn't in the store yet."""
    use = re_.chunking()
    for path in files:
        digest = file_sha256(path)
        if store.has_file(digest):
            continue
        start = time.perf_counter()
        pages = clean_pages(extract(path, images=images))
        chunks = chunk_pages(pages, digest, **use[pages[0].kind])
        store.add_file(chunks)
        print(f"embedded {path.name}: {len(chunks)} chunks, {time.perf_counter() - start:.0f}s", flush=True)


def show(question: str, index, llm) -> None:
    result = answer(question, index, llm, k=K_PAGES)
    print(f"\n{result.readable()}\n")
    print("Cited:", "; ".join(result.citations()) or "none")
    read = [p.citation() for p in result.sources + result.figures[:result.images_sent]]
    print("Retrieved:", "; ".join(dict.fromkeys(read)) or "none", "\n")


def main():
    best = re_.winner()
    embed = re_.CachedImageEmbed(Embedder(best["embedder"]))
    store = ChromaIndex(embed, tag=STORE_TAG)
    index_files(store, rb.corpus_files(), images=best["images"] == "+images")
    index, llm = TopPages(store), make_llm()

    if sys.argv[1:]:
        show(" ".join(sys.argv[1:]), index, llm)
        return
    while question := input("Question: ").strip():
        show(question, index, llm)


if __name__ == "__main__":
    main()
