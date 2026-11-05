"""Step 2: compare the embedders, with and without image chunks, with step 1's chunking winners
(read from chunking.csv) and dense search.

    gemma2   EmbeddingGemma 2      (0.74B, 768-d)
    qwen3vl  Qwen3-VL-Embedding-2B (2B,   2048-d)

    text      text chunks only
    +images   text chunks plus one image chunk per picture (whole slide for decks, cropped figure
              for documents). Text and image chunks of the same page count as one page.

Scored as in eval_retrieval.py, for all questions and by format and evidence. In a text run, image
questions score 0 on evidence, because there is no image chunk to find; compare text_q for a fair
look at text questions. The winner is picked on dev (evidence recall@3, then MRR); report test.

    .venv/bin/python eval/run_embedder.py            # both
    .venv/bin/python eval/run_embedder.py qwen3vl    # one at a time: both models together don't fit in 16 GB

Saves eval/results/embedder_<name>.csv and embedder_<name>_questions.csv per embedder.
"""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402
import run_baseline as rb  # noqa: E402
import run_chunking as rc  # noqa: E402  (CachedEmbed, settings, winner)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402

EMBEDDERS = ("gemma2", "qwen3vl")
IMAGES = ("text", "+images")


class CachedImageEmbed(rc.CachedEmbed):
    """CachedEmbed that also lets image chunks in; the text vectors are reused by the +images run."""
    supports_images = True


def chunking() -> dict:
    """chunk_pages arguments per page kind: the step 1 winners."""
    return {"slide": rc.settings(rc.winner("slides")), "page": rc.settings(rc.winner("documents"))}


def build_chunks(images: bool) -> list:
    """Every corpus file chunked with the step 1 winners; image chunks only if `images`."""
    chunks, use = [], chunking()
    for f in rb.corpus_files():
        pages = clean_pages(extract(f, images=images))
        chunks += chunk_pages(pages, file_sha256(f), **use[pages[0].kind])
    return chunks


def main():
    chunks = {"text": build_chunks(images=False), "+images": build_chunks(images=True)}
    questions = ev.load_gold(rb.GOLD)
    n_img = sum(bool(c.image_path) for c in chunks["+images"])
    print(f"{len(chunks['text'])} text chunks, {n_img} image chunks, {len(questions)} questions\n")

    run = sys.argv[1:] or EMBEDDERS
    for name in run:
        embed = CachedImageEmbed(Embedder(name))
        summary, per_question = [], []
        for images in IMAGES:
            index = DenseIndex(chunks[images], embed)  # +images reuses the saved text vectors
            index_s = embed.take_used_s()  # embedding time, counted in full even when vectors were saved
            start = time.perf_counter()
            labels = {"embedder": name, "images": images}
            scored = ev.evaluate(index, questions, labels)
            ms = (time.perf_counter() - start) / len(questions) * 1000
            per_question += scored
            summary += ev.summarise(scored, questions, labels,
                                 {"chunks": len(index.chunks), "index_s": round(index_s, 1),
                                  "ms_per_question": round(ms, 1)})
            print(f"  index {index_s:.0f}s (embedding), {ms:.0f} ms/q\n")
        ev.write_csv(rb.OUT / f"embedder_{name}.csv", summary)
        ev.write_csv(rb.OUT / f"embedder_{name}_questions.csv", per_question)
        print(f"  saved {rb.OUT / f'embedder_{name}.csv'} and embedder_{name}_questions.csv\n")

    best = winner()
    if best:
        print(f"→ winner on dev: {best['embedder']} {best['images']}  (test: evidence recall@3 "
              f"{float(best['test_ev3_recall']):.3f}, MRR {float(best['test_ev_mrr']):.3f})")


def winner() -> dict | None:
    """The best embedder + image setting on dev (evidence recall@3, then MRR), over every saved run."""
    rows = []
    for name in EMBEDDERS:
        path = rb.OUT / f"embedder_{name}.csv"
        if path.exists():
            with path.open() as f:
                rows += [r for r in csv.DictReader(f) if r["group"] == "all"]
    return max(rows, key=lambda r: (float(r["dev_ev3_recall"]), float(r["dev_ev_mrr"]))) if rows else None


if __name__ == "__main__":
    main()
