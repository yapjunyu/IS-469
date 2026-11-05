"""Step 3: compare the search methods in search.py, with the step 1 and step 2 winners fixed.

    dense    embeddings + cosine similarity
    bm25     keyword scores (image chunks have no words, so BM25 only sees text chunks)
    hybrid   dense and BM25 rankings merged with RRF (20 candidates from each)

The embedder and the image setting are the step 2 dev winner (read from embedder_<name>.csv).
Scored as in eval_retrieval.py, for all questions and by format and evidence.

    .venv/bin/python eval/run_search.py

Saves eval/results/search.csv and search_questions.csv.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402
import run_baseline as rb  # noqa: E402
import run_embedder as re_  # noqa: E402  (build_chunks, winner, CachedImageEmbed)
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.search import BM25Index, DenseIndex, HybridIndex  # noqa: E402

CANDIDATES = 20  # chunks each of dense and BM25 hand to RRF


def main():
    best = re_.winner()
    name, images = best["embedder"], best["images"]
    print(f"step 2 winner: {name} {images}")
    chunks = re_.build_chunks(images=images == "+images")
    questions = ev.load_gold(rb.GOLD)

    embed = re_.CachedImageEmbed(Embedder(name))
    dense = DenseIndex(chunks, embed)
    dense_s = embed.take_used_s()  # embedding time, counted in full even when vectors were saved
    indexes = {"dense": (dense, dense_s)}
    start = time.perf_counter()
    indexes["bm25"] = (BM25Index(chunks), time.perf_counter() - start)
    start = time.perf_counter()
    hybrid = HybridIndex(chunks, embed, CANDIDATES, dense=dense)  # reuses the dense vectors
    indexes["hybrid"] = (hybrid, dense_s + time.perf_counter() - start)
    print(f"{len(chunks)} chunks, {len(questions)} questions\n")

    summary, per_question = [], []
    for mode, (index, index_s) in indexes.items():
        start = time.perf_counter()
        labels = {"search": mode, "embedder": name, "images": images}
        scored = ev.evaluate(index, questions, labels)
        ms = (time.perf_counter() - start) / len(questions) * 1000
        per_question += scored
        summary += ev.summarise(scored, questions, labels,
                                 {"index_s": round(index_s, 1), "ms_per_question": round(ms, 1)})
        print(f"  {ms:.0f} ms/q\n")

    rows = [r for r in summary if r["group"] == "all"]
    top = max(rows, key=lambda r: (r["dev_ev3_recall"], r["dev_ev_mrr"]))
    print(f"→ winner on dev: {top['search']}  (test: evidence recall@3 {top['test_ev3_recall']:.3f}, "
          f"MRR {top['test_ev_mrr']:.3f})")
    ev.write_csv(rb.OUT / "search.csv", summary)
    ev.write_csv(rb.OUT / "search_questions.csv", per_question)
    print(f"Saved {rb.OUT / 'search.csv'} and {rb.OUT / 'search_questions.csv'}")


if __name__ == "__main__":
    main()
