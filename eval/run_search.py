"""Step 3: compare the search methods in search.py, with the chunking and embedder winners fixed.

    slides     one chunk per slide, plus image chunks
    documents  recursive 200 words, 30-word overlap, plus image chunks
    embedder   Gemma 2

    dense          embeddings + cosine similarity
    hybrid         dense and BM25 rankings merged with RRF (20 candidates from each)
    hybrid_rerank  the top 20 text chunks from hybrid, reranked by a cross-encoder. It reads the question
                   and a chunk's text together, so it cannot score images, and images keep their dense
                   positions
    hybrid_rerank_merged  hybrid_rerank, with a slide's text and PNG merged into one result (SlideMergeIndex);
                   document chunks stay separate

Image questions get an answer score and a figure score, as in run_image_embedders.py. The winner is
picked on dev, all questions: evidence recall@3, then MRR. Paired gains are printed against dense.

    .venv/bin/python eval/run_search.py

Saves eval/results_60_40/search.csv and search_questions.csv.
"""
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (load_gold, evaluate, summarise, write_csv)
import common  # noqa: E402  (corpus_files, GOLD, OUT, CachedImageEmbed)
import run_image_embedders as ie  # noqa: E402  (SLIDES, DOCUMENTS, FORMATS, GROUPS, without_also_gold)
import run_minilm_chunking as mc  # noqa: E402  (K, paired_gain)
from tutor.pipeline import EMBEDDER  # noqa: E402
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex, HybridIndex, HybridRerankIndex, SlideMergeIndex  # noqa: E402

CANDIDATES = 20  # chunks each of dense and BM25 hand to RRF, and text chunks the reranker reads


def main():
    chunks = []
    for f in common.corpus_files():
        pages = clean_pages(extract(f, images=True))
        chunks += chunk_pages(pages, file_sha256(f), **(ie.SLIDES if pages[0].kind == "slide" else ie.DOCUMENTS))
    questions = ev.load_gold(common.GOLD)
    images = [q for q in questions if q["evidence_type"] == "image"]
    print(f"{len(chunks)} chunks, {len(questions)} questions, {len(images)} on images\n", flush=True)

    embed = common.CachedImageEmbed(Embedder(EMBEDDER))
    dense = DenseIndex(chunks, embed)
    dense_s = embed.take_used_s()  # embedding time, counted in full even when vectors were saved
    hybrid = HybridIndex(chunks, embed, CANDIDATES, dense=dense)  # reuses the dense vectors
    rerank = HybridRerankIndex(chunks, embed, CANDIDATES, dense=dense)
    merged = SlideMergeIndex(rerank)  # same ranking, a slide's text and PNG counted as one result
    indexes = {"dense": (dense, dense_s), "hybrid": (hybrid, dense_s), "hybrid_rerank": (rerank, dense_s),
               "hybrid_rerank_merged": (merged, dense_s)}

    summary, per_question, scored = [], [], {}
    for mode, (index, index_s) in indexes.items():
        start = time.perf_counter()
        answer = ev.evaluate(index, questions, {"search": mode, "score": "answer"})
        ms = (time.perf_counter() - start) / len(questions) * 1000
        figure = ev.evaluate(index, [ie.without_also_gold(q) for q in images], {"search": mode, "score": "figure"})
        per_question += answer + figure
        scored[mode, "answer"] = {r["id"]: r for r in answer}
        scored[mode, "figure"] = {r["id"]: r for r in figure}
        for fmt, keep in ie.FORMATS.items():
            qs = [q for q in questions if keep(q["corpus"])]
            ids = {q["id"] for q in qs}
            words = mean(sum(len(p.text.split()) for p in index.search(q["question"], mc.K)) for q in qs)
            extra = {"words_top3": round(words, 1), "index_s": round(index_s, 1), "ms_per_question": round(ms, 1)}
            labels = {"search": mode, "format": fmt}
            summary += ev.summarise([r for r in answer if r["id"] in ids], qs, {**labels, "score": "answer"},
                                    extra, ie.GROUPS)
            summary += ev.summarise([r for r in figure if r["id"] in ids],
                                    [q for q in qs if q["evidence_type"] == "image"],
                                    {**labels, "score": "figure"}, extra, {"image_q": ie.GROUPS["image_q"]})
        print(f"  {ms:.0f} ms/q\n", flush=True)

    rows = [r for r in summary if (r["format"], r["score"], r["group"]) == ("all", "answer", "all")]
    top = max(rows, key=lambda r: (r["dev_ev3_recall"], r["dev_ev_mrr"]))
    print(f"→ winner on dev: {top['search']}  (test: evidence recall@3 {top['test_ev3_recall']:.3f}, "
          f"MRR {top['test_ev_mrr']:.3f})\n")

    for mode in list(indexes)[1:]:
        for score, kinds in (("answer", ("all", "text", "image")), ("figure", ("image",))):
            base, other = scored["dense", score], scored[mode, score]
            for kind in kinds:
                dev = [i for i, r in base.items() if r["split"] == "dev" and kind in ("all", r["evidence_type"])]
                for metric in ("ev3_recall", "ev3_rr"):
                    print(f"{mode:13s} over dense {score:6s} {kind:5s} {metric:10s} n={len(dev):3d} "
                          f"{mc.paired_gain(base, other, dev, metric)}")

    ev.write_csv(common.OUT / "search.csv", summary)
    ev.write_csv(common.OUT / "search_questions.csv", per_question)
    print(f"\nSaved {common.OUT / 'search.csv'} and {common.OUT / 'search_questions.csv'}")


if __name__ == "__main__":
    main()
