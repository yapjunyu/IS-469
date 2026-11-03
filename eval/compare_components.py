"""Compare the retrieval components one stage at a time, in pipeline order.

    Step 1  chunking   vary the chunking setup   (embedder: minilm, search: dense)
    Step 2  embedding  vary the embedding model  (chunking: step 1's winner, search: dense)
    Step 3  search     vary the search method    (chunking + embedder: the winners so far)
    Final   the course-lab baseline vs the best combination

Each step holds everything else fixed, so any difference comes from that stage alone.
Winners are picked by Recall@k, then MRR to break ties.

Semantic chunking uses an embedder to decide where to cut. So that step 2 changes only
the embedder, chunks are always cut by step 1's model (CHUNKING_EMBEDDER); the model
being tested then embeds those same chunks for search.

    .venv/bin/python eval/compare_components.py
    .venv/bin/python eval/compare_components.py --models minilm qwen3   # skip a model
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (shares setups, load_gold, score, mean, write_csv)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import MODELS, Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import SEARCH_MODES, make_index  # noqa: E402

# The course lab's setup (Lab B): fixed word windows, MiniLM, plain cosine search.
BASELINE = dict(chunking="fixed-200", embedder="minilm", search="dense")
CHUNKING_EMBEDDER = "minilm"  # cuts semantic chunks in every step, so chunks never change with the search model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", default="slides")
    parser.add_argument("--models", nargs="+", default=list(MODELS), help="embedding models for step 2")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    files = sorted(p for p in (ROOT / "corpora" / args.corpus).iterdir()
                   if p.suffix.lower() in (".pdf", ".pptx") and not p.name.startswith("~$"))
    pages = {f.name: clean_pages(extract(f)) for f in files}
    hashes = {f.name: file_sha256(f) for f in files}
    questions = ev.load_gold(ROOT / "eval" / "gold" / f"{args.corpus}.csv")
    chunking = dict(ev.setups(is_slides=all(p[0].kind == "slide" for p in pages.values())))
    print(f"{args.corpus}: {len(files)} files, {len(questions)} questions, top-{args.k}\n")

    embedders = {}

    def embedder(key):
        if key not in embedders:
            embedders[key] = Embedder(key)
        return embedders[key]

    def run(step, label, chunk_setup, embed_key, search_mode):
        """Chunk, index, search every gold question, and score the top-k."""
        embed = embedder(embed_key)
        start = time.perf_counter()
        chunks = [c for name, p in pages.items()
                  for c in chunk_pages(p, hashes[name], embed=embedder(CHUNKING_EMBEDDER),
                                       **chunking[chunk_setup])]
        index = make_index(search_mode, chunks, embed)
        index_s = time.perf_counter() - start
        start = time.perf_counter()
        results = [index.search(q["question"], args.k) for q in questions]
        ms = (time.perf_counter() - start) / len(questions) * 1000
        scores = [ev.score(q, r) for q, r in zip(questions, results)]
        row = {"step": step, "setup": label, "chunking": chunk_setup, "embedder": embed_key,
               "search": search_mode,
               "hit": ev.mean(s["hit"] for s in scores),
               "recall": ev.mean(s["recall"] for s in scores),
               "mrr": ev.mean(s["rr"] for s in scores),
               "chunks": len(chunks), "index_s": round(index_s, 1), "ms_per_question": round(ms, 1)}
        print(f"  {label:16s} hit@{args.k} {row['hit']:.2f}  recall@{args.k} {row['recall']:.2f}  "
              f"mrr {row['mrr']:.2f}  | {row['chunks']:3d} chunks, index {row['index_s']:5.1f}s, "
              f"{row['ms_per_question']:6.1f} ms/question")
        per_question.extend({"step": step, "setup": label, "id": q["id"], "question": q["question"], **s}
                            for q, s in zip(questions, scores))
        return row

    def winner(rows):
        best = max(rows, key=lambda r: (r["recall"], r["mrr"]))
        close = [r["setup"] for r in rows if r is not best and best["recall"] - r["recall"] <= 1 / len(questions)]
        note = f" (within one question: {', '.join(close)})" if close else ""
        print(f"  → winner: {best['setup']}{note}\n")
        return best

    summary, per_question = [], []

    print("Step 1 · chunking   (embedder minilm, search dense)")
    step1 = [run("1 chunking", name, name, "minilm", "dense") for name in chunking]
    best_chunking = winner(step1)["chunking"]
    summary += step1

    print(f"Step 2 · embedding  (chunking {best_chunking}, search dense)")
    step2 = []
    for key in args.models:
        try:
            step2.append(run("2 embedding", key, best_chunking, key, "dense"))
        except Exception as e:  # e.g. Gemma's gated-access 403 before the licence is approved
            print(f"  {key:16s} skipped: {type(e).__name__}: {str(e).splitlines()[-1][:100]}")
    best_embedder = winner(step2)["embedder"]
    summary += step2

    print(f"Step 3 · search     (chunking {best_chunking}, embedder {best_embedder})")
    step3 = [run("3 search", mode, best_chunking, best_embedder, mode) for mode in SEARCH_MODES]
    best_search = winner(step3)["search"]
    summary += step3

    print("Final · course-lab baseline vs best combination")
    summary.append(run("final", "baseline", BASELINE["chunking"], BASELINE["embedder"], BASELINE["search"]))
    summary.append(run("final", "best", best_chunking, best_embedder, best_search))

    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    ev.write_csv(out / f"components_{args.corpus}.csv", summary)
    ev.write_csv(out / f"components_{args.corpus}_questions.csv", per_question)
    print(f"\nBest: chunking {best_chunking} + embedder {best_embedder} + search {best_search}")
    print(f"Saved {out / f'components_{args.corpus}.csv'} and the per-question breakdown.")


if __name__ == "__main__":
    main()
