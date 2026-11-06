"""Answer gold questions with the champion setup, ready for ragas_eval.py.

One chunk per slide, documents fixed 200 words with 30-word overlap, Gemma 2 with image chunks, dense
search, top 3 chunks. Any retrieved pictures go to the LLM with the text, as in rag.py.

    .venv/bin/python eval/run_generation.py --limit 3   # try a few first
    .venv/bin/python eval/run_generation.py             # the 75 questions marked llm_eval in gold.csv
    .venv/bin/python eval/run_generation.py --all       # every gold question

Saves eval/results/generation_sample.json, or generation_all.json with --all.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (load_gold)
import common  # noqa: E402  (corpus_files, GOLD, OUT, CachedImageEmbed)
from tutor.pipeline import CHUNKING, EMBEDDER  # noqa: E402
from tutor.rag import answer, make_llm  # noqa: E402
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--all", action="store_true", help="every gold question, not only those marked llm_eval")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    args = parser.parse_args()

    chunks = []
    for f in common.corpus_files():
        pages = clean_pages(extract(f, images=True))
        chunks += chunk_pages(pages, file_sha256(f), **CHUNKING[pages[0].kind])
    index = DenseIndex(chunks, common.CachedImageEmbed(Embedder(EMBEDDER)))

    questions = ev.load_gold(common.GOLD)
    if not args.all:
        questions = [q for q in questions if q.get("llm_eval") == "1"]
    run = "all" if args.all else "sample"
    llm, rows = make_llm(), []
    for q in questions[:args.limit]:
        result = answer(q["question"], index, llm)
        rows.append({"id": q["id"], "evidence_type": q["evidence_type"], "user_input": q["question"],
                     "retrieved_contexts": [p.text for p in result.sources],
                     "retrieved_images": [p.image_path for p in result.figures],
                     "response": result.text, "reference": q["reference_answer"],
                     "retrieved": [p.citation() for p in result.sources + result.figures]})
        print(f"{q['id']:7} {len(result.sources)} text, {len(result.figures)} images  {result.text[:70]!r}", flush=True)
        (common.OUT / f"generation_{run}.json").write_text(json.dumps(
            {"rows": rows},
            indent=1, ensure_ascii=False))
    print(f"Saved {common.OUT / f'generation_{run}.json'}")


if __name__ == "__main__":
    main()
