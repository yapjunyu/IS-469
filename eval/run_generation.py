"""Answer the gold questions with one slide chunking, ready for ragas_eval.py.

Everything except the slide chunking stays at the winners: documents fixed, Gemma 2 + images, dense
search, top 3 chunks. Up to 2 retrieved pictures go to the LLM with the text, as in rag.py.

    .venv/bin/python eval/run_generation.py slide                     # all dev slide questions
    .venv/bin/python eval/run_generation.py recursive --limit 3
    .venv/bin/python eval/run_generation.py recursive --llm-sample    # the questions marked llm_eval in gold.csv

Saves eval/results/generation_<slide chunking>.json, or generation_<slide chunking>_sample.json
with --llm-sample.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (load_gold)
import run_baseline as rb  # noqa: E402  (corpus_files, GOLD, OUT)
import run_chunking as rc  # noqa: E402  (settings)
import run_embedder as re_  # noqa: E402  (CachedImageEmbed)
from tutor.rag import answer, make_llm  # noqa: E402
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402

K = 3


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("slide", choices=rc.METHODS["slides"], help="slide chunking method")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--corpus", default="slides")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--llm-sample", action="store_true",
                        help="the questions marked llm_eval in gold.csv; ignores --split and --corpus")
    args = parser.parse_args()

    chunks = []
    for f in rb.corpus_files():
        pages = clean_pages(extract(f, images=True))
        use = rc.settings(args.slide) if pages[0].kind == "slide" else rc.settings("fixed")
        chunks += chunk_pages(pages, file_sha256(f), **use)
    index = DenseIndex(chunks, re_.CachedImageEmbed(Embedder("gemma2")))

    questions = ev.load_gold(rb.GOLD)
    if args.llm_sample:
        questions = [q for q in questions if q.get("llm_eval") == "1"]
    else:
        questions = [q for q in questions if q["split"] == args.split and q["corpus"] == args.corpus]
    run = f"{args.slide}_sample" if args.llm_sample else args.slide  # the sample keeps its own results file
    llm, rows = make_llm(), []
    for q in questions[:args.limit]:
        result = answer(q["question"], index, llm, k=K)
        rows.append({"id": q["id"], "evidence_type": q["evidence_type"], "user_input": q["question"],
                     "retrieved_contexts": [p.text for p in result.sources],
                     "retrieved_images": [p.image_path for p in result.figures[:result.images_sent]],
                     "response": result.text, "reference": q["reference_answer"],
                     "retrieved": [p.citation() for p in result.sources + result.figures[:result.images_sent]]})
        print(f"{q['id']:7} {len(result.sources)} text, {result.images_sent} images  {result.text[:70]!r}", flush=True)
        (rb.OUT / f"generation_{run}.json").write_text(json.dumps(
            {"slide": args.slide, "split": args.split, "corpus": args.corpus, "k": K,
             "llm_sample": args.llm_sample, "rows": rows},
            indent=1, ensure_ascii=False))
    print(f"Saved {rb.OUT / f'generation_{run}.json'}")


if __name__ == "__main__":
    main()
