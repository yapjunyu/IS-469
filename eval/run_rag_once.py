"""Run the whole RAG pipeline once on the Topic 1-8 slides and save what RAGAS needs.

One setup, no comparison:  chunking = one chunk per slide, embedder = EmbeddingGemma 2 (local),
search = hybrid (BM25 + dense), top 5.

    extract -> clean -> chunk -> embed -> Chroma          (offline: once per file)
    for every gold question: search -> LLM answers        (online)
    -> eval/results/run_once.json  (question, retrieved passages, answer, reference answer)

    PYTHONPATH=src python3 eval/run_rag_once.py                     # needs GOOGLE_API_KEY for the answers
    PYTHONPATH=src python3 eval/run_rag_once.py --retrieval-only    # skip the LLM (no API key needed)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (shares load_gold, score, mean)
from dotenv import load_dotenv  # noqa: E402
from tutor.rag import answer, make_llm  # noqa: E402
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import ChromaIndex, HybridIndex  # noqa: E402

SLIDES = ROOT / "corpora" / "slides"
# Topics 5 and 6 are one deck that exists in two versions (Topic5-6_... and Topic6_...);
# only the newer one is used, and the Topic 8 .pptx duplicate is left out.
FILES = ["Topic1_TextProcessing_IS469.pdf", "Topic2_Transformers_IS469.pdf", "Topic3_Training_IS469.pdf",
         "Topic4_Efficient_FineTuning_IS469new.pdf", "Topic6_Evaluation.pdf", "Topic7_Agentic_IS469.pdf",
         "Topic8_AgenticRAG_MITB.pdf"]
SETUP = dict(chunking="slide", embedder="gemma2", search="hybrid", k=5)
STORE_TAG = "genai-slide-text"  # new tag whenever the chunking settings change, see ChromaIndex
GOLD = ROOT / "eval" / "gold" / "genai.csv"
OUT = ROOT / "eval" / "results" / "run_once.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--retrieval-only", action="store_true", help="search only, no LLM answers")
    parser.add_argument("--limit", type=int, help="only the first N questions (for a quick test)")
    args = parser.parse_args()

    load_dotenv()
    generate = not args.retrieval_only
    if generate and not os.environ.get("GOOGLE_API_KEY"):
        print("No GOOGLE_API_KEY found (environment or .env): running retrieval only.\n")
        generate = False

    # 1. Offline: chunk and embed each file once, into Chroma
    embed = Embedder(SETUP["embedder"])
    store = ChromaIndex(embed, tag=STORE_TAG)
    for name in FILES:
        start = time.perf_counter()
        path = SLIDES / name
        pages = clean_pages(extract(path, images=False))  # text only for now
        chunks = chunk_pages(pages, file_sha256(path), slide_method=SETUP["chunking"])
        known = store.has_file(chunks[0].id.split(":")[0])
        store.add_file(chunks)  # skipped if this file is already stored
        print(f"{name:42} {len(chunks):3} chunks  {'already stored' if known else 'embedded'}"
              f"  {time.perf_counter() - start:4.1f}s")
    index = HybridIndex(store.chunks, embed, dense=store)
    print(f"\n{len(store.chunks)} chunks in the store; search = {SETUP['search']}, k = {SETUP['k']}\n")

    # 2. Online: answer every gold question
    questions = ev.load_gold(GOLD)[:args.limit]
    llm = make_llm() if generate else None
    rows = []
    for q in questions:
        if generate:
            result = answer(q["question"], index, llm, k=SETUP["k"])
            passages, response = result.sources, result.text
        else:
            passages, response = index.search(q["question"], SETUP["k"]), None
        s = ev.score(q, passages)
        rows.append({
            "id": q["id"], "type": q["type"], "gold_slides": q["gold_slides"],
            "user_input": q["question"],                            # the four fields RAGAS reads
            "retrieved_contexts": [p.text for p in passages],
            "response": response,
            "reference": q["reference_answer"],
            "retrieved": [p.citation() for p in passages],
            "hit": s["hit"], "recall": s["recall"], "rr": s["rr"],
        })
        print(f"{q['id']:7} hit {s['hit']}  rr {s['rr']:.2f}  {q['question'][:70]}")

    # 3. Save, and the retrieval scores (same metrics as the earlier chunking comparison)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"setup": SETUP, "rows": rows}, indent=1, ensure_ascii=False))
    k = SETUP["k"]
    print(f"\nHit@{k} {ev.mean(r['hit'] for r in rows):.2f}   Recall@{k} {ev.mean(r['recall'] for r in rows):.2f}"
          f"   MRR {ev.mean(r['rr'] for r in rows):.2f}   ({len(rows)} questions)")
    print(f"Saved {OUT}" + ("" if generate else "  (no answers yet: add GOOGLE_API_KEY and run without --retrieval-only)"))


if __name__ == "__main__":
    main()
