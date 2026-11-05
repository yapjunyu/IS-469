"""Baseline retrieval run over the whole corpus, scored on eval/gold/gold.csv.

Baseline = the naive pipeline every later step is compared against:
    chunking  fixed 200-word windows, 30-word overlap (slides and documents alike)
    embedder  EmbeddingGemma 2 (local), text only, no image chunks
    search    dense cosine, top 3
Scored as in eval_retrieval.py: evidence in the top chunks (main), and page level.

    .venv/bin/python eval/run_baseline.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (shares load_gold, score, mean, write_csv)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402

SETUP = dict(chunking="fixed-200", chunk_size=200, chunk_overlap=30, embedder="gemma2", search="dense", k=3, images=False)
GOLD = ROOT / "eval" / "gold" / "gold.csv"
OUT = ROOT / "eval" / "results"
# Every corpus file except the two duplicate decks: the older Topic 5-6 copy and the Topic 8 .pptx
# (their content repeats Topic6_Evaluation.pdf and Topic8_AgenticRAG_MITB.pdf).
SKIP = {"Topic5-6_Inference_Evaluation_IS469.pdf", "Topic8_AgenticRAG_MITB.pptx"}


def corpus_files() -> list[Path]:
    files = [p for p in (ROOT / "corpora").rglob("*")
             if p.suffix.lower() in (".pdf", ".pptx") and not p.name.startswith("~$") and p.name not in SKIP]
    return sorted(files, key=lambda p: p.name)


def top_k_by_page(index, question: str, k: int) -> list:
    """The top k chunks, keeping only the best-scoring chunk per (file, first page)."""
    seen, kept = set(), []
    for p in index.search(question, k * 4):
        key = (p.file, p.page)
        if key not in seen:
            seen.add(key)
            kept.append(p)
        if len(kept) == k:
            break
    return kept


def precision(question: dict, passages) -> float:
    """Share of the returned chunks that land on a gold page of this question."""
    gold = set().union(*question["parts"])
    good = sum(p.file == question["file"] and any(p.page <= s <= p.page_end for s in gold) for p in passages)
    return good / len(passages) if passages else 0.0


def main():
    import run_chunking as rc  # imported here: run_chunking imports this file
    files = corpus_files()
    questions = ev.load_gold(GOLD)  # out-of-scope questions have no gold page, so they are left out
    print(f"{len(files)} files, {len(questions)} questions, setup {SETUP}\n")

    embed = rc.CachedEmbed(Embedder(SETUP["embedder"]))
    chunks = []
    for path in files:
        pages = clean_pages(extract(path, images=SETUP["images"]))
        file_chunks = chunk_pages(pages, file_sha256(path), chunk_size=SETUP["chunk_size"],
                                  chunk_overlap=SETUP["chunk_overlap"], method="fixed", slide_method="fixed")
        chunks += file_chunks
        print(f"  {path.name:44s} {len(pages):3d} pages  {len(file_chunks):3d} chunks")
    index = DenseIndex(chunks, embed)
    index_s = embed.take_used_s()
    print(f"\n{len(chunks)} chunks indexed in {index_s:.0f}s\n")

    start = time.perf_counter()
    labels = {"run": "baseline"}
    rows = ev.evaluate(index, questions, labels)
    ms = (time.perf_counter() - start) / len(questions) * 1000
    extra = {**SETUP, "chunks": len(chunks), "index_s": round(index_s, 1), "ms_per_question": round(ms, 1)}
    table = ev.summarise(rows, questions, labels, extra)
    print(f"\nindex {index_s:.0f}s (embedding only), {ms:.0f} ms per question")

    OUT.mkdir(exist_ok=True)
    ev.write_csv(OUT / "baseline.csv", table)
    ev.write_csv(OUT / "baseline_questions.csv", rows)
    print(f"Saved {OUT / 'baseline.csv'} and {OUT / 'baseline_questions.csv'}")


if __name__ == "__main__":
    main()
