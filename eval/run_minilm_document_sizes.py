"""Compare document chunk sizes with MiniLM on text questions, looking for a size it can read in full.

Slides stay at one chunk per slide, the winner from run_minilm_chunking.py. Fixed and recursive are
compared at 100, 150 and 200 words, each with 15% overlap. Picked on dev: evidence recall@3, then MRR.

    .venv/bin/python eval/run_minilm_document_sizes.py

Saves eval/results_60_40/minilm_document_sizes.csv and minilm_document_sizes_questions.csv.
"""
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (load_gold, evaluate, summarise, normalise, write_csv)
import common  # noqa: E402  (corpus_files, GOLD, OUT, CachedEmbed)
import run_minilm_chunking as mc  # noqa: E402  (K, SLIDE_METHODS, truncation, paired_gain)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402

SIZES = {100: 15, 150: 22, 200: 30}
METHODS = ("fixed", "recursive")


def main():
    files = common.corpus_files()
    pages = {f: clean_pages(extract(f, images=False)) for f in files}
    hashes = {f: file_sha256(f) for f in files}
    kinds = {f: p[0].kind for f, p in pages.items()}
    qs = [q for q in ev.load_gold(common.GOLD) if q["evidence_type"] == "text" and q["corpus"] == "longdoc"]
    dev = [q["id"] for q in qs if q["split"] == "dev"]
    quotes = [quote for q in qs for quote in q["quotes"]]
    print(f"{len(qs)} document text questions, {len(dev)} on dev\n", flush=True)

    model = Embedder("minilm")
    embed = common.CachedEmbed(model)
    summary, per_question, scores = [], [], {}
    for size, overlap in SIZES.items():
        for method in METHODS:
            use = dict(method=method, chunk_size=size, chunk_overlap=overlap)
            chunks, own = [], []
            for f in files:
                file_chunks = chunk_pages(pages[f], hashes[f], **(mc.SLIDE_METHODS["slide"] if kinds[f] == "slide" else use))
                chunks += file_chunks
                if kinds[f] == "page":
                    own += file_chunks
            index = DenseIndex(chunks, embed)
            labels = {"method": method, "size": size, "overlap": overlap}
            scored = ev.evaluate(index, qs, labels)
            per_question += scored
            scores[(method, size)] = {r["id"]: r for r in scored}
            texts = [ev.normalise(c.text) for c in own]
            extra = {"chunks": len(own), "words_per_chunk": round(mean(len(c.text.split()) for c in own), 1),
                     "pages_per_chunk": round(mean(c.page_end - c.page + 1 for c in own), 2),
                     "words_top3": round(mean(sum(len(p.text.split()) for p in index.search(q["question"], mc.K))
                                              for q in qs), 1),
                     "quotes_reachable": round(sum(any(quote in t for t in texts) for quote in quotes) / len(quotes), 3),
                     **mc.truncation(model, own)}
            summary += ev.summarise(scored, qs, labels, extra, {"text_q": lambda q: True})

    ev.write_csv(common.OUT / "minilm_document_sizes.csv", summary)
    ev.write_csv(common.OUT / "minilm_document_sizes_questions.csv", per_question)

    print("\n=== paired gains on dev")
    pairs = [(("fixed", s), ("recursive", s)) for s in SIZES]
    pairs += [(("fixed", 200), ("fixed", s)) for s in SIZES if s != 200]
    pairs += [(("recursive", 200), ("recursive", s)) for s in SIZES if s != 200]
    for a, b in pairs:
        for metric in ("ev3_recall", "ev3_rr", "page3_recall", "page3_rr"):
            print(f"  {b} over {a} {metric:12s} n={len(dev)} {mc.paired_gain(scores[a], scores[b], dev, metric)}")
    print(f"\nSaved {common.OUT / 'minilm_document_sizes.csv'}")


if __name__ == "__main__":
    main()
