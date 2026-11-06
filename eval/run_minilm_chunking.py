"""Compare chunking on text questions with the small text-only baseline model, before any LLM.

Everything except the chunking stays fixed: MiniLM, dense search, no image chunks.

    slides     fixed 100/15  vs  recursive 100/15  vs  one chunk per slide
    documents  fixed 200/30  vs  recursive 200/30

Slides and documents share one index, so all 6 combinations are run. Each format is scored on its
own text questions. The winner for a format is picked on dev (evidence recall@3, then MRR).

Saves eval/results/minilm_chunking.csv and minilm_chunking_questions.csv.
"""
import random
import sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import common  # noqa: E402  (corpus_files, GOLD, OUT, CachedEmbed)
import eval_retrieval as ev  # noqa: E402  (load_gold, evaluate, summarise, normalise, write_csv)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402

K = 3
SLIDE_SIZE = dict(chunk_size=100, chunk_overlap=15)
DOCUMENT_SIZE = dict(chunk_size=200, chunk_overlap=30)
SLIDE_METHODS = {"fixed": dict(method="fixed", slide_method="fixed", **SLIDE_SIZE),
                 "recursive": dict(method="recursive", slide_method="recursive", **SLIDE_SIZE),
                 "slide": dict(slide_method="slide")}
DOCUMENT_METHODS = {"fixed": dict(method="fixed", **DOCUMENT_SIZE),
                    "recursive": dict(method="recursive", **DOCUMENT_SIZE)}
FORMATS = {"slides": ("slide", "slides", "slide_method"), "documents": ("page", "longdoc", "document_method")}
BASELINE = ("fixed", "fixed")
REQUESTED = {("fixed", "fixed"): "baseline", ("recursive", "recursive"): "run A", ("slide", "recursive"): "run B"}


def truncation(embed: Embedder, chunks) -> dict:
    """How many chunks are longer than the model reads, and what share of their text is never embedded."""
    tokens = [ev.n_tokens(embed, c.text) for c in chunks]
    limit = embed.max_tokens
    return {"tokens_median": median(tokens), "tokens_max": max(tokens),
            "chunks_truncated": round(sum(t > limit for t in tokens) / len(tokens), 3),
            "tokens_lost": round(sum(max(0, t - limit) for t in tokens) / sum(tokens), 3)}


def paired_gain(a: dict, b: dict, ids: list[str], metric: str, rounds: int = 5000) -> dict:
    """Mean gain of b over a, question by question, with a bootstrap 95% interval."""
    gains = [b[i][metric] - a[i][metric] for i in ids]
    rng = random.Random(469)
    samples = sorted(mean(rng.choices(gains, k=len(gains))) for _ in range(rounds))
    return {"gain": round(mean(gains), 3), "low": round(samples[int(rounds * .025)], 3),
            "high": round(samples[int(rounds * .975)], 3),
            "better": sum(g > 0 for g in gains), "worse": sum(g < 0 for g in gains), "same": sum(g == 0 for g in gains)}


def main():
    files = common.corpus_files()
    pages = {f: clean_pages(extract(f, images=False)) for f in files}
    hashes = {f: file_sha256(f) for f in files}
    kinds = {f: p[0].kind for f, p in pages.items()}
    questions = [q for q in ev.load_gold(common.GOLD) if q["evidence_type"] == "text"]
    print(f"{len(files)} files, {len(questions)} text questions\n", flush=True)

    model = Embedder("minilm")
    embed = common.CachedEmbed(model)
    summary, per_question, scores = [], [], {}
    for slide_method, slide_use in SLIDE_METHODS.items():
        for document_method, document_use in DOCUMENT_METHODS.items():
            chunks, by_kind = [], {"slide": [], "page": []}
            for f in files:
                use = slide_use if kinds[f] == "slide" else document_use
                file_chunks = chunk_pages(pages[f], hashes[f], **use)
                chunks += file_chunks
                by_kind[kinds[f]] += file_chunks
            index = DenseIndex(chunks, embed)
            config = (slide_method, document_method)
            print(f"--- slides {slide_method}, documents {document_method}: {len(chunks)} chunks "
                  f"({REQUESTED.get(config, 'extra')})", flush=True)
            for fmt, (kind, corpus, _) in FORMATS.items():
                qs = [q for q in questions if q["corpus"] == corpus]
                labels = {"slides": slide_method, "documents": document_method, "format": fmt}
                scored = ev.evaluate(index, qs, labels)
                per_question += scored
                scores[(config, fmt)] = {r["id"]: r for r in scored}

                own = by_kind[kind]
                texts = [ev.normalise(c.text) for c in own]
                quotes = [quote for q in qs for quote in q["quotes"]]
                extra = {"run": REQUESTED.get(config, "extra"), "chunks": len(own),
                         "words_per_chunk": round(mean(len(c.text.split()) for c in own), 1),
                         "pages_per_chunk": round(mean(c.page_end - c.page + 1 for c in own), 2),
                         "words_top3": round(mean(sum(len(p.text.split()) for p in index.search(q["question"], K))
                                                  for q in qs), 1),
                         "quotes_reachable": round(sum(any(t.find(quote) >= 0 for t in texts) for quote in quotes)
                                                   / len(quotes), 3),
                         **truncation(model, own)}
                summary += ev.summarise(scored, qs, labels, extra, {"text_q": lambda q: True})

    ev.write_csv(common.OUT / "minilm_chunking.csv", summary)
    ev.write_csv(common.OUT / "minilm_chunking_questions.csv", per_question)

    print("\n=== paired gains on dev against the baseline, other format held at its baseline method")
    for fmt, (_, corpus, _) in FORMATS.items():
        ids = [q["id"] for q in questions if q["corpus"] == corpus and q["split"] == "dev"]
        base = scores[(BASELINE, fmt)]
        methods = SLIDE_METHODS if fmt == "slides" else DOCUMENT_METHODS
        for method in methods:
            config = (method, "fixed") if fmt == "slides" else ("fixed", method)
            if config == BASELINE:
                continue
            for metric in ("ev3_recall", "ev3_rr", "page3_recall", "page3_rr"):
                print(f"  {fmt:9s} {method:9s} {metric:12s} n={len(ids)} {paired_gain(base, scores[(config, fmt)], ids, metric)}")
    print(f"\nSaved {common.OUT / 'minilm_chunking.csv'} and {common.OUT / 'minilm_chunking_questions.csv'}")


if __name__ == "__main__":
    main()
