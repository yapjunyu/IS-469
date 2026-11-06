"""Compare embedders on text questions, with the chunking fixed at the MiniLM winners.

    slides     one chunk per slide
    documents  fixed 200 words, 30-word overlap
    search     dense, top 3, no image chunks

Each model runs in its own process, because two large models may not fit in memory together. Each
format is scored on its own text questions. The winner is picked on dev: evidence recall@3, then MRR.

    .venv/bin/python eval/run_text_embedders.py minilm
    .venv/bin/python eval/run_text_embedders.py gemma2
    .venv-jina/bin/python eval/run_text_embedders.py jinaclip2   # its own environment: requirements-jina.txt
    .venv/bin/python eval/run_text_embedders.py --compare

Saves eval/results/text_embedder_<model>.csv and _questions.csv. --compare prints paired gains.
"""
import argparse
import csv
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (load_gold, evaluate, summarise, normalise, write_csv)
import common  # noqa: E402  (corpus_files, GOLD, OUT, CachedEmbed)
import run_minilm_chunking as mc  # noqa: E402  (K, SLIDE_METHODS, DOCUMENT_METHODS, FORMATS, truncation, paired_gain)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402

MODELS = ("minilm", "gemma2", "jinaclip2")
SLIDES = mc.SLIDE_METHODS["slide"]
DOCUMENTS = mc.DOCUMENT_METHODS["fixed"]


def run(key: str):
    files = common.corpus_files()
    chunks, by_kind = [], {"slide": [], "page": []}
    for f in files:
        pages = clean_pages(extract(f, images=False))
        kind = pages[0].kind
        file_chunks = chunk_pages(pages, file_sha256(f), **(SLIDES if kind == "slide" else DOCUMENTS))
        chunks += file_chunks
        by_kind[kind] += file_chunks
    questions = [q for q in ev.load_gold(common.GOLD) if q["evidence_type"] == "text"]
    print(f"{key}: {len(files)} files, {len(chunks)} chunks, {len(questions)} text questions", flush=True)

    model = Embedder(key)
    embed = common.CachedEmbed(model)
    began = time.perf_counter()
    index = DenseIndex(chunks, embed)
    index_s = embed.take_used_s()
    print(f"indexed in {time.perf_counter() - began:.0f}s ({index_s:.0f}s of embedding, cached or not)", flush=True)

    summary, per_question = [], []
    for fmt, (kind, corpus, _) in mc.FORMATS.items():
        qs = [q for q in questions if q["corpus"] == corpus]
        labels = {"model": key, "format": fmt}
        scored = ev.evaluate(index, qs, labels)
        per_question += scored
        own = by_kind[kind]
        words = mean(sum(len(p.text.split()) for p in index.search(q["question"], mc.K)) for q in qs)
        extra = {"chunks": len(own), "words_top3": round(words, 1), "index_s": round(index_s, 1),
                 "max_tokens": model.max_tokens, **mc.truncation(model, own)}
        summary += ev.summarise(scored, qs, labels, extra, {"text_q": lambda q: True})
    ev.write_csv(common.OUT / f"text_embedder_{key}.csv", summary)
    ev.write_csv(common.OUT / f"text_embedder_{key}_questions.csv", per_question)
    print(f"Saved {common.OUT / f'text_embedder_{key}.csv'}", flush=True)


def compare():
    scores = {}
    for key in MODELS:
        with (common.OUT / f"text_embedder_{key}_questions.csv").open() as f:
            for r in csv.DictReader(f):
                for name in r:
                    if name.endswith(("_recall", "_rr")):
                        r[name] = float(r[name])
                scores.setdefault((key, r["format"]), {})[r["id"]] = r
    for fmt in mc.FORMATS:
        for a, b in (("minilm", "gemma2"), ("minilm", "jinaclip2"), ("gemma2", "jinaclip2")):
            first, second = scores[(a, fmt)], scores[(b, fmt)]
            dev = [i for i in first if first[i]["split"] == "dev"]
            for metric in ("ev3_recall", "ev3_rr", "page3_recall", "page3_rr"):
                print(f"{fmt:9s} {b} over {a} {metric:12s} n={len(dev)} {mc.paired_gain(first, second, dev, metric)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", nargs="?", choices=MODELS)
    parser.add_argument("--compare", action="store_true", help="paired gains from the saved results")
    args = parser.parse_args()
    if args.compare:
        compare()
    elif args.model:
        run(args.model)
    else:
        parser.error("give a model or --compare")


if __name__ == "__main__":
    main()
