"""Compare Gemma 2 and Jina CLIP v2 on all questions, with image chunks in the index.

    slides     one chunk per slide, plus image chunks
    documents  recursive 200 words, 30-word overlap, plus image chunks
    search     dense, top 3

Image questions get two scores:
    answer  a chunk holds the answer: an image chunk on the gold page, or any chunk on a page listed
            in also_gold, whose text alone answers the question. This score picks the winner.
    figure  the figure itself was found: an image chunk on the gold page, ignoring also_gold.
Text questions are scored as usual. They can change from the text-only run, because figures now
compete for the top 3. The winner is picked on dev, all questions: evidence recall@3, then MRR.

    .venv/bin/python eval/run_image_embedders.py gemma2
    .venv-jina/bin/python eval/run_image_embedders.py jinaclip2   # its own environment: requirements-jina.txt
    .venv/bin/python eval/run_image_embedders.py --compare

Saves eval/results_60_40/image_embedder_<model>.csv and _questions.csv. --compare prints paired gains.
"""
import argparse
import csv
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (load_gold, evaluate, summarise, write_csv)
import common  # noqa: E402  (corpus_files, GOLD, OUT, CachedImageEmbed)
import run_minilm_chunking as mc  # noqa: E402  (K, SLIDE_METHODS, DOCUMENT_METHODS, paired_gain)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402

MODELS = ("gemma2", "jinaclip2")
SLIDES = mc.SLIDE_METHODS["slide"]
DOCUMENTS = mc.DOCUMENT_METHODS["recursive"]
FORMATS = {"all": lambda corpus: True, "slides": lambda corpus: corpus == "slides",
           "documents": lambda corpus: corpus == "longdoc"}
GROUPS = {"all": lambda q: True, "text_q": lambda q: q["evidence_type"] == "text",
          "image_q": lambda q: q["evidence_type"] == "image"}


def without_also_gold(q: dict) -> dict:
    return {**q, "also": [set() for _ in q["parts"]]}


def run(key: str):
    chunks = []
    for f in common.corpus_files():
        pages = clean_pages(extract(f, images=True))
        chunks += chunk_pages(pages, file_sha256(f), **(SLIDES if pages[0].kind == "slide" else DOCUMENTS))
    questions = ev.load_gold(common.GOLD)
    images = [q for q in questions if q["evidence_type"] == "image"]
    n_images = sum(bool(c.image_path) for c in chunks)
    print(f"{key}: {len(chunks) - n_images} text chunks, {n_images} image chunks, "
          f"{len(questions)} questions, {len(images)} on images", flush=True)

    embed = common.CachedImageEmbed(Embedder(key))
    began = time.perf_counter()
    index = DenseIndex(chunks, embed)
    index_s = embed.take_used_s()
    print(f"indexed in {time.perf_counter() - began:.0f}s ({index_s:.0f}s of embedding, cached or not)", flush=True)

    answer = ev.evaluate(index, questions, {"model": key, "score": "answer"})
    figure = ev.evaluate(index, [without_also_gold(q) for q in images], {"model": key, "score": "figure"})

    summary = []
    for fmt, keep in FORMATS.items():
        qs = [q for q in questions if keep(q["corpus"])]
        ids = {q["id"] for q in qs}
        words = mean(sum(len(p.text.split()) for p in index.search(q["question"], mc.K)) for q in qs)
        extra = {"text_chunks": len(chunks) - n_images, "image_chunks": n_images,
                 "words_top3": round(words, 1), "index_s": round(index_s, 1)}
        labels = {"model": key, "format": fmt}
        summary += ev.summarise([r for r in answer if r["id"] in ids], qs, {**labels, "score": "answer"}, extra, GROUPS)
        summary += ev.summarise([r for r in figure if r["id"] in ids], [q for q in qs if q["evidence_type"] == "image"],
                                {**labels, "score": "figure"}, extra, {"image_q": GROUPS["image_q"]})
    ev.write_csv(common.OUT / f"image_embedder_{key}.csv", summary)
    ev.write_csv(common.OUT / f"image_embedder_{key}_questions.csv", answer + figure)
    print(f"Saved {common.OUT / f'image_embedder_{key}.csv'}", flush=True)


def compare():
    scores = {}
    for key in MODELS:
        with (common.OUT / f"image_embedder_{key}_questions.csv").open() as f:
            for r in csv.DictReader(f):
                for name in r:
                    if name.endswith(("_recall", "_rr")):
                        r[name] = float(r[name])
                scores.setdefault((key, r["score"]), {})[r["id"]] = r
    for fmt, keep in FORMATS.items():
        for score, kinds in (("answer", ("all", "text", "image")), ("figure", ("image",))):
            first, second = scores[("gemma2", score)], scores[("jinaclip2", score)]
            for kind in kinds:
                dev = [i for i, r in first.items() if r["split"] == "dev" and keep(r["corpus"])
                       and kind in ("all", r["evidence_type"])]
                for metric in ("ev3_recall", "ev3_rr"):
                    print(f"{fmt:9s} {score:6s} {kind:5s} jinaclip2 over gemma2 {metric:10s} n={len(dev):3d} "
                          f"{mc.paired_gain(first, second, dev, metric)}")


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
