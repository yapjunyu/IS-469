"""Combine every comparison stage into two CSVs, from the saved per-question results.

    .venv/bin/python eval/summarise_split.py

Saves summary_all_stages.csv (scores for every setup, group and split) and paired_gains.csv (each
challenger minus the stage's reference, with a bootstrap 95% interval) in eval/results_60_40.
"""
import csv
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import common  # noqa: E402  (OUT)
import eval_retrieval as ev  # noqa: E402  (write_csv)
import run_minilm_chunking as mc  # noqa: E402  (paired_gain)

METRICS = ("ev3_recall", "ev3_precision", "ev3_f1", "ev5_recall", "ev3_rr", "page3_recall")
GROUPS = {"all": lambda r: True, "text": lambda r: r["evidence_type"] == "text",
          "image": lambda r: r["evidence_type"] == "image", "slides": lambda r: r["corpus"] == "slides",
          "documents": lambda r: r["corpus"] == "longdoc"}
SPLITS = {"all": lambda r: True, "dev": lambda r: r["split"] == "dev", "test": lambda r: r["split"] == "test"}


def load(name: str, label) -> dict:
    """{(setup, score): {question id: row}} from a _questions.csv file."""
    found = {}
    with (common.OUT / f"{name}_questions.csv").open() as f:
        for r in csv.DictReader(f):
            for key in r:
                if key.endswith(METRICS + ("_rr", "_recall")):
                    r[key] = float(r[key])
            found.setdefault((label(r), r.get("score", "answer")), {})[r["id"]] = r
    return found


def stages() -> dict:
    """Stage name -> (results, reference setup, challengers to compare with it)."""
    chunking = load("minilm_chunking", lambda r: f"slides {r['slides']}, documents {r['documents']}")
    merged = {}
    for (setup, score), rows in chunking.items():
        merged.setdefault((setup, score), {}).update(rows)  # a question is scored once per setup, in its own format
    sizes = load("minilm_document_sizes", lambda r: f"{r['method']} {r['size']}/{r['overlap']}")
    text, image = {}, {}
    for model in ("minilm", "gemma2", "jinaclip2"):
        text.update(load(f"text_embedder_{model}", lambda r: r["model"]))
    for model in ("gemma2", "jinaclip2"):
        image.update(load(f"image_embedder_{model}", lambda r: r["model"]))
    search = load("search", lambda r: r["search"])
    return {"1 chunking": (merged, "slides fixed, documents fixed",
                           ["slides recursive, documents fixed", "slides slide, documents fixed",
                            "slides fixed, documents recursive", "slides recursive, documents recursive",
                            "slides slide, documents recursive"]),
            "1b document sizes": (sizes, "recursive 200/30",
                                  ["fixed 200/30", "fixed 100/15", "recursive 100/15", "fixed 150/22", "recursive 150/22"]),
            "2a text embedder": (text, "minilm", ["gemma2", "jinaclip2"]),
            "2b image embedder": (image, "gemma2", ["jinaclip2"]),
            "3 search": (search, "dense", ["hybrid", "hybrid_rerank", "hybrid_rerank_merged"]),
            "3b slide merging": (search, "hybrid_rerank", ["hybrid_rerank_merged"])}


def main():
    summary, paired = [], []
    for stage, (results, reference, challengers) in stages().items():
        for (setup, score), rows in results.items():
            for group, keep in GROUPS.items():
                for split, part in SPLITS.items():
                    picked = [r for r in rows.values() if keep(r) and part(r)]
                    if picked:
                        summary.append({"stage": stage, "setup": setup, "score": score, "group": group, "split": split,
                                        "questions": len(picked),
                                        **{m.replace("ev3_rr", "mrr"): round(mean(r[m] for r in picked), 3) for m in METRICS}})
        for challenger in challengers:
            for score in {s for _, s in results}:
                base, other = results.get((reference, score)), results.get((challenger, score))
                if not base or not other:
                    continue
                for group, keep in GROUPS.items():
                    for split in ("dev", "test"):
                        ids = [i for i, r in base.items() if r["split"] == split and keep(r)]
                        for metric in ("ev3_recall", "ev3_rr"):
                            if ids:
                                paired.append({"stage": stage, "challenger": challenger, "reference": reference,
                                               "score": score, "group": group, "split": split,
                                               "metric": metric.replace("ev3_rr", "mrr"), "questions": len(ids),
                                               **mc.paired_gain(base, other, ids, metric)})
    ev.write_csv(common.OUT / "summary_all_stages.csv", summary)
    ev.write_csv(common.OUT / "paired_gains.csv", paired)
    print(f"Saved {common.OUT / 'summary_all_stages.csv'} and {common.OUT / 'paired_gains.csv'}")


if __name__ == "__main__":
    main()
