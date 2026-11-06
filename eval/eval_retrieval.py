"""The scoring every eval/run_*.py script shares.

Two scores, at k = 3 and 5:
  evidence (main)  the top k chunks. A chunk is relevant if it covers a gold page of the question
                   and contains that part's evidence quote; for an image question, if it is an
                   image chunk from a gold page. Works for any chunking, because it checks the
                   text and not a chunk id.
  page             the first k distinct pages the ranked chunks cover. Every method gets k pages,
                   but the right page doesn't prove the right text.
recall    = share of the question's parts found
precision = share of the k returned chunks (or pages) that are relevant
F1        = 2PR / (P + R), per question, then averaged
MRR       = 1 / rank of the first relevant chunk (or page), within the top 3
"""
import csv
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutor.retrieval.embed import Embedder  # noqa: E402

KS = (3, 5)
GROUPS = {"all": lambda q: True,
          "slides": lambda q: q["corpus"] == "slides", "documents": lambda q: q["corpus"] == "longdoc",
          "text_q": lambda q: q["evidence_type"] == "text", "image_q": lambda q: q["evidence_type"] == "image",
          "single_part": lambda q: q["n_parts"] == "1", "multi_part": lambda q: q["n_parts"] != "1"}


def load_gold(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        questions = list(csv.DictReader(f))
    questions = [q for q in questions if q["gold_slides"]]  # out-of-scope questions have no gold pages: they test refusal, not retrieval
    for q in questions:
        q["parts"] = [{int(s) for s in part.split(";")} for part in q["gold_slides"].split("|")]
        # also_gold: pages in another file that answer a part too, e.g. "2@Lesson6-Reliability.pptx:23"
        q["also"] = [set() for _ in q["parts"]]
        for item in filter(None, (q.get("also_gold") or "").split(";")):
            part, place = item.split("@")
            name, page = place.rsplit(":", 1)
            q["also"][int(part) - 1].add((name, int(page)))
        q["quotes"] = [normalise(e) for e in q["evidence"].split("||")] if q.get("evidence_type") == "text" else []
    return questions


@lru_cache(maxsize=None)
def normalise(text: str) -> str:
    """Lowercase words and numbers only, so line breaks and punctuation can't stop a quote matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def gold_page(q: dict, file: str, page: int, part: int) -> bool:
    return (file == q["file"] and page in q["parts"][part]) or (file, page) in q["also"][part]


def relevant(q: dict, p, part: int) -> bool:
    """Does chunk p hold this part of the answer? It must be on a gold page and hold the evidence."""
    pages = range(p.page, p.page_end + 1)
    if any((p.file, page) in q["also"][part] for page in pages):  # no quote recorded for another file's page
        return True
    if not any(gold_page(q, p.file, page, part) for page in pages):
        return False
    if q["evidence_type"] == "image":
        return bool(p.image_path)
    return q["quotes"][part] in normalise(p.text)


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def score_evidence(q: dict, passages) -> dict:
    parts = range(len(q["parts"]))
    good = [any(relevant(q, p, i) for i in parts) for p in passages]
    recall = sum(any(relevant(q, p, i) for p in passages) for i in parts) / len(q["parts"])
    prec = sum(good) / len(passages) if passages else 0.0
    first = next((rank for rank, g in enumerate(good, start=1) if g), None)
    return {"recall": recall, "precision": prec, "f1": f1(prec, recall), "rr": 1 / first if first else 0.0}


def top_pages(index, question: str, n: int) -> list[tuple[str, int]]:
    """Walk the ranked chunks, expand each into the pages it covers, keep the first n distinct pages."""
    pages, seen = [], set()
    for p in index.search(question, 50):
        for page in range(p.page, p.page_end + 1):
            if (p.file, page) not in seen:
                seen.add((p.file, page))
                pages.append((p.file, page))
            if len(pages) == n:
                return pages
    return pages


def score_pages(q: dict, pages) -> dict:
    parts = range(len(q["parts"]))
    good = [any(gold_page(q, *key, i) for i in parts) for key in pages]
    recall = sum(any(gold_page(q, *key, i) for key in pages) for i in parts) / len(q["parts"])
    prec = sum(good) / len(pages) if pages else 0.0
    first = next((rank for rank, g in enumerate(good, start=1) if g), None)
    return {"recall": recall, "precision": prec, "f1": f1(prec, recall), "rr": 1 / first if first else 0.0}


def evaluate(index, questions, labels: dict) -> list[dict]:
    """Evidence and page scores for every question (one ranking per question, cut at each k)."""
    scored = []
    for q in questions:
        chunks, pages = index.search(q["question"], max(KS)), top_pages(index, q["question"], max(KS))
        r = {**labels, "id": q["id"], "split": q["split"], "corpus": q["corpus"], "collection": q["collection"],
             "evidence_type": q["evidence_type"], "n_parts": q["n_parts"]}
        for k in KS:
            r.update({f"ev{k}_{m}": v for m, v in score_evidence(q, chunks[:k]).items()})
            r.update({f"page{k}_{m}": v for m, v in score_pages(q, pages[:k]).items()})
        scored.append(r)
    return scored


def summarise(scored, questions, labels: dict, extra: dict, groups=None) -> list[dict]:
    """One row per question group: averages, plus dev and test for recall, precision and F1 @3, and MRR."""
    by_id = {q["id"]: q for q in questions}
    rows = []
    for group, keep in (groups or GROUPS).items():
        sub = [r for r in scored if keep(by_id[r["id"]])]
        row = {**labels, "group": group, "questions": len(sub), **extra}
        for level in ("ev", "page"):
            for k in KS:
                for metric in ("recall", "precision", "f1"):
                    row[f"{level}{k}_{metric}"] = mean(r[f"{level}{k}_{metric}"] for r in sub)
            row[f"{level}_mrr"] = mean(r[f"{level}3_rr"] for r in sub)
            for split in ("dev", "test"):
                part = [r for r in sub if r["split"] == split]
                for metric in ("recall", "precision", "f1"):
                    row[f"{split}_{level}3_{metric}"] = mean(r[f"{level}3_{metric}"] for r in part)
                row[f"{split}_{level}_mrr"] = mean(r[f"{level}3_rr"] for r in part)
        rows.append(row)
        print(f"  {' '.join(str(v) for v in labels.values()):20s} {group:11s} n={len(sub):3d} | evidence@3 R {row['ev3_recall']:.3f} "
              f"P {row['ev3_precision']:.3f} F1 {row['ev3_f1']:.3f} MRR {row['ev_mrr']:.3f} | page@3 R {row['page3_recall']:.3f} "
              f"MRR {row['page_mrr']:.3f}", flush=True)
    return rows


def n_tokens(embed: Embedder, text: str) -> int:
    return len(embed.model.tokenizer.tokenize(text)) + 2  # + the [CLS] and [SEP] markers


def mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 3) if values else 0.0


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

