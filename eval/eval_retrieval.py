"""Compare chunking methods on a corpus's gold questions (plan step 20).

For every chunking setup: chunk all files in the corpus, embed the chunks, run each
gold question through dense search, and score the top-k with Hit@k, Recall@k and MRR.
Search runs over the whole corpus at once, like a student's notebook with several files.

    .venv/bin/python eval/eval_retrieval.py                  # slides corpus
    .venv/bin/python eval/eval_retrieval.py --corpus rules   # another corpus

Gold file columns: id, file, question, gold_slides, type. In gold_slides, ";" separates
alternative slides that each hold the whole answer, and "|" separates the parts of a
multi-part question, each of which must be found for full recall. E.g. "46;44|51".
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402

SIZES = (100, 200, 400)  # words
OVERLAP = 0.15  # share of chunk_size, for fixed and recursive


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", default="slides", help="folder under corpora/ and file under eval/gold/")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    files = sorted(p for p in (ROOT / "corpora" / args.corpus).iterdir()
                   if p.suffix.lower() in (".pdf", ".pptx") and not p.name.startswith("~$"))
    questions = load_gold(ROOT / "eval" / "gold" / f"{args.corpus}.csv")
    pages = {f.name: clean_pages(extract(f)) for f in files}
    hashes = {f.name: file_sha256(f) for f in files}
    is_slides = all(p[0].kind == "slide" for p in pages.values())
    embed = Embedder()
    print(f"{args.corpus}: {len(files)} files, {len(questions)} questions, embedder {embed.model_name} "
          f"(reads {embed.max_tokens} tokens)\n")

    summary, per_question = [], []
    for name, settings in setups(is_slides):
        start = time.perf_counter()
        chunks = []
        for file_name, file_pages in pages.items():
            chunks += chunk_pages(file_pages, hashes[file_name], embed=embed, **settings)
        index = DenseIndex(chunks, embed)
        index_s = time.perf_counter() - start

        scores = [score(q, index.search(q["question"], args.k)) for q in questions]
        row = {
            "setup": name,
            f"hit@{args.k}": mean(s["hit"] for s in scores),
            f"recall@{args.k}": mean(s["recall"] for s in scores),
            "mrr": mean(s["rr"] for s in scores),
            "chunks": len(chunks),
            "avg_words": round(mean(len(c.text.split()) for c in chunks)),
            "over_token_limit": mean(n_tokens(embed, c.text) > embed.max_tokens for c in chunks),
            "index_seconds": round(index_s, 1),
        }
        summary.append(row)
        per_question += [{"setup": name, **{k: q[k] for k in ("id", "type", "question", "gold_slides")}, **s}
                         for q, s in zip(questions, scores)]
        print(f"{name:16s} hit@{args.k} {row[f'hit@{args.k}']:.2f}  recall@{args.k} {row[f'recall@{args.k}']:.2f}  "
              f"mrr {row['mrr']:.2f}  | {row['chunks']:3d} chunks, avg {row['avg_words']:3d} words, "
              f"{row['over_token_limit']:.0%} over token limit")

    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    write_csv(out / f"chunking_{args.corpus}.csv", summary)
    write_csv(out / f"chunking_{args.corpus}_questions.csv", per_question)
    print(f"\nSaved {out / f'chunking_{args.corpus}.csv'} and the per-question breakdown.")


def setups(is_slides: bool):
    """(name, chunk_pages settings) for every chunking setup to compare."""
    if is_slides:
        yield "slide", dict(slide_method="slide")
    for method in ("fixed", "recursive", "semantic"):
        for size in SIZES:
            yield (f"{method}-{size}",
                   dict(method=method, slide_method=method, chunk_size=size,
                        chunk_overlap=round(size * OVERLAP)))


def load_gold(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        questions = list(csv.DictReader(f))
    for q in questions:
        q["parts"] = [{int(s) for s in part.split(";")} for part in q["gold_slides"].split("|")]
    return questions


def score(question: dict, passages) -> dict:
    """Which parts of the answer were found in the top-k, and the rank of the first hit."""

    def answers(passage, part) -> bool:
        return passage.file == question["file"] and any(passage.page <= s <= passage.page_end for s in part)

    found = sum(any(answers(p, part) for p in passages) for part in question["parts"])
    first = next((rank for rank, p in enumerate(passages, start=1)
                  if any(answers(p, part) for part in question["parts"])), None)
    return {
        "hit": int(found > 0),
        "recall": found / len(question["parts"]),
        "rr": 1 / first if first else 0.0,
        "retrieved": " ; ".join(p.citation() for p in passages),
    }


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


if __name__ == "__main__":
    main()
