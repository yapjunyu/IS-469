"""Step 1: compare the chunking methods in chunk.py, everything else as in the baseline
(Gemma 2, dense, text only).

    slides     slide  vs  fixed  vs  recursive
    documents          fixed  vs  recursive        ("slide" only applies to slide decks)

fixed and recursive use chunk.py's defaults: 200 words, 30-word overlap.
While the slide chunking varies, the documents stay at the baseline (fixed), and vice versa.
Every run searches one index over the whole corpus; only that format's questions are scored.

Scored as in eval_retrieval.py. There are no image chunks in this step, so the evidence scores cover
text questions only (an image question needs an image chunk); the page scores cover every question.
The winner is picked on the dev questions (evidence recall@3, then MRR); test is what to report.

    .venv/bin/python eval/run_chunking.py
"""
import csv
import hashlib
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]

import eval_retrieval as ev  # noqa: E402  (load_gold, evaluate, summarise, mean, write_csv)
import run_baseline as rb  # noqa: E402  (corpus_files)
from tutor.retrieval.chunk import chunk_pages  # noqa: E402
from tutor.retrieval.clean import clean_pages  # noqa: E402
from tutor.retrieval.embed import Embedder  # noqa: E402
from tutor.retrieval.extract import extract, file_sha256  # noqa: E402
from tutor.retrieval.search import DenseIndex  # noqa: E402

BASELINE = "fixed"
METHODS = {"slides": ("slide", "fixed", "recursive"), "documents": ("fixed", "recursive")}
CACHE = ROOT / "data" / "embed_cache"
BATCH = {"text": 64, "image": 16}  # vectors saved to disk after each batch, so a stopped run can resume


def settings(method: str) -> dict:
    """chunk_pages arguments for one method (chunk_size / chunk_overlap left at chunk.py's defaults)."""
    if method == "slide":
        return dict(slide_method="slide")
    return dict(method=method, slide_method=method)


class CachedEmbed:
    """Embedder that saves every vector to disk, so each text or picture is embedded once per model,
    across all runs. Also records how long each vector took: take_used_s() gives the embedding time
    of the vectors used since the last call, as if none had been cached."""

    def __init__(self, embed):
        self.embed, self.key, self.used_s, self.memory = embed, embed.key, 0.0, {}
        CACHE.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE / f"{embed.key}.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB, seconds REAL)")

    def _load(self, key):
        if key not in self.memory:
            row = self.db.execute("SELECT vector, seconds FROM vectors WHERE key = ?", (key,)).fetchone()
            if row:
                self.memory[key] = (np.frombuffer(row[0], dtype=np.float32), row[1])
        return self.memory.get(key)

    def _vectors(self, keys, inputs, encode, batch):
        missing = {k: i for k, i in zip(keys, inputs) if self._load(k) is None}
        todo = list(missing)
        for start in range(0, len(todo), batch):
            part = todo[start:start + batch]
            began = time.perf_counter()
            vectors = np.asarray(encode([missing[k] for k in part]), dtype=np.float32)
            seconds = (time.perf_counter() - began) / len(part)
            self.db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                                [(k, v.tobytes(), seconds) for k, v in zip(part, vectors)])
            self.db.commit()
            self.memory.update({k: (v, seconds) for k, v in zip(part, vectors)})
        return [self.memory[k] for k in keys]

    def __call__(self, texts):
        keys = ["d" + hashlib.sha256(t.encode()).hexdigest() for t in texts]
        found = self._vectors(keys, texts, self.embed, BATCH["text"])
        self.used_s += sum(s for _, s in found)
        return [v for v, _ in found]

    def embed_images(self, paths):
        keys = ["i" + hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
        found = self._vectors(keys, paths, self.embed.embed_images, BATCH["image"])
        self.used_s += sum(s for _, s in found)
        return [v for v, _ in found]

    def embed_query(self, text):
        key = "q" + hashlib.sha256(text.encode()).hexdigest()
        return self._vectors([key], [text], lambda t: [self.embed.embed_query(t[0])], 1)[0][0]

    def take_used_s(self) -> float:
        used, self.used_s = self.used_s, 0.0
        return used


def winner(fmt: str) -> str:
    """The method saved as the winner for one format ('slides' or 'documents')."""
    with (rb.OUT / "chunking.csv").open() as f:
        return next(r["method"] for r in csv.DictReader(f) if r["format"] == fmt and r["winner"] == "True")


def main():
    files = rb.corpus_files()
    pages = {f: clean_pages(extract(f, images=False)) for f in files}
    hashes = {f: file_sha256(f) for f in files}
    kinds = {f: p[0].kind for f, p in pages.items()}
    questions = ev.load_gold(rb.GOLD)
    n_dev = sum(q["split"] == "dev" for q in questions)
    print(f"{len(files)} files, {len(questions)} questions ({n_dev} dev / {len(questions) - n_dev} test)\n", flush=True)

    embed = CachedEmbed(Embedder("gemma2"))
    summary, per_question = [], []
    for fmt, kind, corpus in (("slides", "slide", "slides"), ("documents", "page", "longdoc")):
        qs = [q for q in questions if q["corpus"] == corpus]
        text_qs = [q for q in qs if q["evidence_type"] == "text"]
        print(f"=== {fmt}: {len(qs)} questions, {len(text_qs)} text", flush=True)
        rows = []
        for method in METHODS[fmt]:
            chunks, spans = [], []
            for f in files:
                use = settings(method) if kinds[f] == kind else settings(BASELINE)
                file_chunks = chunk_pages(pages[f], hashes[f], **use)
                chunks += file_chunks
                if kinds[f] == kind:
                    spans += [c.page_end - c.page + 1 for c in file_chunks]
            index = DenseIndex(chunks, embed)
            index_s = embed.take_used_s()

            # a quote no chunk contains can never be found: the chunking cut through it
            texts = [ev.normalise(c.text) for c in chunks]
            quotes = [quote for q in text_qs for quote in q["quotes"]]
            reachable = sum(any(quote in t for t in texts) for quote in quotes) / len(quotes)

            labels = {"format": fmt, "method": method}
            scored = ev.evaluate(index, qs, labels)
            per_question += scored
            extra = {"chunks": len(spans), "pages_per_chunk": round(sum(spans) / len(spans), 2),
                     "index_s": round(index_s, 1), "quotes_reachable": round(reachable, 3)}
            table = ev.summarise(scored, qs, labels, extra, {"all": lambda q: True, "text_q": lambda q: q["evidence_type"] == "text"})
            everything, text = table
            # the page scores use every question, the evidence scores the text questions
            row = {**everything, **{key: text[key] for key in text if key.startswith(("ev", "dev_ev", "test_ev"))},
                   "group": "ev=text_q, page=all"}
            rows.append(row)
            print(f"  {method}: {len(spans)} chunks, {row['pages_per_chunk']} pages/chunk, "
                  f"embedding {index_s:.0f}s, quotes reachable {reachable:.1%}\n", flush=True)
        best = max(rows, key=lambda r: (r["dev_ev3_recall"], r["dev_ev_mrr"]))
        for r in rows:
            r["winner"] = r is best
        print(f"  → winner on dev: {best['method']}  (test: evidence recall@3 {best['test_ev3_recall']:.3f}, "
              f"MRR {best['test_ev_mrr']:.3f})\n", flush=True)
        summary += rows

    ev.write_csv(rb.OUT / "chunking.csv", summary)
    ev.write_csv(rb.OUT / "chunking_questions.csv", per_question)
    print(f"Saved {rb.OUT / 'chunking.csv'} and {rb.OUT / 'chunking_questions.csv'}")


if __name__ == "__main__":
    main()
