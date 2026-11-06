"""The corpus, gold questions, results folder and embedding cache that the eval scripts share."""
import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np

from tutor.pipeline import corpus_files  # noqa: F401  (the eval scripts call common.corpus_files)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "embed_cache"
BATCH = {"text": 64, "image": 16}  # vectors saved to disk after each batch, so a stopped run can resume
GOLD = ROOT / "eval" / "gold" / "gold_60_40.csv"
OUT = ROOT / "eval" / "results_60_40"


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


class CachedImageEmbed(CachedEmbed):
    """CachedEmbed that also lets image chunks in; the text vectors are reused by the +images run."""
    supports_images = True
