"""What the agent experiments share: the eval-gold notebook, a fresh learner model, and a recorded turn.

    .venv/Scripts/python.exe eval/agent_common.py build    # add every corpora file to the eval-gold notebook, once

Every experiment calls run(): one run_turn with the app's defaults and the switches the experiment flips,
its own trace file and LLM cache per arm, and everything the scorers need (tool calls with full results,
the passages the tutor read, the final trace row).
"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
os.chdir(ROOT)  # learner DB and trace paths are relative to the repo root

from agent import s1_llm, s2_tools, s3_loop, s4_trace  # noqa: E402
from agent.memory import learner  # noqa: E402
from tutor import notebook  # noqa: E402
from tutor.pipeline import corpus_files  # noqa: E402

STUDENT = "eval-gold"
AGENT = ROOT / "eval" / "agent"  # the hand-checked input CSVs
OUT = ROOT / "eval" / "results_agent"
CACHE = ROOT / "data" / "llm_cache_eval"  # one folder per experiment and arm, and one for the Scorer
MEMORY_TABLES = ("mastery", "wrong_beliefs", "facts", "session_notes")  # concepts stay: they describe the notebook


def build(student=STUDENT, files=None):
    for f in files or corpus_files():
        print(f"{f.name}: {notebook.add_file(student, f)['status']}", flush=True)
    print(f"{len(learner.concepts(student))} concepts in {student}")


def reset_learner(student):
    """Empty everything the tutor remembers about the student, keeping the notebook and its concept list."""
    memory = learner.read_all(student)
    for table in MEMORY_TABLES:
        for row in memory[table]:
            learner.delete(student, table, row[learner.TABLE_KEYS[table]])


def passages_at(file, page, student=STUDENT):
    """The notebook's passages from this file that cover this page or slide."""
    return [notebook._passage(p) for p in notebook._store(student).chunks
            if p.file == file and p.page <= page <= p.page_end]


def run(experiment, arm, message, student=STUDENT, history=None, **switches):
    """One recorded turn. Each arm gets its own LLM cache, so arms never share a reply (that would make
    the second arm look faster) and a rerun of the same arm resumes for free."""
    s1_llm.CACHE_DIR = CACHE / f"{experiment}-{arm}"
    path = s4_trace.TRACE_PATH = OUT / f"trace_{experiment}_{arm}.jsonl"
    start = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
    out = s3_loop.run_turn(message, student, 1, history=history, search_fn=notebook.search,
                           passages_fn=notebook.get_passages, memory="learner", **switches)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[start:]]  # this turn's
    final = rows[-1]
    assert final["kind"] == "final", "run_turn always ends with its final row"
    read = {p["id"]: p for c in out["tool_calls"] if c["tool"] == "search_notes"
            for p in c["result"].get("passages", [])}  # what search_notes gave the tutor
    # shortcut: a rerun of an item that crashed mid-turn replays the arm's cached replies, so its duration_ms is too
    # short; cached_replies > 0 marks such a turn rather than fixing it, upgrade if many items need rerunning
    return {**out, "passages": list(read.values()),
            "cached_replies": sum(r["kind"] == "llm" and r["cached"] for r in rows),
            **{k: final[k] for k in ("steps", "cap_hit", "searches", "tokens", "duration_ms")}}


SCORER = "opencode-go/claude-haiku-5-5"  # a family apart from the tutor (DeepSeek) and the judge (GPT), see GLOSSARY.md
SCORER_TOKENS = {"in": 0, "out": 0}  # every Scorer call's tokens, cached or not, as a turn counts the tutor's; runners zero it per item


def scorer_yes(instructions, material):
    """Ask the Scorer one yes/no question about some material. Returns (bool, its one-line reason).
    Its tokens go to SCORER_TOKENS, not to any trace: Scorer calls run outside a student turn."""
    saved, s1_llm.CACHE_DIR = s1_llm.CACHE_DIR, CACHE / "scorer"
    try:
        reply = s1_llm.chat([{"role": "system", "content": instructions + "\nReply with JSON only: "
                                                       '{"yes": true or false, "reason": "one sentence"}'},
                             {"role": "user", "content": material}], SCORER, max_tokens=1000)  # 300 cut a long reason mid-JSON
    finally:
        s1_llm.CACHE_DIR = saved
    for k in SCORER_TOKENS:
        SCORER_TOKENS[k] += reply["usage"][k]
    text = reply["message"].get("content") or ""
    verdict = s2_tools.reply_json(text, "{")
    if not (isinstance(verdict, dict) and isinstance(verdict.get("yes"), bool)):
        raise ValueError(f"Scorer reply was not the JSON asked for: {text[:200]!r}")
    return verdict["yes"], str(verdict.get("reason", ""))


def load_checked(name):
    """Rows of eval/agent/<name>.csv that a person marked checked=1; the rest are skipped, never run."""
    with open(AGENT / f"{name}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    checked = [r for r in rows if r.get("checked") == "1"]
    if len(checked) < len(rows):
        print(f"{name}.csv: skipping {len(rows) - len(checked)} unchecked rows")
    return checked


class Results:
    """A JSONL of finished items, keyed by (id, arm), so a stopped run resumes and keeps its first latencies."""

    def __init__(self, experiment):
        OUT.mkdir(parents=True, exist_ok=True)
        self.path = OUT / f"{experiment}.jsonl"
        self.done = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.done[(row["id"], row["arm"])] = row

    def has(self, item_id, arm):
        return (item_id, arm) in self.done

    def add(self, row):
        self.done[(row["id"], row["arm"])] = row
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
    else:
        sys.exit(__doc__)
