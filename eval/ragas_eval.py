"""Score a saved generation run with RAGAS (run eval/run_generation.py first).

Reads eval/results_60_40/generation_<name>.json (question, retrieved text and pictures, answer, reference
answer) and asks a judge LLM to score each question on four metrics:

  faithfulness         is the answer supported by the retrieved text and pictures? (1 or 0)
  factual_correctness  does the answer agree with the reference answer?
  context_precision    are the passages that matter ranked near the top? (text questions)
  context_recall       do the retrieved passages cover what the reference says? (text questions)

The two context metrics only read text, so they are skipped for image questions instead of being
scored without the figure. The judge is a different model from the one that wrote the answers, so
it does not grade its own work. Judge replies are cached in data/ragas_cache, so a rerun is free.

    .venv/bin/python eval/ragas_eval.py sample --limit 3   # try a few first, to see the token cost
    .venv/bin/python eval/ragas_eval.py sample             # all questions in the run
    .venv/bin/python eval/ragas_eval.py --agent agent_search_again_on   # an agent run from run_agent_generation.py

Both runs are scored by claude-haiku-5-5 on OpenCode Go. For a pipeline run it is a family apart from the
generator (gpt-6-luna). --agent reads eval/results_agent instead, where it is a family apart from the agent's
tutor (DeepSeek) and its faithfulness judge (GPT).
"""
import argparse
import asyncio
import csv
import json
import os
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall, FactualCorrectness, MultiModalFaithfulness

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results_60_40"
JUDGE_MODEL = "claude-haiku-5-5"  # reads pictures; a different model from generator.py (gpt-6-luna)
JUDGE_MAX_TOKENS = 8192  # room for the longest RAGAS reply (claim lists); unused tokens cost nothing
AGENT_RESULTS = ROOT / "eval" / "results_agent"
GOLD = ROOT / "eval" / "gold" / "gold_60_40.csv"
METRICS = ("faithfulness", "factual_correctness", "context_precision", "context_recall")
SCORER_MODEL = "claude-haiku-5-5"  # eval/agent_common.py SCORER, without the opencode-go/ routing prefix
# OpenCode Go's OpenAI-compatible endpoint, which RAGAS expects. Go also serves Claude through it, with the
# image_url parts MultiModalFaithfulness sends. As src/agent/s1_llm.py, Go rejects requests without a session id
OPENCODE_URL = "https://opencode.ai/inference/go/openai/v1"

tokens_used = 0  # counted from every judge reply, to see what a run costs


async def _count_tokens(response: httpx.Response):
    global tokens_used
    if response.url.path.endswith("/chat/completions") and response.status_code == 200:
        await response.aread()
        tokens_used += response.json().get("usage", {}).get("total_tokens", 0)


def write_csvs(results_dir: Path, run: str, results: list[dict]) -> None:
    """Pipeline runs only. ragas_<run>_questions.csv: one row per question with its split, format and four scores.
    ragas_<run>.csv: the mean and count of each metric, overall and by split and evidence type."""
    gold = {g["id"]: g for g in csv.DictReader(GOLD.open(newline=""))}
    rows = [{"id": r["id"], **{k: gold.get(r["id"], {}).get(k, "") for k in ("split", "evidence_type", "corpus", "collection")},
             **{m: r[m] for m in METRICS}, "errors": "; ".join(f"{k}: {v}" for k, v in r["errors"].items())}
            for r in results]
    with (results_dir / f"ragas_{run}_questions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = [("all", "all"), ("dev", "all"), ("test", "all")] + [(s, e) for s in ("all", "dev", "test") for e in ("text", "image")]
    with (results_dir / f"ragas_{run}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "evidence_type", "questions"] + [c for m in METRICS for c in (m, f"{m}_n")])
        for split, kind in groups:
            part = [r for r in rows if split in ("all", r["split"]) and kind in ("all", r["evidence_type"])]
            line = [split, kind, len(part)]
            for m in METRICS:
                vals = [r[m] for r in part if r[m] is not None]  # a skipped or failed metric is left out
                line += [round(sum(vals) / len(vals), 3) if vals else "", len(vals)]
            w.writerow(line)


def make_judge(agent=False):
    """Both runs score with Claude Haiku on OpenCode Go; each keeps its own cache, so agent and pipeline runs don't mix."""
    http = httpx.AsyncClient(event_hooks={"response": [_count_tokens]}, timeout=120)
    client = AsyncOpenAI(api_key=os.environ["OPENCODE_API_KEY"], base_url=OPENCODE_URL, max_retries=8, http_client=http,
                         default_headers={"User-Agent": "is-469-tutor", "x-opencode-session": str(uuid.uuid4())})
    model, cache = (SCORER_MODEL, "ragas_cache_scorer") if agent else (JUDGE_MODEL, "ragas_cache")
    return llm_factory(model, client=client, cache=DiskCacheBackend(str(ROOT / "data" / cache)),
                       temperature=0.0, max_tokens=JUDGE_MAX_TOKENS)


async def score_row(row: dict, metrics: dict) -> dict:
    """All four scores for one question; a metric that fails is recorded as None with its error."""
    contexts = row["retrieved_contexts"]
    inputs = {
        "faithfulness": dict(response=row["response"], retrieved_contexts=contexts + row["retrieved_images"]),
        "factual_correctness": dict(response=row["response"], reference=row["reference"]),
        "context_precision": dict(user_input=row["user_input"], reference=row["reference"], retrieved_contexts=contexts),
        "context_recall": dict(user_input=row["user_input"], reference=row["reference"], retrieved_contexts=contexts),
    }
    if row["evidence_type"] == "image":  # the answer is in a figure these two metrics can't read
        inputs.pop("context_precision")
        inputs.pop("context_recall")
    scores, errors = {}, {}
    for name, metric in metrics.items():  # one at a time, to stay inside the per-minute limit
        if name not in inputs:
            scores[name] = None
            continue
        try:
            scores[name] = float((await metric.ascore(**inputs[name])).value)
        except Exception as e:
            scores[name], errors[name] = None, f"{type(e).__name__}: {e}"[:300]
    return {"id": row["id"], **scores, "errors": errors}


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="the run to score: generation_<run>.json")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--agent", action="store_true", help="an agent run in eval/results_agent, scored by the Scorer")
    args = parser.parse_args()
    results_dir = AGENT_RESULTS if args.agent else RESULTS

    load_dotenv(ROOT / ".env")
    # shortcut: a pipeline run is read in the locale's encoding, as run_generation.py writes it, so one saved on another
    # machine can misread, read utf-8 always once run_generation.py writes utf-8
    encoding = "utf-8" if args.agent else None
    rows = [r for r in json.loads((results_dir / f"generation_{args.run}.json").read_text(encoding=encoding))["rows"] if r["response"]][:args.limit]
    judge = make_judge(args.agent)
    metrics = {"faithfulness": MultiModalFaithfulness(llm=judge), "factual_correctness": FactualCorrectness(llm=judge),
               "context_precision": ContextPrecision(llm=judge), "context_recall": ContextRecall(llm=judge)}

    results = []
    for row in rows:
        before = tokens_used
        result = await score_row(row, metrics)
        results.append(result)
        shown = "  ".join(f"{k[:12]} {'-' if result[k] is None else f'{result[k]:.2f}'}" for k in metrics)
        print(f"{row['id']:7} {shown}   ({tokens_used - before} tokens)", flush=True)
        for name, err in result["errors"].items():
            print(f"        {name} failed: {err}")

    scored = {k: [r[k] for r in results if r[k] is not None] for k in metrics}
    means = {k: round(sum(v) / len(v), 3) if v else None for k, v in scored.items()}  # None: no question scored
    out = results_dir / f"ragas_{args.run}.json"
    out.write_text(json.dumps({"judge": SCORER_MODEL if args.agent else JUDGE_MODEL, "means": means, "scored": {k: len(v) for k, v in scored.items()}, "tokens": tokens_used, "rows": results}, indent=1))
    if not args.agent and results:
        write_csvs(results_dir, args.run, results)
    print("\nmean  " + "  ".join(f"{k} {'-' if v is None else f'{v:.2f}'} (n={len(scored[k])})" for k, v in means.items()))
    print(f"{len(results)} questions, {tokens_used} judge tokens this run "
          f"(cached replies cost nothing), saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
