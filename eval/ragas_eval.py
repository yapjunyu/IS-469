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
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall, FactualCorrectness, MultiModalFaithfulness

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results_60_40"
JUDGE_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"  # reads pictures; a different model from rag.py
JUDGE_MAX_TOKENS = 8192  # the judge reasons before it replies, and the reasoning counts towards this limit
OPENROUTER_URL = "https://openrouter.ai/api/v1"  # the OpenAI API, which RAGAS expects

tokens_used = 0  # counted from every judge reply, to see how much of the daily quota a run takes


async def _count_tokens(response: httpx.Response):
    global tokens_used
    if response.url.path.endswith("/chat/completions") and response.status_code == 200:
        await response.aread()
        tokens_used += response.json().get("usage", {}).get("total_tokens", 0)


def make_judge():
    client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_URL, max_retries=8,
                         http_client=httpx.AsyncClient(event_hooks={"response": [_count_tokens]}, timeout=120))
    return llm_factory(JUDGE_MODEL, client=client, cache=DiskCacheBackend(str(ROOT / "data" / "ragas_cache")),
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
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    rows = [r for r in json.loads((RESULTS / f"generation_{args.run}.json").read_text())["rows"] if r["response"]][:args.limit]
    judge = make_judge()
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
    out = RESULTS / f"ragas_{args.run}.json"
    out.write_text(json.dumps({"judge": JUDGE_MODEL, "means": means, "scored": {k: len(v) for k, v in scored.items()}, "rows": results}, indent=1))
    print("\nmean  " + "  ".join(f"{k} {'-' if v is None else f'{v:.2f}'} (n={len(scored[k])})" for k, v in means.items()))
    print(f"{len(results)} questions, {tokens_used} judge tokens this run "
          f"(cached replies cost nothing), saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
