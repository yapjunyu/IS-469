"""Answer the 75 llm_eval gold questions with the agent, search-again on and off, ready for ragas_eval.py --agent.

The same questions run_generation.py gives the RAG pipeline, but through run_turn on the eval-gold notebook,
with the app's defaults and a fresh learner model per question. The retrieved contexts are every passage
search_notes gave the tutor that turn.

    .venv/Scripts/python.exe eval/run_agent_generation.py --limit 3
    .venv/Scripts/python.exe eval/run_agent_generation.py
    .venv/Scripts/python.exe eval/ragas_eval.py --agent agent_search_again_on
    .venv/Scripts/python.exe eval/ragas_eval.py --agent agent_search_again_off

Saves eval/results_agent/generation_agent_<arm>.json, rewritten after each question; a rerun resumes.
"""
import argparse
import json

import agent_common as ac
import eval_retrieval as ev

ARMS = {"search_again_on": {"search_again": True}, "search_again_off": {"search_again": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only the first N questions")
    args = parser.parse_args()

    questions = [q for q in ev.load_gold(ac.ROOT / "eval" / "gold" / "gold_60_40.csv") if q.get("llm_eval") == "1"]
    ac.OUT.mkdir(parents=True, exist_ok=True)
    for arm, switches in ARMS.items():
        path = ac.OUT / f"generation_agent_{arm}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"] if path.exists() else []
        done = {r["id"] for r in rows}
        for q in questions[:args.limit]:
            if q["id"] in done:
                continue
            ac.reset_learner(ac.STUDENT)
            out = ac.run("generation", arm, q["question"], **switches)
            rows.append({"id": q["id"], "evidence_type": q["evidence_type"], "user_input": q["question"],
                         "retrieved_contexts": [p["text"] for p in out["passages"] if p["text"].strip()],
                         "retrieved_images": [p["image_path"] for p in out["passages"] if p.get("image_path")],
                         "response": out["answer"], "reference": q["reference_answer"],
                         "retrieved": [f"{p['file']} p{p['page']}" for p in out["passages"]],
                         "warning": out["warning"], "searches": out["searches"], "steps": out["steps"],
                         "tokens": out["tokens"], "duration_ms": out["duration_ms"]})
            print(f"{arm:17} {q['id']:7} {len(out['passages'])} passages, {out['searches']} searches  "
                  f"{out['answer'][:60]!r}", flush=True)
            path.write_text(json.dumps({"rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
