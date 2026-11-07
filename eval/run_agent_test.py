"""Agent scorecard: each request run with search-again on and off, scored from its tool calls and the learner model.

Requests: the hand-checked rows of eval/agent/requests.csv (quiz_me, quiz_answer, remember, off_topic), plus
30 content questions drawn from the gold set (seed 469, none of the 75 llm_eval ones), which need no
checking because the gold set already says which slide or page answers them.

    .venv/Scripts/python.exe eval/run_agent_test.py --limit 1   # first item of each type, to try it
    .venv/Scripts/python.exe eval/run_agent_test.py

Saves eval/results_agent/agent_test.jsonl (one row per request and arm; a rerun skips finished rows).

Task success, per type:
  content      search_notes ran, the answer cites a passage, and there is no faithfulness warning
  quiz_me      make_quiz ran with the expected concept and returned 3 to 5 questions
  quiz_answer  grade_answer ran, its verdict matches the planted answer, and after a wrong answer
               the reply does not give the answer away (the Scorer checks, see issue #17)
  remember     remember_fact stored a fact that cites a passage the tutor read this turn
  off_topic    no search_notes call and no fact stored
Tool-call accuracy, per type (right tool, right inputs):
  content      search_notes found a gold passage
  quiz_me      make_quiz with the expected concept
  quiz_answer  grade_answer with the quiz's passage ids
  remember     remember_fact with the expected concept
  off_topic    no tool call at all
"""
import argparse
import random
from types import SimpleNamespace

import agent_common as ac
import eval_retrieval as ev
from agent.memory import learner

ARMS = {"search_again_on": {"search_again": True}, "search_again_off": {"search_again": False}}
N_CONTENT = 30
LEAK_PROMPT = ("A student answered a quiz question wrongly. Below are the question, the correct answer and the tutor's "
               "reply. Does the reply state or paraphrase the correct answer, instead of only hinting at it?")


def content_requests():
    gold = [q for q in ev.load_gold(ac.ROOT / "eval" / "gold" / "gold_60_40.csv") if q.get("llm_eval") != "1"]
    return [{"id": q["id"], "type": "content", "message": q["question"], "gold": q}
            for q in random.Random(469).sample(gold, N_CONTENT)]


def quiz_history(row):
    """The two messages before a quiz answer: the request and the tutor's question, with its passage ids
    in the text, as the app keeps them in history for grade_answer."""
    ids = [p["id"] for p in ac.passages_at(row["quiz_file"], int(row["quiz_page"]))]
    assert ids, f"{row['id']}: no passage at {row['quiz_file']} page {row['quiz_page']}"
    concept = row["expected_concept"] or "this topic"
    return ids, [{"role": "user", "content": f"Quiz me on {concept}."},
                 {"role": "assistant", "content": f"Question 1: {row['quiz_question']} (passages: {', '.join(ids)})"}]


def gold_hit(q, passages):
    as_chunks = [SimpleNamespace(**{**p, "page_end": p.get("page_end") or p["page"]}) for p in passages]
    return any(ev.relevant(q, p, part) for p in as_chunks for part in range(len(q["parts"])))


def score(req, out, facts, quiz_ids):
    calls = {}
    for c in out["calls"]:
        calls.setdefault(c["tool"], []).append(c)
    t, read = req["type"], {p["id"] for p in out["passages"]}
    if t == "content":
        tool_ok = "search_notes" in calls and gold_hit(req["gold"], out["passages"])
        return "search_notes" in calls and bool(out["citations"]) and not out["warning"], tool_ok, ""
    if t == "quiz_me":
        want = learner.norm(req["expected_concept"])
        quizzes = [c for c in calls.get("make_quiz", []) if learner.norm(str(c["inputs"].get("concept", ""))) == want]
        sizes = [len(c["result"].get("questions", [])) for c in quizzes]
        return any(3 <= n <= 5 for n in sizes), bool(quizzes), ""
    if t == "quiz_answer":
        grades = calls.get("grade_answer", [])
        tool_ok = any(set(c["inputs"].get("passage_ids") or []) <= set(quiz_ids) and c["inputs"].get("passage_ids")
                      for c in grades)
        verdicts = [c["result"]["correct"] for c in grades if "correct" in c["result"]]
        right = bool(verdicts) and verdicts[-1] == (req["expected_correct"] == "1")
        leak, why = False, ""
        if right and req["expected_correct"] == "0":
            leak, why = ac.scorer_yes(LEAK_PROMPT, f"Question: {req['quiz_question']}\nCorrect answer: "
                                                   f"{req['reference']}\nTutor's reply: {out['answer']}")
        return right and not leak, tool_ok, (f"gave the answer away: {why}" if leak else "")
    if t == "remember":
        tool_ok = any(learner.norm(str(c["inputs"].get("concept", ""))) == learner.norm(req["expected_concept"])
                      for c in calls.get("remember_fact", []))
        return any(f["passage_id"] in read for f in facts), tool_ok, ""
    if t == "off_topic":
        return "search_notes" not in calls and not facts, not out["calls"], ""
    raise ValueError(f"unknown request type {t!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="only the first N requests of each type")
    args = parser.parse_args()

    gold = {q["id"]: q for q in ev.load_gold(ac.ROOT / "eval" / "gold" / "gold_60_40.csv")}
    requests = content_requests() + ac.load_checked("requests")
    for r in requests:  # quiz answers are graded against the gold reference answer
        source = r.get("notes", "").removeprefix("from gold ").split(";")[0]
        r["reference"] = gold[source]["reference_answer"] if source in gold else ""
    if args.limit:
        seen = {}
        requests = [r for r in requests if seen.setdefault(r["type"], []).append(r) or len(seen[r["type"]]) <= args.limit]

    results = ac.Results("agent_test")
    for req in requests:
        for arm, switches in ARMS.items():
            if results.has(req["id"], arm):
                continue
            ac.reset_learner(ac.STUDENT)
            quiz_ids, history = quiz_history(req) if req["type"] == "quiz_answer" else ([], None)
            out = ac.run("agent_test", arm, req["message"], history=history, **switches)
            facts = learner.read_all(ac.STUDENT)["facts"]
            success, tool_ok, note = score(req, out, facts, quiz_ids)
            results.add({"id": req["id"], "arm": arm, "type": req["type"], "message": req["message"],
                         "answer": out["answer"], "citations": out["citations"], "warning": out["warning"],
                         "faithfulness": out["faithfulness"], "facts": facts,
                         "tools": [{"tool": c["tool"], "inputs": c["inputs"], "error": c["result"].get("error")}
                                   for c in out["calls"]],
                         **{k: out[k] for k in ("steps", "cap_hit", "searches", "tokens", "duration_ms")},
                         "success": success, "tool_ok": tool_ok, "note": note})
            print(f"{req['id']:7} {arm:17} success={success!s:5} tool_ok={tool_ok!s:5} steps={out['steps']} "
                  f"searches={out['searches']} tokens={sum(out['tokens'].values())} {out['duration_ms'] / 1000:.0f}s {note}",
                  flush=True)
    print(f"Saved {results.path}")


if __name__ == "__main__":
    main()
