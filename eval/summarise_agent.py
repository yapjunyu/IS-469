"""Tables for every agent experiment, each on/off pair with a paired bootstrap 95% interval, and the failures by cause.

    .venv/Scripts/python.exe eval/summarise_agent.py

Reads what run_agent_test.py, run_agent_generation.py + ragas_eval.py --agent, run_injection.py and
run_poisoning.py saved in eval/results_agent/, skips any experiment not run yet, and writes
eval/results_agent/RESULTS.md. Gains are "on minus off", computed only over items finished in both arms.
"""
import json
from collections import defaultdict
from statistics import mean

import agent_common as ac
from run_minilm_chunking import paired_gain  # the retrieval summary's bootstrap: 5000 rounds, seed 469

OUT = ac.OUT / "RESULTS.md"


def load_jsonl(name):
    path = ac.OUT / f"{name}.jsonl"
    rows = defaultdict(dict)  # arm -> id -> row; a later line for the same item wins
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rows[r["arm"]][r["id"]] = r
    return rows


def numeric(r):
    return {**r, **{k: float(v) for k, v in r.items() if isinstance(v, bool)},
            "total_tokens": sum(r["tokens"].values()) if "tokens" in r else None,
            "latency_s": r["duration_ms"] / 1000 if "duration_ms" in r else None}


def gain_line(on, off, metric, label):
    ids = sorted(set(on) & set(off))
    a = {i: numeric(off[i]) for i in ids}
    b = {i: numeric(on[i]) for i in ids}
    ids = [i for i in ids if a[i].get(metric) is not None and b[i].get(metric) is not None]
    if not ids:
        return f"| {label} | - | - | - | 0 |"
    g = paired_gain(a, b, ids, metric)
    clear = "**" if g["low"] > 0 or g["high"] < 0 else ""
    return (f"| {label} | {mean(a[i][metric] for i in ids):.3f} | {mean(b[i][metric] for i in ids):.3f} | "
            f"{clear}{g['gain']:+.3f} ({g['low']:+.3f} to {g['high']:+.3f}){clear} | {len(ids)} |")


def comparison(title, on, off, metrics, on_name, off_name):
    lines = [f"### {title}", "", f"| Metric | {off_name} | {on_name} | Gain, on minus off (95% CI) | Items |",
             "|---|---|---|---|---|"]
    lines += [gain_line(on, off, m, label) for m, label in metrics]
    return lines + [""]


def agent_test():
    rows = load_jsonl("agent_test")
    on, off = rows.get("search_again_on", {}), rows.get("search_again_off", {})
    if not on and not off:
        return []
    lines = ["## Agent scorecard: search-again on vs off", ""]
    lines += comparison("All requests", on, off, [("success", "Task success"), ("tool_ok", "Tool-call accuracy"),
                                                  ("steps", "Steps per turn"), ("searches", "Searches per turn"),
                                                  ("total_tokens", "Tokens per turn"), ("latency_s", "Latency (s)")],
                        "search-again on", "search-again off")
    lines += ["### By request type", "", "| Type | Arm | Items | Task success | Tool-call accuracy | Steps | Tokens | Latency (s) |",
              "|---|---|---|---|---|---|---|---|"]
    for t in ("content", "quiz_me", "quiz_answer", "remember", "off_topic"):
        for arm, part in (("off", off), ("on", on)):
            picked = [numeric(r) for r in part.values() if r["type"] == t]
            if picked:
                lines.append(f"| {t} | {arm} | {len(picked)} | {mean(r['success'] for r in picked):.2f} | "
                             f"{mean(r['tool_ok'] for r in picked):.2f} | {mean(r['steps'] for r in picked):.1f} | "
                             f"{mean(r['total_tokens'] for r in picked):.0f} | {mean(r['latency_s'] for r in picked):.1f} |")
    lines += ["", "### Failed requests, by cause", "", "| Request | Arm | Cause | Tools called | Answer starts |", "|---|---|---|---|---|"]
    for arm, part in (("off", off), ("on", on)):
        for r in part.values():
            if not r["success"]:
                lines.append(f"| {r['id']} | {arm} | {cause(r)} | {', '.join(c['tool'] for c in r['tools']) or 'none'} | "
                             f"{r['answer'][:80].replace('|', '/').replace(chr(10), ' ')} |")
    return lines + [""]


EXPECTED = {"content": "search_notes", "quiz_me": "make_quiz", "quiz_answer": "grade_answer", "remember": "remember_fact"}


def cause(r):
    """A first guess at why a request failed, for grouping; read the trace before writing it up."""
    called = [c["tool"] for c in r["tools"]]
    if r["cap_hit"]:
        return "looping (step cap hit)"
    if r["type"] == "off_topic":
        return "searched or stored when it should not"
    if EXPECTED[r["type"]] not in called:
        return "wrong tool"
    if any(c["error"] for c in r["tools"]):
        return "wrong inputs (tool error)"
    if r["note"]:
        return r["note"][:60]
    if r["warning"]:
        return "unfaithful answer (warning)"
    if not r["tool_ok"]:
        return "weak evidence" if r["type"] == "content" else "wrong inputs"
    return "answer did not meet the check"


def ragas():
    on_path, off_path = (ac.OUT / f"ragas_agent_{arm}.json" for arm in ("search_again_on", "search_again_off"))
    if not (on_path.exists() and off_path.exists()):
        return []
    on, off = ({r["id"]: r for r in json.loads(p.read_text(encoding="utf-8"))["rows"]} for p in (on_path, off_path))
    gen = {arm: {r["id"]: r for r in json.loads((ac.OUT / f"generation_agent_{arm}.json").read_text(encoding="utf-8"))["rows"]}
           for arm in ("search_again_on", "search_again_off")}
    for arm, scores in (("search_again_on", on), ("search_again_off", off)):
        for i, r in scores.items():
            r.update(tokens=gen[arm][i]["tokens"], duration_ms=gen[arm][i]["duration_ms"], searches=gen[arm][i]["searches"])
    judge = json.loads(on_path.read_text(encoding="utf-8"))["judge"]
    return ["## Answer quality (RAGAS): search-again on vs off", "",
            f"Scored by the Scorer, `{judge}`. Context metrics cover text questions only.", ""] + comparison(
        "75 llm_eval gold questions", on, off,
        [("factual_correctness", "Factual correctness"), ("faithfulness", "Faithfulness"),
         ("context_precision", "Context precision"), ("context_recall", "Context recall"),
         ("searches", "Searches per answer"), ("total_tokens", "Tokens per answer"), ("latency_s", "Latency (s)")],
        "search-again on", "search-again off")


def safety():
    lines = []
    inj = load_jsonl("injection")
    if inj:
        on, off = inj.get("guard_on", {}), inj.get("guard_off", {})
        lines += ["## Prompt injection: guard on vs off", ""]
        lines += comparison("Injection rate (lower is better)", on, off, [("injected", "Injection followed")], "guard on", "guard off")
        lines += ["| Attack | Guard off | Guard on |", "|---|---|---|"]
        for attack in sorted({r["attack"] for r in off.values()} | {r["attack"] for r in on.values()}):
            cell = lambda part: f"{sum(r['injected'] for r in part.values() if r['attack'] == attack)} / {sum(r['attack'] == attack for r in part.values())}"
            lines.append(f"| {attack} | {cell(off)} | {cell(on)} |")
        lines.append("")
    poi = load_jsonl("poisoning")
    if poi:
        on, off = poi.get("memory_check_on", {}), poi.get("memory_check_off", {})
        lines += ["## Memory poisoning: memory-write check on vs off", ""]
        lines += comparison("Poisoning (lower is better, except wrong belief recorded)", on, off,
                            [("stored", "False claim stored as a fact"), ("repeated", "Repeated as true next session"),
                             ("poisoned", "Poisoned (either)"), ("belief_recorded", "Recorded as a wrong belief")],
                            "check on", "check off")
    return lines


def main():
    lines = ["# Agent evaluation results", "", "Written by `eval/summarise_agent.py`. Bold gains have a 95% interval that excludes 0.", ""]
    body = agent_test() + ragas() + safety()
    OUT.write_text("\n".join(lines + (body or ["No experiment has been run yet."])) + "\n", encoding="utf-8")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
