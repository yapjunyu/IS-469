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


def scorer_tokens(arms):
    """The Scorer's tokens per arm, summed over the rows that record them (older rows do not)."""
    lines = []
    for name, part in arms:
        used = [r["scorer_tokens"] for r in part.values() if "scorer_tokens" in r]
        if used:
            lines.append(f"| {name} | {sum(t['in'] for t in used)} | {sum(t['out'] for t in used)} |")
    return ["| Scorer tokens | In | Out |", "|---|---|---|"] + lines + [""] if lines else []


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
    lines += scorer_tokens([("search-again off", off), ("search-again on", on)])
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
# right tool, right inputs, no warning, still failed: an uncited answer or a wrong verdict on the right passages is
# unfaithful to them; too few quiz questions or a fact citing no passage read means the evidence was weak
LEFT_OVER = {"content": "unfaithful answer", "quiz_answer": "unfaithful answer", "quiz_me": "weak evidence",
             "remember": "weak evidence"}


def cause(r):
    """A first guess at why a request failed, one of spec #24's causes; read the trace before writing it up."""
    if r["cap_hit"]:
        return "looping"
    if r["note"].startswith("gave the answer away"):
        return "answer given away"
    if r["type"] == "off_topic" or EXPECTED[r["type"]] not in [c["tool"] for c in r["tools"]]:
        return "wrong tool"  # off topic fails only on a search or a stored fact
    if any(c["error"] for c in r["tools"]):
        return "wrong inputs"
    if r["warning"]:
        return "unfaithful answer"
    if not r["tool_ok"]:
        return "weak evidence" if r["type"] == "content" else "wrong inputs"  # content: no gold passage was read
    return LEFT_OVER[r["type"]]  # these per-type fallbacks are first guesses, not checked against traces yet


def ragas():
    arms = ("search_again_on", "search_again_off")
    on_path, off_path = (ac.OUT / f"ragas_agent_{arm}.json" for arm in arms)
    gen_paths = {arm: ac.OUT / f"generation_agent_{arm}.json" for arm in arms}
    if not all(p.exists() for p in (on_path, off_path, *gen_paths.values())):
        return []
    saved_on, saved_off = (json.loads(p.read_text(encoding="utf-8")) for p in (on_path, off_path))
    on, off = ({r["id"]: r for r in saved["rows"]} for saved in (saved_on, saved_off))
    gen = {arm: {r["id"]: r for r in json.loads(p.read_text(encoding="utf-8"))["rows"]} for arm, p in gen_paths.items()}
    for arm, scores in (("search_again_on", on), ("search_again_off", off)):
        for i, r in scores.items():
            r.update(tokens=gen[arm][i]["tokens"], duration_ms=gen[arm][i]["duration_ms"], searches=gen[arm][i]["searches"])
    return ["## Answer quality (RAGAS): search-again on vs off", "",
            f"Scored by the Scorer, `{saved_on['judge']}`. Context metrics cover text questions only.", "",
            "| Scorer tokens | Total |", "|---|---|", f"| search-again off | {saved_off['tokens']} |",
            f"| search-again on | {saved_on['tokens']} |", ""] + comparison(
        "llm_eval gold questions", on, off,
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
        lines += scorer_tokens([("check off", off), ("check on", on)])
        lines += ["Facts never reach the tutor's prompt (only weak spots and wrong beliefs do), so \"repeated\" "
                  "cannot come from a stored fact: only from the wrong-belief line or the tutor itself.", ""]
    return lines


def main():
    lines = ["# Agent evaluation results", "", "Written by `eval/summarise_agent.py`. Bold gains have a 95% interval that excludes 0.", ""]
    body = agent_test() + ragas() + safety()
    out = ac.OUT / "RESULTS.md"  # read at call time, so tests redirect it
    out.write_text("\n".join(lines + (body or ["No experiment has been run yet."])) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
