"""Step 12: the agent loop.

memory 'learner': weak spots + wrong beliefs into the system prompt; 'vector': the session notes most
similar to the message; 'none': only the concept list
for step in range(step_cap):
    reply = llm.chat(messages, tools)
    if no tool_calls: break
    for each tool_call: run_tool, append `role: tool` message, write a tool row
if cap hit: one last llm.chat(..., tool_choice="none") to force an answer
faithfulness on: the judge checks the answer; on a fail, regenerate once, then return it with a warning
write the final trace row (see s4_trace for the row format)
"""

import json
import re
import time

from agent.prompts import GUARD_PROMPT, REGENERATE_PROMPT, SYSTEM_PROMPT
from agent.s2_tools import MODEL, TOOL_SCHEMAS, TurnContext, run_tool, stub_passages, stub_search
from agent import s4_trace, s6_guard, s7_faithfulness
from agent.memory import learner, vector

# 1. TurnContext: bundles the student_id and config switches to the current turn (all the loops tgt). Basically giving the turn context on which student it's doing + how many steps alrdy, etc.

# 2. messages builds the list of messages the LLM needs. Every chat sends the whole list of the previous context (chat history, information etc), each step taken appends to the history to give to LLM. If smth NOT in messages, LLM will not know

# 3. Citations come last: the passages retrieved this turn that the final answer names by ID

# 4. the loop itself, each step is 1 loop
#   4a: Asking LLM: Give LLM whole convo + tool menu => Either answer now or use tools
#   4b: Add LLM reply to the messages array as history, comes BEFORE the tool result
#   4c: if the LLM did not choose to use a tool, that means it has a final answer => break
#   4d: Run the tools the LLM wants to use. Bad input never raises errors, but instead goes back to the LLM so it can fix its calls. Each call gets a tool row
#   4e: repeat itself until meet turn cap
#   4f: write the final row (every LLM call already wrote its own llm row through s4_trace.traced_chat)

def _cited(answer, passage_ids):
    """The ids among passage_ids that the answer names, in order of first appearance."""
    first = {}
    for i in set(passage_ids):
        # the lookarounds keep p1 from matching inside p10 or a longer id
        if found := re.search(rf"(?<![\w:-]){re.escape(i)}(?![\w:-])", answer):
            first[i] = found.start()
    return sorted(first, key=first.get)


def run_turn(user_message, student_id, turn, *, history=None, search_fn=stub_search,
             passages_fn=stub_passages, search_again=True, guard=True, memory_check=True, faithfulness=True, memory="learner", embedder=None, step_cap=6):
    """Run one student turn. Returns {"answer": str, "citations": list, "steps": int,
    "faithfulness": the judge's {"supported", "reason"} or None when off, "warning": bool}.

    answer: the tutor's text with its passage ids left in, because the next turn's grade_answer
    needs them in history. The app removes them for display only.
    citations: the ids of the passages retrieved this turn that the final answer names, in order
    of first appearance, without repeats. Empty on a turn that calls make_quiz: the ids after quiz
    questions are for grading, and showing them as sources would point at the answer.

    turn: the app's turn number, written on every trace row.
    history: earlier {"role": "user"/"assistant", "content": ...} messages, so quiz
    answers can be graded against the question asked last turn.
    search_fn: search(student_id, query) -> [{id, file, page, text}, ...].
    passages_fn: get_passages(student_id, ids) -> the passages with those ids, same shape.
    search_again: rephrase and search again, up to 3 searches, while an LLM check says the passages
    are not enough; off uses the first result, the fixed-pipeline baseline.
    memory_check: remember_fact stores only facts citing a passage retrieved this turn;
    off is the unsafe baseline for the poisoning test.
    faithfulness: judge the answer against every passage retrieved this turn; warning is true
    when it still fails after one regeneration.
    memory: "learner" puts the student's weak spots and wrong beliefs in the system prompt;
    "vector" puts only the past session notes most similar to user_message there;
    "none" leaves the student out (the concept list stays, it describes the notebook).
    The app calls memory.reflection.end_session at session end with "learner" and "vector",
    since both read the session notes it writes.
    embedder: a tutor.retrieval Embedder, needed only with memory "vector".
    """
    if memory not in ("none", "learner", "vector"):
        raise ValueError(f"unknown memory variant: {memory!r}")
    if memory == "vector" and embedder is None:
        raise ValueError('memory "vector" needs an embedder')

    t_start = time.perf_counter()
    ctx = TurnContext(student_id=student_id, turn=turn, search_fn=search_fn,
                      passages_fn=passages_fn, search_again=search_again, guard=guard,
                      memory_check=memory_check, faithfulness=faithfulness, memory=memory, step_cap=step_cap)
    system = SYSTEM_PROMPT + (GUARD_PROMPT if guard else "")
    concepts = learner.concepts(student_id)
    if concepts:
        system += "\nConcepts in this notebook (use these exact names in tools): " + ", ".join(concepts)
    if memory == "learner":
        if weak := learner.weak_concepts(student_id):
            system += ("\nThis student's weak spots (score 0 to 1): " + ", "
                       .join(f"{w['concept']} {w['score']:.2f}" for w in weak))
        if beliefs := learner.wrong_beliefs(student_id):
            system += ("\nWrong beliefs this student showed (never treat them as true): " + "; "
                       .join(f"{b['concept']}: {b['belief']}" for b in beliefs))
    if memory == "vector" and (notes := vector.similar_notes(student_id, user_message, embedder)):
        system += "\nNotes from this student's past sessions: " + " | ".join(notes)

    messages = [
        {
            "role": "system",
            "content": system
        },
        *(history or []),
        {
            "role": "user",
            "content": user_message
        }
    ]

    answer, cap_hit = None, False
    for step in range(1, step_cap+1):
        ctx.step = step
        msg = s4_trace.traced_chat(ctx, "tutor", messages, MODEL, tools=TOOL_SCHEMAS)["message"]

        messages.append({k: msg[k] for k in ("role", "content", "tool_calls") if k in msg})

        if not msg.get("tool_calls"):
            answer = msg.get("content") or ""
            break

        for call in msg["tool_calls"]:
            name, args = call["function"]["name"], call["function"]["arguments"]
            t0 = time.perf_counter()
            result = run_tool(name, args, ctx)
            try:
                inputs = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                inputs = args  # bad JSON: keep the raw string so the trace shows what the LLM sent

            s4_trace.log_row(ctx, "tool", tool=name, inputs=inputs, ok="error" not in result,
                             error=result.get("error"),
                             duration_ms=round((time.perf_counter() - t0) * 1000),
                             result_summary=json.dumps(result)[:200])

            content = s6_guard.material(result["passages"], guard) if "passages" in result else json.dumps(result)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    else:
        # cap hit: one last call with tools off, so it must answer from what it has
        ctx.step, cap_hit = step_cap + 1, True
        msg = s4_trace.traced_chat(ctx, "forced_answer", messages, MODEL,
                                   tools=TOOL_SCHEMAS, tool_choice="none")["message"]
        answer = msg.get("content") or ""
        messages.append({"role": "assistant", "content": answer})

    verdict = s7_faithfulness.check(ctx, answer) if ctx.faithfulness else None
    if verdict and not verdict["supported"]:
        # one regeneration with the judge's reason; tools stay listed (history holds tool calls) but off
        messages.append({"role": "user", "content": REGENERATE_PROMPT.format(reason=verdict["reason"])})
        ctx.step += 1
        answer = s4_trace.traced_chat(ctx, "regenerate", messages, MODEL, tools=TOOL_SCHEMAS,
                                      tool_choice="none")["message"].get("content") or ""
        verdict = s7_faithfulness.check(ctx, answer)
    warning = bool(verdict) and not verdict["supported"]

    s4_trace.log_row(ctx, "final", steps=ctx.step, cap_hit=cap_hit, searches=ctx.searches,
                     tokens=dict(ctx.tokens),
                     duration_ms=round((time.perf_counter() - t_start) * 1000),
                     faithfulness=verdict, answer_preview=answer[:100])
    citations = [] if ctx.quiz_called else _cited(answer, ctx.passage_ids)
    return {"answer": answer, "citations": citations, "steps": ctx.step, "faithfulness": verdict, "warning": warning}
