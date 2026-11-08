"""Step 12: the agent loop.

memory 'learner': weak spots + wrong beliefs into the system prompt; 'vector': the session notes most
similar to the message; 'none': only the concept list
a numbered answer to the open quiz: grade_answer runs first, at step 0, as if the tutor had called it
for step in range(step_cap):
    reply = llm.chat(messages, tools)
    if no tool_calls: break
    for each tool_call: run_tool, append `role: tool` message, write a tool row
        grade_answer with correct false: its tool message also carries the rule against stating the answer
    pictures among the passages the tools retrieved: one user message after the results, holding the turn's newest pictures
if cap hit: one last llm.chat(..., tool_choice="none") to force an answer
faithfulness on: the judge checks the answer; on a fail, regenerate once back in the loop (tools on, same step cap),
then return it with a warning if it still fails
write the final trace row (see s4_trace for the row format)
"""

import json
import re
import time

from agent.prompts import GUARD_PROMPT, REGENERATE_PROMPT, SYSTEM_PROMPT, WRONG_ANSWER_PROMPT
from agent.s2_tools import MODEL, TOOL_SCHEMAS, TurnContext, picture_parts, run_tool, stub_passages, stub_search
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
    for passage_id in set(passage_ids):
        # the same boundaries as PASSAGE_ID, and they keep p1 from matching inside p10
        if found := re.search(rf"\b{re.escape(passage_id)}\b", answer):
            first[passage_id] = found.start()
    return sorted(first, key=first.get)


# shortcut: only a leading number with punctuation ("2.", "Q2:", "question 2 -") counts, widen it if live runs show other forms
QUIZ_ANSWER = re.compile(r"\s*(?:q(?:uestion)?\s*)?(\d+)\s*[.):-]\s*", re.I)


def _graded_quiz_answer(user_message, open_quiz):
    """An assistant message calling grade_answer for a numbered answer to the open quiz, or None."""
    found = open_quiz and QUIZ_ANSWER.match(user_message)
    if not found or not 1 <= int(found[1]) <= len(open_quiz):
        return None
    question = open_quiz[int(found[1]) - 1]
    args = {"question": question["question"], "student_answer": user_message[found.end():].strip() or user_message,
            "passage_ids": question["passage_ids"]}
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "quiz-answer", "type": "function", "function": {"name": "grade_answer", "arguments": json.dumps(args)}}]}


def run_turn(user_message, student_id, turn, *, history=None, open_quiz=None, search_fn=stub_search,
             passages_fn=stub_passages, search_again=True, guard=True, memory_check=True, faithfulness=True, images=True, memory="learner", embedder=None, step_cap=6):
    """Run one student turn. Returns {"answer": str, "citations": list, "steps": int,
    "faithfulness": the judge's {"supported", "reason"} or None when off, "warning": bool,
    "quiz": the questions of this turn's last successful make_quiz, each {"question", "passage_ids"}, or None,
    "tool_calls": list}.

    answer: the tutor's text with its passage ids left in, because the next turn's grade_answer
    needs them in history. The app removes them for display only.
    citations: the ids of the passages retrieved this turn that the final answer names, in order
    of first appearance, without repeats. Empty on a turn that called grade_answer or where make_quiz
    passed its checks: the ids in quiz questions and hints would point at the answer.
    tool_calls: every tool call of the turn, in the order it ran, as {"tool": name, "inputs": the
    parsed arguments (the raw string when the JSON was bad, as the trace keeps it), "result": the
    tool's full result}. Empty when no tool ran. For evaluation: the trace keeps only 200 characters.

    turn: the app's turn number, written on every trace row.
    history: earlier {"role": "user"/"assistant", "content": ...} messages, so quiz
    answers can be graded against the question asked last turn.
    open_quiz: the "quiz" an earlier turn returned, still open: listed in the system prompt with each
    question's passage ids, so an answer to it is graded with grade_answer. A message starting with a
    question number ("2. ...", "Q2: ...") is graded by code at step 0, before the tutor's first call.
    search_fn: search(student_id, query) -> [{id, file, page, text}, ...], plus image_path if a passage has a picture.
    passages_fn: get_passages(student_id, ids) -> the passages with those ids, same shape.
    search_again: rephrase and search again, up to 3 searches, while an LLM check says the passages
    are not enough; off uses the first result, the fixed-pipeline baseline.
    memory_check: remember_fact stores only facts citing a passage retrieved this turn;
    off is the unsafe baseline for the poisoning test.
    faithfulness: judge the answer against every passage retrieved this turn and the student's record
    (concepts, weak spots, wrong beliefs, session notes, open quiz), with history and user_message as the
    support for claims about the session only; warning is true when it still fails
    after one regeneration, which may search or grade first.
    images: every LLM call over passages also receives their pictures, at most 3 per request, newest kept;
    off leaves pictures and passages with no text out, the text-only baseline. Pictures never reach
    the answer, the trace or history, so a follow-up question searches again.
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
                      memory_check=memory_check, faithfulness=faithfulness, images=images, memory=memory, step_cap=step_cap)
    record = ""  # the student's record: what the tutor is told about this student and notebook, also the judge's support for it
    concepts = learner.concepts(student_id)
    if concepts:
        record += "\nConcepts in this notebook (use these exact names in tools): " + ", ".join(concepts)
    if memory == "learner":
        if weak := learner.weak_concepts(student_id):
            record += ("\nThis student's weak spots (score 0 to 1): " + ", "
                       .join(f"{w['concept']} {w['score']:.2f}" for w in weak))
        if beliefs := learner.wrong_beliefs(student_id):
            record += ("\nWrong beliefs this student showed (never treat them as true): " + "; "
                       .join(f"{b['concept']}: {b['belief']}" for b in beliefs))
    if memory == "vector" and (notes := vector.similar_notes(student_id, user_message, embedder)):
        record += "\nNotes from this student's past sessions: " + " | ".join(notes)
    if open_quiz:
        record += ("\nOpen quiz (grade an answer to one of these with grade_answer and that question's passage ids):\n"
                   + "\n".join(f"{n}. {q['question']} [{', '.join(q['passage_ids'])}]"
                               for n, q in enumerate(open_quiz, start=1)))
    ctx.record = record.strip()
    ctx.session = "\n".join([*(f"{'Student' if m['role'] == 'user' else 'Tutor'}: {m['content']}" for m in history or []),
                             f"Student's message this turn: {user_message}"])

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + (GUARD_PROMPT if guard else "") + record
        },
        *(history or []),
        {
            "role": "user",
            "content": user_message
        }
    ]

    answer, cap_hit, verdict, regenerated = None, False, None, False
    picture_message = None
    tool_calls = []
    purpose = "tutor"  # "regenerate" for the first call after the judge fails the answer
    # a numbered answer to the open quiz is graded by code at step 0: the tutor skipped grade_answer too often
    pending = _graded_quiz_answer(user_message, open_quiz)
    while True:
        if pending:
            msg, pending = pending, None
            messages.append(msg)
        elif ctx.step >= step_cap:
            # cap hit: one last call with tools off, so it must answer from what it has
            ctx.step, cap_hit = ctx.step + 1, True
            msg = s4_trace.traced_chat(ctx, "forced_answer" if purpose == "tutor" else purpose, messages, MODEL,
                                       tools=TOOL_SCHEMAS, tool_choice="none")["message"]
            answer = msg.get("content") or ""
            messages.append({"role": "assistant", "content": answer})
        else:
            ctx.step += 1
            msg = s4_trace.traced_chat(ctx, purpose, messages, MODEL, tools=TOOL_SCHEMAS)["message"]
            purpose = "tutor"
            messages.append({k: msg[k] for k in ("role", "content", "tool_calls") if k in msg})
            if not msg.get("tool_calls"):
                answer = msg.get("content") or ""
        if answer is not None:
            verdict = s7_faithfulness.check(ctx, answer) if ctx.faithfulness else None
            if regenerated or not verdict or verdict["supported"]:
                break
            # one regeneration with the judge's reason, back in the loop with tools on, so it can search or grade first
            regenerated, answer, purpose = True, None, "regenerate"
            messages.append({"role": "user", "content": REGENERATE_PROMPT.format(reason=verdict["reason"])})
            continue

        for call in msg["tool_calls"]:
            name, args = call["function"]["name"], call["function"]["arguments"]
            t0 = time.perf_counter()
            result = run_tool(name, args, ctx)
            try:
                inputs = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                inputs = args  # bad JSON: keep the raw string so the trace shows what the LLM sent
            tool_calls.append({"tool": name, "inputs": inputs, "result": result})

            # the trace keeps text only: not even the picture's local path
            summary = {**result, "passages": s6_guard.text_fields(result["passages"])} if "passages" in result else result
            s4_trace.log_row(ctx, "tool", tool=name, inputs=inputs, ok="error" not in result,
                             error=result.get("error"),
                             duration_ms=round((time.perf_counter() - t0) * 1000),
                             result_summary=json.dumps(summary)[:200])

            content = s6_guard.material(result["passages"], guard) if "passages" in result else json.dumps(result)
            if name == "grade_answer" and result.get("correct") is False:
                # in this step's request, since the system prompt's rule alone lets the reply give the answer away
                content += "\n" + WRONG_ANSWER_PROMPT
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

        # only after all of a step's tool results: providers reject a user message between them.
        # One message, rebuilt each step, so old pictures can't push a request over the cap.
        if picture_message:
            messages.remove(picture_message)
        parts = picture_parts(ctx.passages)
        picture_message = {"role": "user", "content": parts} if parts else None
        if picture_message:
            messages.append(picture_message)
    warning = bool(verdict) and not verdict["supported"]

    s4_trace.log_row(ctx, "final", steps=ctx.step, cap_hit=cap_hit, searches=ctx.searches,
                     tokens=dict(ctx.tokens),
                     duration_ms=round((time.perf_counter() - t_start) * 1000),
                     faithfulness=verdict, message_preview=user_message[:100], answer_preview=answer[:100])
    citations = [] if ctx.hide_citations else _cited(answer, ctx.passage_ids)
    return {"answer": answer, "citations": citations, "steps": ctx.step, "faithfulness": verdict, "warning": warning,
            "quiz": ctx.quiz, "tool_calls": tool_calls}
