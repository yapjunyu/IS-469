"""Step 12: the agent loop.

load learner model + recent session notes
for step in range(step_cap):
    reply = llm.chat(messages, tools)
    if no tool_calls: break
    for each tool_call: run_tool, append `role: tool` message, log to trace
if cap hit: one last llm.chat(..., tool_choice="none") to force an answer
"""

import json
import time
from pathlib import Path

from agent.prompts import SYSTEM_PROMPT
from agent.s1_llm import chat
from agent.s2_tools import TOOL_SCHEMAS, TurnContext, run_tool
from agent.s4_trace import log_call

# shortcut: hardcoded, move to config.yaml once the team agrees on it
MODEL = "llama-3.3-70b-versatile"
TRACE_PATH = Path("logs/trace.jsonl")

# 1. TurnContext: bundles the student_id and config switches to the current turn (all the loops tgt). Basically giving the turn context on which student it's doing + how many steps alrdy, etc.

# 2. messages builds the list of messages the LLM needs. Every chat sends the whole list of the previous context (chat history, information etc), each step taken appends to the history to give to LLM. If smth NOT in messages, LLM will not know

# 3. Citation array to store source of every passage searched in the turn

# 4. the loop itself, each step is 1 loop
#   4a: Asking LLM: Give LLM whole convo + tool menu => Either answer now or use tools
#   4b: Add LLM reply to the messages array as history, comes BEFORE the tool result
#   4c: if the LLM did not choose to use a tool, that means it has a final answer => return
#   4d: Run the tools the LLM wants to use. Bad input never raises errors, but instead goes back to the LLM so it can fix its calls. logged and sources added to citation array
#   4e: repeat itself until meet turn cap

def run_turn(user_message, student_id, *, history=None, search_again=True, guard=True, step_cap=6):
    """Run one student turn. Returns {"answer": str, "citations": list, "steps": int}.

    history: earlier {"role": "user"/"assistant", "content": ...} messages, so quiz
    answers can be graded against the question asked last turn.
    """

    ctx = TurnContext(student_id=student_id, search_again=search_again, guard=guard, step_cap=step_cap)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        *(history or []),
        {
            "role": "user",
            "content": user_message
        }
    ]
    citations = []

    for step in range(1, step_cap+1):
        reply = chat(messages, MODEL, tools=TOOL_SCHEMAS)
        msg = reply["message"]

        messages.append({k: msg[k] for k in ("role", "content", "tool_calls") if k in msg})

        if not msg.get("tool_calls"):
            return {"answer": msg.get("content", ""), "citations": citations, "steps": step}

        for call in msg["tool_calls"]:
            name, args = call["function"]["name"], call["function"]["arguments"]
            t0 = time.perf_counter()
            result = run_tool(name, args, ctx)

            log_call(TRACE_PATH, student=student_id, turn=ctx.turn, step=step, tool=name,
                     inputs=args, result_summary=json.dumps(result)[:200],
                     duration_ms=round((time.perf_counter() - t0) * 1000),
                     tokens=reply.get("usage"), ok="error" not in result,
                     error=result.get("error"))

            citations += [{"file": c["file"], "page": c["page"]} for c in result.get("chunks", [])]
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})

    # cap hit: one last call with tools off, so it must answer from what it has
    msg = chat(messages, MODEL, tools=TOOL_SCHEMAS, tool_choice="none")["message"]
    return {"answer": msg.get("content", ""), "citations": citations, "steps": step_cap + 1}
