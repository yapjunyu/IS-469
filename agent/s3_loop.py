"""Step 12: the agent loop.

load learner model + recent session notes
for step in range(step_cap):
    reply = llm.chat(messages, tools)
    if no tool_calls: break
    for each tool_call: run_tool, append `role: tool` message, log to trace
if cap hit: one last llm.chat(..., tool_choice="none") to force an answer
"""

import json

from agent.prompts import SYSTEM_PROMPT
from agent.s1_llm import chat
from agent.s2_tools import TOOL_SCHEMAS, TurnContext, run_tool

# shortcut: hardcoded, move to config.yaml once the team agrees on it
MODEL = "llama-3.3-70b-versatile"

# 1. Setup context and messages in the format to be sent to LLM
# 2. Looping call the LLM replying with a request which is added to messages so LLM remember what it called
# 3. Tool is ran and chunks/result is returned. result added to messages linking to LLM's request
# 4. Calling LLM again now with its own request and search results
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
        msg = chat(messages, MODEL, tools=TOOL_SCHEMAS)["message"]
        messages.append({k: msg[k] for k in ("role", "content", "tool_calls") if k in msg})

        if not msg.get("tool_calls"):
            return {"answer": msg.get("content", ""), "citations": citations, "steps": step}

        for call in msg["tool_calls"]:
            result = run_tool(call["function"]["name"], call["function"]["arguments"], ctx)
            citations += [{"file": c["file"], "page": c["page"]} for c in result.get("chunks", [])]
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})

    # cap hit: one last call with tools off, so it must answer from what it has
    msg = chat(messages, MODEL, tools=TOOL_SCHEMAS, tool_choice="none")["message"]
    return {"answer": msg.get("content", ""), "citations": citations, "steps": step_cap + 1}
