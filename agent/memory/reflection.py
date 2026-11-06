"""Step 24, learner-model memory: reflection at the end of a session.

end_session(): the app calls it once when a session ends, only with memory "learner".
One LLM call (purpose "reflection") reads the session and replies with a session note and the
concepts the student struggled with. The note is stored; each listed concept on the notebook's
concept list gets one wrong mark, so it moves toward a weak spot.
"""
import json

from agent import s4_trace
from agent.memory import learner
from agent.prompts import REFLECTION_PROMPT
from agent.s2_tools import MODEL, TurnContext


def end_session(student_id, history):
    """Reflect on one session's {"role", "content"} messages. Returns the stored note, or None if the reply was bad."""
    ctx = TurnContext(student_id=student_id)  # turn 0: after the session, not a chat turn
    known = learner.concepts(student_id)
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history if m.get("content"))
    messages = [{"role": "system", "content": REFLECTION_PROMPT.format(known=", ".join(known) or "none")},
                {"role": "user", "content": transcript}]
    content = s4_trace.traced_chat(ctx, "reflection", messages, MODEL)["message"].get("content") or ""
    try:
        reply = json.loads(content[content.find("{"):content.rfind("}") + 1])
    except ValueError:
        return None
    if not (isinstance(reply, dict) and isinstance(reply.get("note"), str) and reply["note"].strip()):
        return None
    learner.add_session_note(student_id, reply["note"])
    weak = reply.get("weak_spots") if isinstance(reply.get("weak_spots"), list) else []
    # only names on the concept list, each once, so reflection cannot invent concepts
    for concept in {learner.norm(c) for c in weak if isinstance(c, str)} & set(known):
        learner.update_mastery(student_id, concept, False)
    return reply["note"]
