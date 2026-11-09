"""Step 24, vector memory: the past session notes most similar to the student's message.

similar_notes(): run_turn calls it each turn with memory "vector". The notes reflection wrote
are embedded with the team's embedder and ranked by cosine similarity to the message.
"""
import numpy as np

from agent.memory import learner

K = 2  # shortcut: hard-coded, move to config.yaml once the team's config file exists


def similar_notes(student_id, text, embedder):
    """The K session notes most similar to text, most similar first. embedder is a tutor.retrieval Embedder."""
    notes = [n["note"] for n in learner.read_all(student_id)["session_notes"]]
    if not notes:
        return []
    # shortcut: re-embeds every note each turn; store the vector with the note once a student has dozens
    scores = embedder(notes) @ embedder.embed_query(text)  # vectors have length 1, so dot product is cosine
    return [notes[i] for i in np.argsort(-scores, kind="stable")[:K]]
