# Study Tutor

A tutor that answers, quizzes and tracks each student using only the files that student uploaded.

## Material

**Notebook**:
One student's uploaded files and everything derived from them. Students never see each other's notebooks.
_Avoid_: Workspace, library

**Passage**:
A short piece of an uploaded file, with the file and page or slide it came from. The unit that is searched and cited.
_Avoid_: Chunk (everywhere outside `src/tutor/retrieval/`, where `Chunk` is the pipeline stage before search), snippet, document

**Concept**:
A named idea taught in the notebook (e.g. "LoRA", "KV cache"), taken from a fixed list built from the uploaded files. Quizzes, mastery scores and weak spots all refer to the same concept list.
_Avoid_: Topic, subject, skill

## Learner model

**Mastery score**:
How well the student knows one concept, from 0 to 1. A concept never tested starts at 0.5: unknown, not weak.
_Avoid_: Grade, level

**Weak spot**:
A concept whose mastery score is below the weak threshold.
_Avoid_: Gap, struggle

**Wrong belief**:
A misconception the student showed about a concept, recorded so the tutor can target it. Never treated as true.
_Avoid_: Mistake, error, misconception (in code)

**Fact**:
A true claim the tutor remembers about a concept, always backed by a passage from the student's own files. A student's word alone never makes a fact.
_Avoid_: Note, knowledge

**Session note**:
A short summary of what the student struggled with in one past session, written by reflection at the end of that session.
_Avoid_: Memory (alone), log

## Conversation

**Turn**:
One student message and the tutor's final reply to it.
_Avoid_: Round, exchange

**Step**:
One LLM call inside a turn, plus the tool calls it asks for. A turn has at most the step cap of steps, plus one forced-answer step (tools off) when the cap is hit.
_Avoid_: Iteration, hop
