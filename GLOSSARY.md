# Study Tutor

A tutor that answers, quizzes and tracks each student using only the files that student uploaded.

## Material

**Notebook**:
One student's uploaded files and everything derived from them. Students never see each other's notebooks.
_Avoid_: Workspace, library

**Passage**:
A short piece of an uploaded file, with the file and page or slide it came from. The unit that is searched and cited.
_Avoid_: Chunk (everywhere outside `src/tutor/retrieval/`, where `Chunk` is the pipeline stage before search), snippet, document

**Citation**:
A passage the tutor's answer actually refers to, shown so the student can read it. A passage retrieved but not used is not a citation.
_Avoid_: Source, reference

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

**Session**:
A student's turns from entering their name until they end the session, which writes a session note. A session closed any other way writes no note.
_Avoid_: Conversation, chat

**Turn**:
One student message and the tutor's final reply to it.
_Avoid_: Round, exchange

**Step**:
One LLM call inside a turn, plus the tool calls it asks for. A turn has at most the step cap of steps, plus one forced-answer step (tools off) when the cap is hit, plus one regenerate step when the judge fails the answer.
_Avoid_: Iteration, hop

## Checks

**Judge**:
The LLM, from a different model family than the tutor, that runs the faithfulness check.
_Avoid_: Grader (that is `grade_answer`), checker, critic

**Faithfulness**:
Whether every claim about course content in the tutor's final answer is stated in the passages retrieved that turn or shown in their pictures. It checks the answer matches the notes, not that the notes are right.
_Avoid_: Correctness, accuracy, groundedness

**Regenerate**:
The tutor's single retry after the judge fails an answer, with the judge's reason added. It counts as one more step.
_Avoid_: Retry (alone), rewrite

**Warning**:
The flag on a turn's answer when it still fails the faithfulness check after the regeneration. The answer is shown, marked as unsupported.
_Avoid_: Error, alert
