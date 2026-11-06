# Study Tutor

A study tutor for any subject, built on your own files, with learner memory. Students upload their own notes, slides or readings into a notebook. The tutor answers questions with citations, quizzes them, and remembers which concepts each student is weak on across sessions.

It is similar in spirit to Google's NotebookLM, which answers questions from a user's own sources. Our focus is the tutoring side: quizzing, grading with hints, and a memory of each student's weak spots that carries across sessions.

> **Status:** in development for the IS469 group project. Due G1 Tue 10 Nov / G2 Fri 13 Nov 2026. See [`docs/IS469_Project_Plan_Study_Tutor_General.docx`](docs/IS469_Project_Plan_Study_Tutor_General.docx) for the full project plan, and [`docs/build_and_test_steps.md`](docs/build_and_test_steps.md) for the step-by-step build and test guide.

## Features

- **Upload your own material:** PowerPoint (.pptx) files and readable PDFs (selectable text, not scans), several per notebook. Anything else is rejected with a clear message. Each file is indexed once, with a progress bar.
- **Chat with citations:** every answer cites its source (e.g. "Lecture3.pdf, page 12" or "Week5.pptx, slide 23"). Click a citation to see that passage.
- **Quiz-me mode:** pick a topic from your files and get 3–5 practice questions grounded in them. The tutor grades your answer and gives a hint, not the answer.
- **"My weak spots" panel:** a mastery bar per concept (e.g. LoRA 40%, photosynthesis 85%) that updates after quizzes and chats. Concepts are pulled from the uploaded files automatically.
- **Memory controls:** view, edit and delete what the tutor remembers about you, and remove any uploaded file.
- **One notebook per student:** each student enters a name or ID; their files, index and memory are kept separate.
- **Search-again:** the agent checks whether the retrieved passages answer the question, and if not, rephrases and searches again (up to 3 times). It can be switched off to get the fixed single-search baseline.

## How it works

1. The student opens their notebook and uploads files. Each new file is extracted, cleaned, chunked, embedded and indexed once, and its key concepts are listed.
2. The student asks a question or starts a quiz.
3. The agent reads the student's learner model and recent session notes.
4. If the request needs a search, it calls `search_notes` on the student's own notebook, using the search mode set in the config.
5. With search-again on, it judges whether the evidence is enough and searches again if not.
6. It answers with citations, or calls `make_quiz` / `grade_answer` in quiz mode. Text from files is passed to the AI as data, never as instructions.
7. A faithfulness check, run on a different model family, confirms the answer is supported by the cited passages.
8. It calls `update_mastery`. Wrong beliefs are stored as weak spots; facts are stored only if the student's files support them.
9. Every tool call goes to the trace file and every AI response is saved to disk.

The agent is a single tool-calling loop with four tools (`search_notes`, `make_quiz`, `grade_answer`, `update_mastery`) and a step cap per turn.

## Tech stack

| Component | Choice |
|---|---|
| UI | Streamlit: file uploader, chat, quiz mode, weak-spots panel, memory page |
| File extraction | python-pptx for .pptx, PyMuPDF for .pdf; PDFs without selectable text are rejected |
| Chunking | One passage per slide for decks; a few hundred tokens with overlap for documents, split at headings; size set in config |
| Tutor LLM | Llama on Groq, model ID pinned in config, temperature 0 |
| Judge / faithfulness checker | A model from a different family than Llama (TBC) |
| Embeddings | sentence-transformers |
| Vector store | FAISS or Chroma; one index per notebook, keyed by file hash so each file is embedded once |
| Keyword search | rank_bm25 |
| Reranker | Cross-encoder |
| Concept extraction | At upload, the LLM lists each file's key concepts for the weak-spots panel |
| Learner model | SQLite per student (mastery, wrong beliefs, file-backed facts) |
| Session memory | Small vector store of past session notes |
| Logging | JSONL trace of every tool call; saved AI responses |

**Search modes** (set in config): `dense`, `hybrid` (BM25 + dense merged with RRF), or `hybrid + rerank` (top 20 reranked to top 5).

## Getting started

> Setup commands will be added once the one-command launch is in place.

Turn on the pre-commit checks (ruff and pytest) once per clone: `git config core.hooksPath .githooks`.

1. Set your Groq API key and choose settings in the config file: model ID, temperature, search mode, chunk size, search-again switch and random seeds all live there.
2. Launch the app with a single command.
3. Enter your name or ID, then upload your .pptx or PDF files into your notebook.

AI responses are cached to disk, so reruns of the evaluation are free, repeatable and don't hit Groq rate limits.

## Evaluation

All results come from one frozen system (feature freeze before testing starts). Every test runs on three fixed test corpora bundled with the repository, and results are reported per corpus:

- **Slide decks:** the IS469 lecture slides as .pptx (short, bullet-point text)
- **Long document:** a textbook chapter or lecture notes as PDF (dense prose)
- **Rules document:** e.g. SMU academic regulations as PDF (exact numbers, dates and conditions)

| Area | Compare | Measure |
|---|---|---|
| Retrieval | Dense vs hybrid vs hybrid + rerank; 2–3 chunk sizes; search-again off vs on; best setup vs long-context baseline | Hit@5, Recall@5, answer correctness on ~120 gold questions (~40 per corpus); tokens and latency |
| Agent | Search-again off vs on | 30 test requests across the corpora: task success, tool-call accuracy, steps, searches per answer, tokens, latency |
| Memory | None vs vector vs learner model + reflection | 6 simulated students (2 per corpus) with planted misconceptions, 4 sessions each |
| Usability | — | 6–8 classmates using their own notes, 3 tasks each: completion, time, SUS score, feedback |
| Safety | Memory-write check on vs off; data-not-instructions guardrail on vs off | Poisoning rate over ~20 attempts; injection rate over ~20 planted files |
| Upload | — | Indexing time by file size; extracted-text spot-checks; scanned PDFs and unsupported types rejected |

The LLM judge comes from a different model family than the tutor, is validated against 30 hand-marked answers, and has answer order swapped to control for position bias. No BLEU/ROUGE.

## Guardrails

- Every claim cites a passage; faithfulness is checked by a different model family. Answers are worded as "according to your notes", since grounded is not the same as correct.
- Uploaded text is wrapped in markers and treated as data, never as instructions; nothing in a file can trigger a tool call.
- Separate notebook, index and memory per student; search only runs on the current student's notebook.
- Student beliefs and facts are stored separately; facts only with support from the student's files.
- Deleting a file removes its text, passages and index entries. File size limit; only .pptx and readable PDFs accepted.
- Step cap per turn and at most 3 searches per answer.
- Hints instead of answers in quiz mode; quiz-dump requests are refused.
- User-test logs are anonymised and collected with consent.

## Course topics used

- **Topic 8: Agentic RAG:** retrieval, chunk size, hybrid search, reranking, citations, search-again, untrusted input
- **Topic 7: Agents with tools and memory:** tool calling, agent loop, harness, memory, reflection, guardrails
- **Topics 5–6: Evaluating LLMs:** LLM-as-judge, judge bias controls, cost and latency
- **Topic 1: Text processing:** cleaning extracted text; BM25 builds on TF-IDF

## Team

The system splits into three parts, each owned by one team member:

1. Retrieval
2. Tutor and memory
3. Harness, product and safety

## Future work

Fine-tuning and DPO, other file types (Word, plain text), multimodal reading of images and diagrams, OCR for scanned PDFs, web search as a fallback source, an MCP server, a multi-agent supervisor, and larger ablation studies.
