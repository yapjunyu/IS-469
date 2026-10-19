# IS469 Study Tutor

A RAG tutor with learner memory for the IS469 course. It answers questions about the lecture slides and lab notebooks with slide citations, quizzes students on topics they pick, and remembers which concepts each student is weak on across sessions.

> **Status:** in development. Build window 13–28 Oct 2026; due G1 Tue 10 Nov / G2 Fri 13 Nov 2026. See [`docs/IS469_Project_Plan_Study_Tutor_Updated.docx`](docs/IS469_Project_Plan_Study_Tutor_Updated.docx) for the full project plan.

## Features

- **Chat with citations:** every answer cites its source (e.g. "Topic 5, slide 23"). Click a citation to see that slide's text.
- **Quiz-me mode:** pick a topic and get 3–5 practice questions grounded in the slides. The tutor grades your answer and gives a hint, not the answer.
- **"My weak spots" panel:** a mastery bar per concept (e.g. LoRA 40%, KV cache 85%) that updates after quizzes and chats.
- **Memory controls:** view, edit and delete what the tutor remembers about you.
- **Search-again:** the agent checks whether the retrieved slides answer the question, and if not, rephrases and searches again (up to 3 times). It can be switched off to get the fixed single-search baseline.

## How it works

1. The student asks a question or starts a quiz in the Streamlit app.
2. The agent reads the student's learner model and recent session notes.
3. If the request needs a search, it calls `search_slides` using the search mode set in the config.
4. With search-again on, it judges whether the evidence is enough and searches again if not.
5. It answers with citations, or calls `make_quiz` / `grade_answer` in quiz mode.
6. A faithfulness check, run on a different model family, confirms the answer is supported by the cited slides.
7. It calls `update_mastery`. Wrong beliefs are stored as weak spots; facts are stored only if a slide supports them.
8. Every tool call goes to the trace file and every AI response is saved to disk.

The agent is a single tool-calling loop with four tools (`search_slides`, `make_quiz`, `grade_answer`, `update_mastery`) and a step cap per turn.

## Tech stack

| Component | Choice |
|---|---|
| UI | Streamlit |
| Tutor LLM | Llama on Groq, model ID pinned in config, temperature 0 |
| Judge / faithfulness checker | A model from a different family than Llama (TBC) |
| Embeddings | sentence-transformers |
| Vector store | FAISS or Chroma |
| Keyword search | rank_bm25 |
| Reranker | Cross-encoder |
| Learner model | SQLite (mastery, wrong beliefs, slide-backed facts) |
| Session memory | Small vector store of past session notes |
| Logging | JSONL trace of every tool call; saved AI responses |

**Search modes** (set in config): `dense`, `hybrid` (BM25 + dense merged with RRF), or `hybrid + rerank` (top 20 reranked to top 5).

## Getting started

> Setup commands will be added once the one-command launch is in place.

1. Put the IS469 slide decks and lab notebooks in `data/`.
2. Set your Groq API key and choose settings in the config file: model ID, temperature, search mode, search-again switch and random seeds all live there.
3. Launch the app with a single command. The slides come pre-indexed.

AI responses are cached to disk, so reruns of the evaluation are free, repeatable and don't hit Groq rate limits.

## Evaluation

All results come from one frozen system (feature freeze after day 9).

| Area | Compare | Measure |
|---|---|---|
| Retrieval | Dense vs hybrid vs hybrid + rerank; search-again off vs on; best setup vs long-context baseline | Hit@5, Recall@5, answer correctness on an 80-question gold set; tokens and latency |
| Agent | Search-again off vs on | 30 test requests: task success, tool-call accuracy, steps, tokens, latency |
| Memory | None vs vector vs learner model + reflection | 6 simulated students with planted misconceptions, 4 sessions each |
| Usability | — | 6–8 classmates, 3 tasks each: completion, time, SUS score, feedback |
| Safety | Memory-write check on vs off | Poisoning rate over ~20 attempts |

The LLM judge comes from a different model family than the tutor, is validated against 30 hand-marked answers, and has answer order swapped to control for position bias. No BLEU/ROUGE.

## Guardrails

- Every claim cites a slide; faithfulness is checked by a different model family.
- Student beliefs and facts are stored separately; facts only with slide support.
- Step cap per turn and at most 3 searches per answer.
- Hints instead of answers in quiz mode; quiz-dump requests are refused.
- Students control their own memory; user-test logs are anonymised and collected with consent.

## Course topics used

- **Topic 8: Agentic RAG:** retrieval, hybrid search, reranking, citations, search-again
- **Topic 7: Agents with tools and memory:** tool calling, agent loop, harness, memory, reflection, guardrails
- **Topics 5–6: Evaluating LLMs:** LLM-as-judge, judge bias controls, cost and latency
- **Topic 1: Text processing:** BM25 builds on TF-IDF

## Team

| Owner | Area |
|---|---|
| A | Retrieval: ingestion, chunking, hybrid search, reranker, search-again, gold set, retrieval eval, long-context baseline, judge validation |
| B | Tutor and memory: agent loop and tools, quizzes and grading, learner model, reflection, agent test, simulated students |
| C | Harness, product and safety: Streamlit UI, citations, memory controls, trace file, saved responses, faithfulness check, poisoning test, user test, README |

## Future work

Fine-tuning and DPO, multimodal slide captioning, an MCP server, a multi-agent supervisor, and larger ablation studies.
