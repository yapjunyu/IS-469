# Retrieval evaluation results

All numbers come from the run on 2026-11-05. Every step is in `summary_all_steps.csv`, and each step has its own CSV with per-question scores.

## Setup

**Corpus (17 files):** 8 cloud lesson decks and 7 GenAI topic decks (slides), plus OpenStax *Principles of Data Science* Ch 2 and *Software Development* Ch 9, 13 and 19 (documents). PPTX speaker notes are not indexed.

**Gold set:** 206 answerable questions, plus 5 out-of-scope questions kept for refusal testing.

| | Text | Image | Total |
|---|---|---|---|
| Dev | 53 | 15 | 68 |
| Test | 108 | 30 | 138 |
| Total | 161 | 45 | 206 |

| Topic | Text | Image | Total |
|---|---|---|---|
| GenAI (slides) | 61 | 16 | 77 |
| Cloud (slides) | 51 | 16 | 67 |
| Software Development (document) | 27 | 9 | 36 |
| Data Science (document) | 22 | 4 | 26 |

**Method:** one component is tuned at a time, keeping the earlier winners. Winners are picked on dev by evidence recall@3, then MRR. Test scores are for reporting.

## How scoring works

**Evidence score (main).** Search returns the top k chunks. A chunk is relevant if it comes from a gold page and:
- for a text question, contains that part's evidence quote;
- for an image question, is an image chunk.

This works for any chunking method because it checks the text, not a chunk ID.

**Page score (backup).** The ranked chunks are expanded into pages, and the first k distinct pages are scored against the gold pages.

- **Recall@k:** share of the question's parts found.
- **Precision@k:** share of the k returned chunks or pages that are relevant.
- **F1:** calculated per question, then averaged.
- **MRR:** 1 / rank of the first relevant result, within the top 5.

A text-only index scores 0 on evidence for image questions, because it has no image chunk to find.

## Baseline

Fixed 200-word chunks with 30-word overlap everywhere, Gemma 2, text only, dense search. It has 401 chunks.

| Group | n | Evidence R@3 | P@3 | F1@3 | MRR | Page R@3 | Page MRR |
|---|---|---|---|---|---|---|---|
| All | 206 | 0.713 | 0.333 | 0.440 | 0.704 | 0.692 | 0.526 |
| Slides | 144 | 0.698 | 0.317 | 0.423 | 0.706 | 0.607 | 0.413 |
| Documents | 62 | 0.747 | 0.371 | 0.478 | 0.699 | 0.890 | 0.788 |
| Text questions | 161 | 0.912 | 0.427 | 0.563 | 0.901 | 0.724 | 0.586 |
| Image questions | 45 | 0.000 | 0.000 | 0.000 | 0.000 | 0.578 | 0.313 |

Test only, all questions: evidence recall@3 0.709, MRR 0.692; page recall@3 0.717, MRR 0.554.

## Step 1: Chunking

Gemma 2, dense, text only. Evidence scores use text questions only, and page scores use all questions.

**Slides**

| Method | Chunks | Slides per chunk | Evidence R@3 | P@3 | MRR | Page R@3 | Page MRR | Dev evidence R@3 / MRR |
|---|---|---|---|---|---|---|---|---|
| One per slide | 689 | 1.00 | 0.873 | 0.432 | 0.852 | **0.901** | **0.876** | 0.900 / 0.871 |
| Fixed | 248 | 4.25 | 0.897 | 0.408 | 0.908 | 0.607 | 0.413 | 0.912 / 0.940 |
| **Recursive (dev winner)** | 306 | 3.85 | **0.926** | **0.458** | **0.912** | 0.590 | 0.430 | **0.963 / 0.958** |

**Documents**

| Method | Chunks | Pages per chunk | Evidence R@3 | P@3 | MRR | Page R@3 | Page MRR | Dev evidence R@3 / MRR |
|---|---|---|---|---|---|---|---|---|
| **Fixed (dev winner)** | 153 | 1.73 | **0.946** | **0.469** | **0.884** | 0.890 | **0.788** | **0.962 / 0.923** |
| Recursive | 172 | 1.61 | 0.905 | 0.435 | 0.852 | 0.890 | 0.752 | 0.923 / 0.872 |

### Open decision: recursive or one chunk per slide?

The two scores disagree for slides. Recursive wins on evidence, but one chunk per slide wins clearly on pages. Recursive chunks are bigger, so its top 3 chunks return much more text:

| Slide chunking | Words per chunk | Words in top 3 chunks |
|---|---|---|
| One per slide | 60 | 160 |
| Recursive | 163 | 391 |
| Fixed | 194 | 449 |

Part of recursive's evidence lead probably comes from returning about 2.4 times as much text, not from better search.

Full pipeline comparison (documents fixed, Gemma 2 + images, dense), on test and all questions:

| Slide chunking | Evidence R@3 | Evidence MRR | Page R@3 | Page MRR |
|---|---|---|---|---|
| Recursive (current) | 0.878 | 0.816 | 0.810 | 0.713 |
| One per slide | 0.856 | 0.792 | 0.916 | 0.882 |

The rest of this file uses recursive for slides, because that's what the dev rule chose. If one chunk per slide is chosen, steps 2 and 3 need rerunning for slides.

## Step 2: Embedder and images

Slides recursive, documents fixed, dense search. Results are for all 206 questions.

| Config | Evidence R@3 | P@3 | F1@3 | MRR | Dev R@3 / MRR | Test R@3 / MRR | Page R@3 | Embedding time |
|---|---|---|---|---|---|---|---|---|
| Gemma 2, text | 0.728 | 0.361 | 0.466 | 0.706 | 0.750 / 0.740 | 0.717 / 0.690 | 0.680 | 136 s |
| **Gemma 2, + images (dev winner)** | 0.896 | 0.414 | 0.548 | 0.844 | **0.934** / 0.899 | 0.878 / 0.816 | 0.785 | 990 s |
| Qwen3-VL, text | 0.755 | 0.380 | 0.488 | 0.733 | 0.746 / 0.765 | 0.760 / 0.717 | 0.682 | 435 s |
| Qwen3-VL, + images | **0.926** | **0.432** | **0.570** | **0.867** | 0.930 / **0.908** | **0.924 / 0.847** | 0.771 | 2,845 s |

Embedding time was measured on an Apple M2 and excludes reading files and rendering slides.

| Per item | Gemma 2 | Qwen3-VL |
|---|---|---|
| Image | 2.3 s | about 6.9 s |
| Text chunk | 0.24 s | 0.95 s |
| Question | 0.05 s | 0.22 s |

**By question type (evidence R@3 / MRR):**

| Config | Text questions | Image questions |
|---|---|---|
| Gemma 2, text | 0.932 / 0.904 | 0.000 / 0.000 |
| Gemma 2, + images | 0.911 / 0.865 | 0.844 / 0.767 |
| Qwen3-VL, text | 0.966 / 0.937 | 0.000 / 0.000 |
| Qwen3-VL, + images | 0.961 / 0.919 | 0.800 / 0.684 |

**Findings:**
- Image chunks make figure questions answerable, from 0 to about 0.8.
- They cost text questions a little, because image chunks can take top-3 slots: Gemma 2 drops from 0.932 to 0.911.
- The dev gap between the embedders is tiny (0.934 vs 0.930), while Qwen is ahead on test (0.924 vs 0.878). So Gemma 2's win is not strong evidence that it retrieves better. Its case rests on the dev rule and on indexing about 2.9 times faster.

## Step 3: Search

Slides recursive, documents fixed, Gemma 2 + images. Results are for all 206 questions.

| Search | Evidence R@3 | P@3 | F1@3 | MRR | Dev R@3 / MRR | Test R@3 / MRR | Page R@3 | Page MRR |
|---|---|---|---|---|---|---|---|---|
| **Dense (dev winner)** | **0.896** | **0.414** | **0.548** | **0.844** | **0.934 / 0.899** | **0.878 / 0.816** | **0.785** | **0.684** |
| Hybrid (RRF) | 0.737 | 0.359 | 0.467 | 0.717 | 0.754 / 0.728 | 0.728 / 0.712 | 0.680 | 0.527 |
| BM25 | 0.697 | 0.332 | 0.435 | 0.680 | 0.658 / 0.680 | 0.716 / 0.680 | 0.617 | 0.470 |

| Evidence R@3 / MRR | Dense | Hybrid | BM25 |
|---|---|---|---|
| Text questions | 0.911 / 0.865 | **0.933 / 0.905** | 0.892 / 0.870 |
| Image questions | **0.844 / 0.767** | 0.033 / 0.046 | 0.000 / 0.000 |

**Findings:**
- Hybrid is better on text questions but almost never returns an image chunk in the top 3. BM25 can't see image chunks, and RRF rewards chunks that appear in both lists.
- That's why dense wins overall, even though hybrid is stronger on text.
- A future option is to rank text and image chunks separately, then combine them. This has not been tested.

## Final pipeline vs baseline

Final pipeline under the dev rule: slides recursive, documents fixed, Gemma 2 + images, dense search.

| Test questions | Baseline evidence R@3 / MRR | Final evidence R@3 / MRR | Baseline page R@3 / MRR | Final page R@3 / MRR |
|---|---|---|---|---|
| All (138) | 0.709 / 0.692 | **0.878 / 0.816** | 0.717 / 0.554 | **0.810 / 0.713** |
| Text questions (108) | **0.906 / 0.884** | 0.890 / 0.837 | 0.749 / 0.617 | **0.776 / 0.693** |
| Image questions (30) | 0.000 / 0.000 | **0.833 / 0.742** | 0.600 / 0.328 | **0.933 / 0.786** |

**The overall gain comes mainly from image questions.** On text questions, the final pipeline is slightly below the baseline on evidence, and above it on pages.

## Notes for the report

- **Precision ceiling:** many questions have one needed passage, so precision@3 usually tops out near 0.33. Recall and MRR matter more.
- **Big chunks:** long chunks hold more text, so they have more chances to contain a quote and cover a gold page. Compare slides and documents separately.
- **Evidence quotes:** one quote isn't the whole answer, so a pass shows the key text was retrieved, not that the answer is complete. RAGAS is still needed to check the LLM's answers.
- **Gold set:** questions were written by an AI assistant rather than the instructor or students, and hand-checking covered only part of the set. Wording may sit close to the notes, which could help retrieval.
- **Small groups:** dev has only 15 image questions and 6 Data Science questions, so don't draw conclusions from those groups alone.
- **Cross-file answers:** only pages reviewed in the audit count, so a few valid answers elsewhere may be scored as misses.
- **Tuning order:** one-at-a-time tuning can miss interactions between components, such as hybrid with one chunk per slide.
- **Corpus:** the Software Development extract has short pages, so document chunks often span several pages.
