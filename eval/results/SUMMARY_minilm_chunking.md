# Chunking evaluation with MiniLM (text questions)

Run on 2026-11-05 with `eval/run_minilm_chunking.py`. No LLM or RAGAS was used. Raw scores are in `minilm_chunking.csv` and `minilm_chunking_questions.csv`.

## Result

| Format | Decision | Confidence |
|---|---|---|
| **Slides** | **One chunk per slide** | Strong: best on every dev metric, with the least text retrieved |
| **Documents** | **No winner yet.** Carry both fixed and recursive to the next step | Low: every difference on 24 dev questions is within noise, and the evidence winner flips with the slide method |

## Setup

- **Embedder:** `all-MiniLM-L6-v2`, text only, which reads up to 256 tokens per chunk.
- **Search:** dense, top 3 chunks. There are no image chunks.
- **Corpus:** 22 files, with student annotations removed.
- **Questions:** 210 text questions.

| Format | Dev | Test | Total |
|---|---|---|---|
| Slides | 58 | 103 | 161 |
| Documents | 24 | 25 | 49 |

**Chunking compared**

| Format | Options |
|---|---|
| Slides | Fixed 100/15, recursive 100/15, one chunk per slide |
| Documents | Fixed 200/30, recursive 200/30 |

Slides and documents share one index, so all 6 combinations were run. Three were requested, and three were added so that every possible winning combination was tested.

**Rule:** the winner is picked on dev by evidence recall@3, then MRR. Test scores are for reporting and were not used to choose.

**Split change:** the first version of this run had only 13 document dev questions, so 11 document text questions outside the LLM sample moved from test to dev (`ds-08`, `ds-14`, `ds-18`, `ds-20`, `ds-21`, `sd-02`, `sd-14`, `sd-15`, `sd-16`, `sd-18`, `sd-21`). All 11 were moved, with none chosen by score. Aggregate test scores from the first version had already been seen, so document test scores here are not fully untouched.

## The requested runs

Dev evidence recall@3 for each format:

| Run | Slides | Documents | Slides score | Documents score |
|---|---|---|---|---|
| Baseline | Fixed | Fixed | 0.756 | 0.826 |
| Run A | Recursive | Recursive | 0.825 | 0.806 |
| Run B | One per slide | Recursive | 0.838 | 0.785 |
| Added | One per slide | Fixed | 0.842 | 0.806 |

## Slides

Documents held at fixed 200/30.

| Method | Chunks | Slides per chunk | Words in top 3 | Dev evidence R@3 / MRR | Dev page R@3 / MRR | Test evidence R@3 / MRR | Test page R@3 / MRR |
|---|---|---|---|---|---|---|---|
| Fixed 100/15 (baseline) | 658 | 2.76 | 306 | 0.756 / 0.737 | 0.772 / 0.592 | 0.690 / 0.669 | 0.608 / 0.493 |
| Recursive 100/15 | 814 | 2.33 | 260 | 0.825 / 0.806 | 0.764 / 0.595 | 0.755 / 0.729 | 0.576 / 0.536 |
| **One per slide** | 983 | 1.00 | 202 | **0.842 / 0.864** | **0.842 / 0.864** | **0.788 / 0.799** | **0.798 / 0.808** |

**Question-by-question gain for one chunk per slide, on dev (58 questions, 95% bootstrap interval):**

| Metric | Against recursive | Against fixed |
|---|---|---|
| Evidence R@3 | +0.017 (-0.040 to +0.080) | +0.086 (-0.007 to +0.182) |
| Evidence MRR | +0.058 (-0.014 to +0.130) | +0.127 (+0.022 to +0.231) |
| Page R@3 | +0.078 (+0.006 to +0.152) | +0.070 (-0.003 to +0.151) |
| Page MRR | +0.269 (+0.188 to +0.350) | +0.272 (+0.178 to +0.365) |

**Findings**

- One chunk per slide scores highest on every dev metric, while retrieving about 22% less text than recursive and 34% less than fixed.
- Its evidence recall lead over recursive is small and within noise. The clear gains are in ranking and page-level retrieval.
- This settles the earlier conflict, where recursive won on evidence and one chunk per slide won on pages. That run used 200-word chunks with Gemma 2, so the change may come from the smaller chunks, the model, or the updated corpus and questions.
- The result held on test.

## Documents

| Slides held at | Method | Chunks | Words in top 3 | Dev evidence R@3 / MRR | Dev page R@3 / MRR | Test evidence R@3 / MRR | Test page R@3 / MRR |
|---|---|---|---|---|---|---|---|
| One per slide | Fixed 200/30 | 153 | 529 | **0.806** / 0.847 | 0.840 / 0.786 | 0.940 / 0.873 | 0.880 / 0.830 |
| One per slide | Recursive 200/30 | 172 | 480 | 0.785 / 0.812 | 0.924 / 0.795 | 0.920 / 0.888 | 0.880 / 0.790 |
| Fixed 100/15 | Fixed 200/30 | 153 | 568 | 0.826 / 0.868 | 0.861 / 0.788 | 0.940 / 0.893 | 0.880 / 0.850 |
| Fixed 100/15 | Recursive 200/30 | 172 | 515 | **0.847** / 0.847 | 0.944 / 0.816 | 0.980 / 0.873 | 0.840 / 0.760 |

**Recursive against fixed on dev, with slides at one per slide (24 questions):**

| Metric | Gain for recursive (95% interval) | Better / worse questions |
|---|---|---|
| Evidence R@3 | -0.021 (-0.188 to +0.125) | 5 / 4 |
| Evidence MRR | -0.035 (-0.177 to +0.111) | 4 / 5 |
| Page R@3 | +0.083 (-0.062 to +0.229) | 5 / 1 |
| Page MRR | +0.009 (-0.107 to +0.131) | 5 / 3 |

**Findings**

- Fixed wins the dev rule when slides use one chunk per slide, but recursive wins when slides use fixed chunks. The winner depends on the other format's setup.
- Every interval includes zero, and a single-part question changes the dev score by about 0.04.
- Recursive finds more of the right pages (page recall +0.083, 5 questions better and 1 worse), while evidence recall and MRR are about level. This is a hint only, because the interval includes zero.
- MiniLM truncation also affects this comparison (see below).
- The evidence does not support choosing either method, so both should move to the next step.

## MiniLM truncation

Chunks over 256 tokens are cut off before embedding, but evidence is still scored against the full chunk text.

| Chunking | Median tokens | Longest chunk | Chunks truncated | Tokens never embedded |
|---|---|---|---|---|
| Slides: fixed 100/15 | 140 | 284 | 0.3% | 0.1% |
| Slides: recursive 100/15 | 121 | 283 | 0.4% | 0.1% |
| Slides: one per slide | 81 | 302 | 0.5% | 0.1% |
| **Documents: fixed 200/30** | 249 | 577 | **38.6%** | **5.8%** |
| **Documents: recursive 200/30** | 233 | 626 | **19.8%** | **3.6%** |

Truncation has almost no effect on the slide comparison. It does affect documents, and it hits fixed chunks about twice as often as recursive ones, so fixed is at a disadvantage in this run.

## Limits of this run

- **The results are specific to MiniLM.** Winners may change with a stronger embedder, so they should be rechecked.
- **These scores cover text questions only.** Image questions and image chunks were not part of this step.
- **Evidence recall has a built-in bias.** Larger chunks have more chances to contain a quote, which is why words retrieved is reported alongside it.
- **Reachability is close to complete.** All evidence quotes can be found within a single chunk, except 0.4% under fixed 100/15 on slides.
- **The gold questions are not from real students.** They were written by an AI assistant to resemble student questions.
- **Sizes were not tuned.** One size per format was chosen from word counts, and other sizes were not tested.
- **Earlier results are not comparable.** `SUMMARY_retrieval.md` used a different embedder, chunk sizes, corpus and question wording.

## Next step

Compare Gemma 2 and Qwen on text questions, using one chunk per slide for slides and both fixed and recursive for documents. Gemma 2 reads 8,192 tokens and Qwen about 32K, so those runs should not have the truncation problem seen here. They should also give a fairer documents comparison, with 24 dev questions. Larger gains should be detectable, but a small difference may still be too close to call. If so, the simpler method or the one retrieving less text is a reasonable tie-break.
