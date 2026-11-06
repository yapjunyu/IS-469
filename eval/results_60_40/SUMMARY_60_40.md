# Retrieval comparison: final decision

**Final decision:** one chunk per slide, recursive 200/30 for documents, Gemma 2 with image chunks, and hybrid + rerank search, top 3 chunks. Each one is the winner of its stage under the dev rule on the 60/40 split.

| Stage | Winner | Dev R@3 | Runner-up | Strength of the win |
|---|---|---|---|---|
| Slide chunking | One chunk per slide | 0.834 (slide questions) | Recursive, 0.789 | Strong against the fixed baseline |
| Document chunking | Recursive 200/30 | 0.879 (document questions) | Fixed 150/22, 0.845 | Within noise |
| Embedder | Gemma 2 | 0.823 | Jina CLIP v2, 0.768 | Statistically clear |
| Search | Hybrid + rerank | 0.850 | Dense, 0.823 | At the edge of noise |

`src/tutor/pipeline.py` uses these settings.

## Setup

All results use `eval/gold/gold_60_40.csv`, with 268 answerable questions.

**How the split was made:** a random 60/40 split within each format and evidence type, using seed 469, drawn once. It replaced an earlier 38/62 split.

| Group | Dev | Test | Total | Dev share |
|---|---|---|---|---|
| **All questions** | **161** | **107** | **268** | **60.1%** |
| Text questions | 126 | 84 | 210 | 60.0% |
| Image questions | 35 | 23 | 58 | 60.3% |
| Slides, text | 97 | 64 | 161 | 60.2% |
| Slides, image | 27 | 18 | 45 | 60.0% |
| Documents, text | 29 | 20 | 49 | 59.2% |
| Documents, image | 8 | 5 | 13 | 61.5% |

**Winner rule:** the highest dev evidence recall@3 (R@3), then MRR, as before. Test is the reported result.

**Reading the tables:** cells are evidence scores at top 3 unless marked otherwise. R@3 is recall, P@3 is precision, and F1@3 combines the two. Each question has at most a few relevant chunks, so P@3 has a low ceiling. The best value in each row or column of a comparison is in bold. Gains are in bold when their bootstrap 95% interval excludes 0.

## Stage 1: Chunking

MiniLM, dense search, no image chunks, text questions only.

| Slides | Documents | Dev R@3 | Dev P@3 | Dev F1@3 | Dev MRR | Test R@3 | Test P@3 | Test F1@3 | Test MRR | All R@3 | All P@3 | All F1@3 | All MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed | Fixed (baseline) | 0.745 | 0.347 | 0.457 | 0.702 | 0.766 | 0.373 | 0.482 | 0.752 | 0.754 | 0.357 | 0.467 | 0.722 |
| Fixed | Recursive | 0.761 | 0.357 | 0.469 | 0.694 | 0.748 | 0.373 | 0.477 | 0.744 | 0.756 | 0.363 | 0.472 | 0.714 |
| Recursive | Fixed | 0.798 | 0.378 | 0.495 | 0.778 | **0.809** | **0.385** | **0.505** | 0.766 | 0.802 | 0.381 | 0.499 | 0.773 |
| Recursive | Recursive | 0.817 | 0.392 | 0.510 | 0.774 | 0.779 | 0.381 | 0.494 | 0.758 | 0.802 | **0.387** | 0.504 | 0.767 |
| One per slide | Fixed | 0.833 | 0.394 | 0.517 | **0.852** | **0.809** | 0.369 | 0.492 | **0.780** | **0.823** | 0.384 | **0.507** | **0.823** |
| **One per slide** | **Recursive (dev winner)** | **0.843** | **0.399** | **0.524** | 0.841 | 0.779 | 0.365 | 0.481 | 0.770 | 0.817 | 0.386 | **0.507** | 0.813 |

- **Slides:** one chunk per slide is well ahead of the fixed baseline on dev (+0.119 R@3, 95% CI +0.041 to +0.198, on slide questions).
- **Documents:** recursive and fixed are close. With one chunk per slide, recursive leads on dev (0.879 vs 0.828), and fixed leads on test (0.942 vs 0.817).

### Document chunk sizes

Slides stay at one chunk per slide. Results are for the 49 document text questions.

| Method and size | Dev R@3 | Dev P@3 | Dev F1@3 | Dev MRR | Test R@3 | Test P@3 | Test F1@3 | Test MRR | All R@3 | All P@3 | All F1@3 | All MRR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed 100/15 | 0.741 | 0.322 | 0.434 | 0.718 | 0.683 | 0.317 | 0.418 | 0.708 | 0.718 | 0.320 | 0.428 | 0.714 |
| Recursive 100/15 | 0.759 | 0.368 | 0.483 | 0.707 | 0.800 | 0.383 | 0.500 | 0.683 | 0.776 | 0.374 | 0.490 | 0.697 |
| Fixed 150/22 | 0.845 | 0.402 | 0.531 | 0.833 | 0.883 | **0.417** | 0.548 | 0.692 | 0.861 | **0.408** | **0.538** | 0.776 |
| Recursive 150/22 | 0.810 | 0.391 | 0.514 | 0.753 | 0.925 | **0.417** | 0.555 | 0.825 | 0.857 | 0.401 | 0.531 | 0.782 |
| Fixed 200/30 | 0.828 | 0.379 | 0.503 | **0.856** | **0.942** | **0.417** | **0.562** | **0.867** | **0.874** | 0.395 | 0.527 | **0.861** |
| **Recursive 200/30 (dev winner)** | **0.879** | **0.414** | **0.545** | 0.828 | 0.817 | 0.400 | 0.515 | 0.850 | 0.854 | **0.408** | 0.533 | 0.837 |

Fixed 200/30 minus recursive 200/30 is within noise on dev (R@3 -0.052, 95% CI -0.172 to +0.086, 29 questions).

## Stage 2: Embedder

The later stages use the stage 1 winners: one chunk per slide and recursive 200/30, which gives 1,155 text chunks and 515 image chunks.

### Text-only index

| Metric | MiniLM | Gemma 2 | Jina CLIP v2 |
|---|---|---|---|
| Dev R@3 | 0.843 | 0.877 | **0.886** |
| Dev P@3 | 0.399 | 0.415 | **0.426** |
| Dev F1@3 | 0.524 | 0.545 | **0.554** |
| Dev MRR | 0.841 | 0.872 | **0.874** |
| Test R@3 | 0.779 | 0.811 | **0.815** |
| Test P@3 | 0.365 | **0.389** | **0.389** |
| Test F1@3 | 0.481 | **0.510** | 0.509 |
| Test MRR | 0.770 | 0.784 | **0.833** |
| All R@3 | 0.817 | 0.850 | **0.858** |
| All P@3 | 0.386 | 0.405 | **0.411** |
| All F1@3 | 0.507 | 0.531 | **0.536** |
| All MRR | 0.813 | 0.837 | **0.858** |
| Embedding time | **5.8 s** | 215.1 s | 131.6 s |

Neither larger model is clearly ahead of MiniLM on dev: Gemma 2 gains +0.034 R@3 (95% CI -0.011 to +0.083), and Jina gains +0.043 (95% CI -0.005 to +0.091). MiniLM cannot embed images, so it cannot go forward as the multimodal embedder.

### Text and images together

| Metric | Gemma 2 | Jina CLIP v2 |
|---|---|---|
| **Dev R@3 (decides the winner)** | **0.823** | 0.768 |
| Dev P@3 | **0.381** | 0.360 |
| Dev F1@3 | **0.504** | 0.473 |
| Dev MRR | **0.793** | 0.753 |
| **Test R@3 (reported result)** | **0.776** | 0.715 |
| Test P@3 | **0.358** | 0.333 |
| Test F1@3 | **0.476** | 0.440 |
| Test MRR | **0.720** | 0.707 |
| All R@3 | **0.804** | 0.747 |
| All P@3 | **0.372** | 0.350 |
| All F1@3 | **0.493** | 0.460 |
| All MRR | **0.764** | 0.734 |
| Text questions R@3 (210) | 0.841 | **0.858** |
| Text questions P@3 (210) | 0.398 | **0.411** |
| Text questions F1@3 (210) | 0.523 | **0.536** |
| Image questions, answer R@3 (58) | **0.672** | 0.345 |
| Image questions, answer P@3 (58) | **0.276** | 0.126 |
| Image questions, answer F1@3 (58) | **0.383** | 0.183 |
| Image questions, figure R@3 (58) | **0.483** | 0.052 |
| Image questions, figure P@3 (58) | **0.167** | 0.017 |
| Image questions, figure F1@3 (58) | **0.247** | 0.026 |
| Embedding time | 1,327.1 s | **437.5 s** |

| Jina minus Gemma 2, R@3 | Dev gain (95% CI) | Test gain (95% CI) |
|---|---|---|
| All questions | **-0.055 (-0.106 to -0.004)** | -0.062 (-0.136 to +0.009) |
| Text questions | +0.021 (-0.019 to +0.063) | +0.011 (-0.058 to +0.079) |
| Image questions, answer | **-0.329 (-0.486 to -0.186)** | **-0.326 (-0.522 to -0.152)** |
| Image questions, figure | **-0.471 (-0.643 to -0.300)** | **-0.370 (-0.565 to -0.174)** |

**Gemma 2 wins**, and with more dev questions its overall lead is now statistically clear on dev. The gain comes from image questions: the two models are level on text.

## Stage 3: Search

Gemma 2 with image chunks. In hybrid + rerank, the cross-encoder cannot score images, so image chunks keep their dense positions.

| Metric | Dense | Hybrid | Hybrid + rerank |
|---|---|---|---|
| **Dev R@3 (decides the winner)** | 0.823 | 0.790 | **0.850** |
| Dev P@3 | 0.381 | 0.364 | **0.400** |
| Dev F1@3 | 0.504 | 0.483 | **0.526** |
| Dev MRR | 0.793 | 0.754 | **0.818** |
| **Test R@3 (reported result)** | **0.776** | 0.762 | 0.770 |
| Test P@3 | **0.358** | 0.355 | 0.355 |
| Test F1@3 | **0.476** | 0.470 | 0.472 |
| Test MRR | 0.720 | 0.710 | **0.769** |
| All R@3 | 0.804 | 0.779 | **0.818** |
| All P@3 | 0.372 | 0.361 | **0.382** |
| All F1@3 | 0.493 | 0.478 | **0.504** |
| All MRR | 0.764 | 0.736 | **0.799** |
| Text questions R@3 (210) | 0.841 | **0.889** | 0.863 |
| Text questions P@3 (210) | 0.398 | **0.425** | 0.411 |
| Text questions F1@3 (210) | 0.523 | **0.558** | 0.539 |
| Image questions, answer R@3 (58) | **0.672** | 0.379 | 0.655 |
| Image questions, answer P@3 (58) | **0.276** | 0.126 | **0.276** |
| Image questions, answer F1@3 (58) | **0.383** | 0.190 | 0.379 |
| Image questions, figure R@3 (58) | **0.483** | 0.069 | **0.483** |
| Image questions, figure P@3 (58) | **0.167** | 0.023 | **0.167** |
| Image questions, figure F1@3 (58) | **0.247** | 0.034 | **0.247** |
| Search time per question | **0.6 ms** | 3.2 ms | 82.6 ms |
| Average words in top 3 | **210.8** | 275.8 | 261.2 |

Search times come from one run on a laptop and exclude embedding the question, because question vectors were cached.

| Minus dense, R@3 | Hybrid, dev | Hybrid, test | Hybrid + rerank, dev | Hybrid + rerank, test |
|---|---|---|---|---|
| All questions | -0.033 (-0.080 to +0.016) | -0.014 (-0.087 to +0.056) | +0.027 (0.000 to +0.057) | -0.006 (-0.059 to +0.045) |
| Text questions | +0.025 (-0.020 to +0.071) | **+0.083 (+0.024 to +0.147)** | +0.035 (-0.001 to +0.073) | +0.004 (-0.058 to +0.065) |
| Image questions, answer | **-0.243 (-0.386 to -0.114)** | **-0.370 (-0.565 to -0.174)** | 0.000 (0.000 to 0.000) | -0.043 (-0.130 to 0.000) |
| Image questions, figure | **-0.414 (-0.571 to -0.257)** | **-0.413 (-0.609 to -0.217)** | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |

**Hybrid + rerank wins on dev.** Its gain over dense sits right at the edge of the interval, and it is level with dense on test R@3, although its test MRR is higher (+0.050, 95% CI -0.012 to +0.111). It also adds a keyword index, a second model and about 82 ms per question.

## Final decision and reasoning

We applied one rule at every stage: the highest dev evidence R@3, then MRR. Test scores are reported but did not pick any winner.

- **One chunk per slide:** it beats the fixed baseline on dev (+0.119 R@3 on slide questions, 95% CI +0.041 to +0.198). A slide is usually one idea, so this also avoids mixing topics across slides.
- **Recursive 200/30 for documents:** it has the highest dev R@3 of the document methods (0.879). The lead is within noise, and fixed 200/30 scores higher on test (0.942 vs 0.817).
- **Gemma 2:** its overall lead over Jina CLIP v2 is statistically clear on dev (+0.055 R@3, 95% CI +0.004 to +0.106), and it comes from image questions. Gemma 2 also runs in the main environment without custom remote code.
- **Hybrid + rerank:** it has the highest dev R@3 and MRR (0.850 and 0.818), and the highest test MRR (0.769). Its dev gain over dense is at the edge of noise (+0.027, 95% CI 0.000 to +0.057), and its test R@3 is level with dense (0.770 vs 0.776). It costs a keyword index, a second model and about 82 ms per question.

**Figures remain the weak spot:** the chosen setup finds the figure on under half of image questions (0.483), the same as dense, because the cross-encoder cannot read images.

## Limits

- **Two winners are narrow:** document chunking and search lead on dev but not on test R@3, so a different split could favour another method.
- **Small groups:** documents have only 29 dev and 20 test text questions, and 8 dev and 5 test image questions, so one question can move those scores a lot.
- **The RAGAS sample includes dev questions:** 45 of the 75 `llm_eval` questions are in dev on this split.

## Files

Everything is in `eval/results_60_40/`.

| File | Contents |
|---|---|
| `summary_all_stages.csv` | Scores for every setup, group and split |
| `paired_gains.csv` | Paired gains with 95% intervals |
| `gold_split.csv` | Split sizes by group |
| `*_questions.csv` and the other stage CSVs | Results saved by each stage |

## Reproduce

```bash
.venv/bin/python eval/run_minilm_chunking.py
.venv/bin/python eval/run_minilm_document_sizes.py
.venv/bin/python eval/run_text_embedders.py minilm
.venv/bin/python eval/run_text_embedders.py gemma2
.venv-jina/bin/python eval/run_text_embedders.py jinaclip2
.venv/bin/python eval/run_image_embedders.py gemma2
.venv-jina/bin/python eval/run_image_embedders.py jinaclip2
.venv/bin/python eval/run_search.py
.venv/bin/python eval/summarise_split.py
```
