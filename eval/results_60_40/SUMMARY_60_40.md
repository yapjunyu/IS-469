# Retrieval comparison: final decision

**Final decision:** one chunk per slide, recursive 200/30 for documents, Gemma 2 with image chunks, and hybrid + rerank search with slide merging, top 3 results. Each one is the winner of its stage under the dev rule on the 60/40 split.

**Slide merging:** a slide PNG shows the slide's own text, so a slide's text chunk and PNG chunk are merged into one result, and k = 3 counts results, not chunks. Document chunks are not merged, because a cropped figure does not repeat its page's text. Search fetches 2k chunks so that merging still leaves k results. The embedder stage with images and the final search are scored this way, as the app runs.

| Stage | Winner | Dev R@3 | Runner-up | Strength of the win |
|---|---|---|---|---|
| Slide chunking | One chunk per slide | 0.834 (slide questions) | Recursive, 0.789 | Strong against the fixed baseline |
| Document chunking | Recursive 200/30 | 0.879 (document questions) | Fixed 150/22, 0.845 | Within noise |
| Embedder | Gemma 2 | 0.851 | Jina CLIP v2, 0.774 | Statistically clear |
| Search | Hybrid + rerank, slides merged | 0.866 | Hybrid + rerank, 0.850 | Small but never worse (0 of 161 dev questions) |

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

Dense search, with slide merging.

| Metric | Gemma 2 | Jina CLIP v2 |
|---|---|---|
| **Dev R@3 (decides the winner)** | **0.851** | 0.774 |
| Dev P@3 | **0.381** | 0.362 |
| Dev F1@3 | **0.510** | 0.476 |
| Dev MRR | **0.831** | 0.759 |
| **Test R@3 (reported result)** | **0.790** | 0.734 |
| Test P@3 | **0.358** | 0.336 |
| Test F1@3 | **0.479** | 0.446 |
| Test MRR | **0.743** | 0.726 |
| All R@3 | **0.827** | 0.758 |
| All P@3 | **0.372** | 0.352 |
| All F1@3 | **0.497** | 0.464 |
| All MRR | **0.796** | 0.746 |
| Text questions R@3 (210) | 0.848 | **0.858** |
| Text questions P@3 (210) | 0.403 | **0.411** |
| Text questions F1@3 (210) | 0.529 | **0.536** |
| Image questions, answer R@3 (58) | **0.750** | 0.397 |
| Image questions, answer P@3 (58) | **0.259** | 0.138 |
| Image questions, answer F1@3 (58) | **0.383** | 0.203 |
| Image questions, figure R@3 (58) | **0.647** | 0.172 |
| Image questions, figure P@3 (58) | **0.224** | 0.063 |
| Image questions, figure F1@3 (58) | **0.331** | 0.091 |
| Embedding time | 1,327.1 s | **437.5 s** |

| Jina minus Gemma 2, R@3 | Dev gain (95% CI) | Test gain (95% CI) |
|---|---|---|
| All questions | **-0.077 (-0.131 to -0.025)** | -0.057 (-0.128 to +0.012) |
| Text questions | +0.013 (-0.026 to +0.054) | +0.005 (-0.063 to +0.071) |
| Image questions, answer | **-0.400 (-0.571 to -0.257)** | **-0.283 (-0.478 to -0.109)** |
| Image questions, figure | **-0.514 (-0.686 to -0.343)** | **-0.413 (-0.609 to -0.217)** |

**Gemma 2 wins**, and its overall lead is statistically clear on dev. The gain comes from image questions: the two models are level on text. Without slide merging the result was the same (Gemma 2 0.823, Jina 0.768 dev R@3); merging lifts Gemma 2 more, because more of its slide PNGs rank next to their own text.

## Stage 3: Search

Gemma 2 with image chunks. In hybrid + rerank, the cross-encoder cannot score images, so image chunks keep their dense positions. The three search methods are compared without slide merging; the last column adds merging to the winner (stage 3b below).

| Metric | Dense | Hybrid | Hybrid + rerank | Hybrid + rerank, slides merged |
|---|---|---|---|---|
| **Dev R@3 (decides the winner)** | 0.823 | 0.790 | 0.850 | **0.866** |
| Dev P@3 | 0.381 | 0.364 | **0.400** | 0.393 |
| Dev F1@3 | 0.504 | 0.483 | **0.526** | 0.524 |
| Dev MRR | 0.793 | 0.754 | 0.818 | **0.847** |
| **Test R@3 (reported result)** | 0.776 | 0.762 | 0.770 | **0.789** |
| Test P@3 | **0.358** | 0.355 | 0.355 | 0.355 |
| Test F1@3 | **0.476** | 0.470 | 0.472 | 0.475 |
| Test MRR | 0.720 | 0.710 | 0.769 | **0.798** |
| All R@3 | 0.804 | 0.779 | 0.818 | **0.835** |
| All P@3 | 0.372 | 0.361 | **0.382** | 0.378 |
| All F1@3 | 0.493 | 0.478 | **0.504** | **0.504** |
| All MRR | 0.764 | 0.736 | 0.799 | **0.827** |
| Text questions R@3 (210) | 0.841 | **0.889** | 0.863 | 0.868 |
| Text questions P@3 (210) | 0.398 | **0.425** | 0.411 | 0.414 |
| Text questions F1@3 (210) | 0.523 | **0.558** | 0.539 | 0.543 |
| Image questions, answer R@3 (58) | 0.672 | 0.379 | 0.655 | **0.716** |
| Image questions, answer P@3 (58) | **0.276** | 0.126 | **0.276** | 0.247 |
| Image questions, answer F1@3 (58) | **0.383** | 0.190 | 0.379 | 0.366 |
| Image questions, figure R@3 (58) | 0.483 | 0.069 | 0.483 | **0.612** |
| Image questions, figure P@3 (58) | 0.167 | 0.023 | 0.167 | **0.213** |
| Image questions, figure F1@3 (58) | 0.247 | 0.034 | 0.247 | **0.314** |
| Search time per question | **0.6 ms** | 3.2 ms | 92.4 ms | about the same as hybrid + rerank |
| Average words in top 3 | **210.8** | 275.8 | 261.2 | 272.8 |

Search times come from one run on a laptop and exclude embedding the question, because question vectors were cached. The merged run measured 4.4 ms only because it reused the reranker scores already computed for hybrid + rerank in the same run; merging itself adds well under a millisecond.

| Minus dense, R@3 | Hybrid, dev | Hybrid, test | Hybrid + rerank, dev | Hybrid + rerank, test |
|---|---|---|---|---|
| All questions | -0.033 (-0.080 to +0.016) | -0.014 (-0.087 to +0.056) | +0.027 (0.000 to +0.057) | -0.006 (-0.059 to +0.045) |
| Text questions | +0.025 (-0.020 to +0.071) | **+0.083 (+0.024 to +0.147)** | +0.035 (-0.001 to +0.073) | +0.004 (-0.058 to +0.065) |
| Image questions, answer | **-0.243 (-0.386 to -0.114)** | **-0.370 (-0.565 to -0.174)** | 0.000 (0.000 to 0.000) | -0.043 (-0.130 to 0.000) |
| Image questions, figure | **-0.414 (-0.571 to -0.257)** | **-0.413 (-0.609 to -0.217)** | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |

**Hybrid + rerank wins among the three methods on dev.** Its gain over dense sits right at the edge of the interval, and it is level with dense on test R@3, although its test MRR is higher (+0.050, 95% CI -0.012 to +0.111). It also adds a keyword index, a second model and about 90 ms per question.

### Stage 3b: Slide merging

Hybrid + rerank, with and without a slide's text and PNG merged into one result.

| Merged minus not merged, R@3 | Dev gain (95% CI) | Dev better / worse | Test gain (95% CI) | Test better / worse |
|---|---|---|---|---|
| All questions | +0.016 (0.000 to +0.037) | 3 / 0 | +0.019 (0.000 to +0.042) | 3 / 0 |
| Text questions | 0.000 (0.000 to 0.000) | 0 / 0 | +0.012 (0.000 to +0.030) | 2 / 0 |
| Image questions, answer | +0.071 (0.000 to +0.157) | 3 / 0 | +0.043 (0.000 to +0.130) | 1 / 0 |
| Image questions, figure | **+0.157 (+0.057 to +0.286)** | 6 / 0 | +0.087 (0.000 to +0.217) | 2 / 0 |
| All questions, MRR | **+0.029 (+0.009 to +0.054)** | | **+0.028 (+0.006 to +0.059)** | |

**Merging is kept.** It never makes a question worse, on dev or test, and it costs nothing at search time. Against dense, merged hybrid + rerank is now statistically clear on dev (+0.043 R@3, 95% CI +0.012 to +0.079) and ahead on test (+0.012, 95% CI -0.044 to +0.067).

The gain is not where duplicates were expected to hurt: text questions are almost unchanged. It comes from image questions. When a slide's text ranks in the top 3, its PNG now comes with it, so the figure reaches the LLM even when the PNG's own embedding ranked too low. Documents are unchanged in every group, as expected.

## Final decision and reasoning

We applied one rule at every stage: the highest dev evidence R@3, then MRR. Test scores are reported but did not pick any winner.

- **One chunk per slide:** it beats the fixed baseline on dev (+0.119 R@3 on slide questions, 95% CI +0.041 to +0.198). A slide is usually one idea, so this also avoids mixing topics across slides.
- **Recursive 200/30 for documents:** it has the highest dev R@3 of the document methods (0.879). The lead is within noise, and fixed 200/30 scores higher on test (0.942 vs 0.817).
- **Gemma 2:** its overall lead over Jina CLIP v2 is statistically clear on dev (+0.077 R@3, 95% CI +0.025 to +0.131), and it comes from image questions. Gemma 2 also runs in the main environment without custom remote code.
- **Hybrid + rerank:** it has the highest dev R@3 and MRR of the three search methods (0.850 and 0.818), and the highest test MRR (0.769). Its dev gain over dense is at the edge of noise (+0.027, 95% CI 0.000 to +0.057), and its test R@3 is level with dense (0.770 vs 0.776). It costs a keyword index, a second model and about 90 ms per question.
- **Slide merging:** it raises dev R@3 to 0.866 and MRR to 0.847, never makes a question worse, and costs nothing at search time. With it, the final setup's lead over dense is statistically clear on dev (+0.043, 95% CI +0.012 to +0.079).

**Figures remain the weak spot:** the chosen setup finds the figure on 0.612 of image questions (dense: 0.483), so about 4 in 10 figures are still missed. The cross-encoder cannot read images, and document figures (0.462) gain nothing from merging.

## Answer quality (RAGAS)

The final pipeline answered the 75 `llm_eval` questions, and RAGAS scored the answers. **Generator:** `gpt-6-luna`, low reasoning effort. **Judge:** `claude-haiku-5-5`, a different model, so it doesn't grade its own answers. The 75 questions are 55 text and 20 image, 15 per collection, 45 dev and 30 test. No metric failed on any question.

| Group | Questions | Faithfulness | Factual correctness | Context precision | Context recall |
|---|---|---|---|---|---|
| **All** | 75 | **0.987** | **0.706** | **0.953** (n=55) | **0.896** (n=55) |
| Dev | 45 | 1.000 | 0.672 | 0.941 (n=34) | 0.890 (n=34) |
| Test | 30 | 0.967 | 0.758 | 0.972 (n=21) | 0.906 (n=21) |
| Text | 55 | 1.000 | 0.759 | 0.953 | 0.896 |
| Image | 20 | 0.950 | 0.560 | – | – |

**What each metric checks:** faithfulness checks that the answer's claims are supported by the retrieved passages and pictures. Factual correctness compares the answer's claims with the reference answer, as an F1. Context precision checks that useful passages rank first, and context recall checks that the passages cover the reference. The two context metrics read only text, so image questions skip them.

**Findings:**

- **The tutor stays grounded:** 74 of 75 answers are fully faithful. The prompt asks for a citation on every claim and forbids outside knowledge, and the scores show it works. The one failure is `sd-36` (score 0.0). Its answer lists HTTP methods and names JSON as the most common format, and the judge found no support for these in the passages or the figure crop, so it needs a manual check.
- **Retrieval holds up in the end-to-end setting:** context precision is 0.953 and context recall 0.896 on text questions, in line with the retrieval evaluation. Low recall comes from partial misses on questions with several parts. For example, in `ai-15` (0.25), only the residual-connection slide was retrieved, and the answer correctly says the passages don't cover layer norm.
- **Test is not worse than dev:** test scores are at or above dev on every metric. So the retrieval choices, tuned on dev, show no sign of overfitting dev.
- **Factual correctness is lower than the other metrics, as expected:** it is an F1 against a short reference, so a correct extra detail lowers it. Answers below 0.4 are mostly image questions.
- **Image questions are the weak spot, as in the retrieval evaluation:** factual correctness is 0.560 against 0.759 for text. The gap is 0.198 (95% bootstrap CI +0.057 to +0.343), so it is real, not noise. Two causes:
  - **No picture reached the generator** on 5 of the 20 questions (factual correctness 0.360, against 0.627 with a picture). For `ai-101` and `ai-104`, the right page was retrieved as text only. For `cl-96` and `ds-26`, retrieval missed the gold page (0.180 over the 2 questions where the gold page was missed).
  - **The answer read the caption but not the diagram:** in `cl-87` (0.25), the right slide and its picture were retrieved, but the answer repeats the caption ("CSP or 3rd party network provider") and leaves out what the diagram shows (colocation facility, Zone 1). The prompt's "if they are brief, keep the answer brief" may cut answers short when a slide's text is only a caption.
- **The judge is not perfect:** `ai-103` scores 0.0 factual correctness although its token counts (6, 8 and 25) match the reference. One judge model scoring 75 questions has some noise, so read single question scores with care.

**Picture prompt experiment (not kept):** we added a line to the prompt that is used only when a picture is attached: "When a picture helps answer the question, describe what it shows (labels, steps, values, positions), not only its caption, and cite it." Text-only prompts stayed the same. The same rule as in the retrieval stages decided: keep it only if dev image factual correctness rises while faithfulness stays near 1. 27 answers changed, all of them questions with a retrieved picture.

| Group | Metric | Before | After |
|---|---|---|---|
| **Dev image (11), the deciding group** | Factual correctness | **0.470** | **0.472** |
| Dev image | Faithfulness | 1.000 | 1.000 |
| Dev text (34) | Factual correctness | 0.737 | 0.727 |
| Test image (9) | Factual correctness | 0.671 | 0.671 |
| All (75) | Factual correctness | 0.706 | 0.700 |

Dev image factual correctness did not move, so the line was removed and the results above are from the original prompt. Single questions swung both ways (`ai-103` 0.00 to 0.89, `ds-24` 0.57 to 0.14), and faithfulness swapped between `sd-36` and `pm-49`. The generator has no temperature setting and the judge has some noise, so a change in one question's score means little. **The model already uses pictures:** it cites the attached picture in 13 of the 15 image answers that had one. So the image gap comes mostly from retrieval: no picture reached the generator on 5 of the 20, and those score 0.360. The prompt is not the main cause. **A next step:** look up a slide's PNG whenever its text is retrieved. Slide merging attaches the PNG only when the PNG chunk is also among the search candidates. In `ai-101` and `ai-104`, the gold slide's PNG is stored but was not a candidate, so the generator got the text only.

**Cost:** about 92k generator tokens and 1.33M judge tokens, under $0.30. The prompt experiment added about 0.5M judge tokens. Answers and judge replies are cached, so rerunning unchanged questions is free.

## Limits

- **Two winners are narrow:** document chunking leads on dev but not on test R@3, and the final search's test gain over dense is within noise (+0.012, 95% CI -0.044 to +0.067), so a different split could favour another method.
- **Slide merging was added after the comparisons:** chunking and the text-only embedder runs have no slide PNGs, so merging cannot change them. The embedder run with images was re-scored with merging, and Gemma 2 wins either way. The three search methods were compared without merging, and merging was then tested on the winner.
- **Small groups:** documents have only 29 dev and 20 test text questions, and 8 dev and 5 test image questions, so one question can move those scores a lot.
- **The RAGAS sample includes dev questions:** 45 of the 75 `llm_eval` questions are in dev on this split. Dev and test are reported separately.
- **RAGAS image results are on 20 questions:** that is enough to show the gap with text, but a single question moves the image mean by about 0.05. Image questions get only faithfulness and factual correctness.

## Files

Everything is in `eval/results_60_40/`.

| File | Contents |
|---|---|
| `summary_all_stages.csv` | Scores for every setup, group and split |
| `paired_gains.csv` | Paired gains with 95% intervals |
| `gold_split.csv` | Split sizes by group |
| `*_questions.csv` and the other stage CSVs | Results saved by each stage |
| `generation_sample.json` | The 75 answers, with retrieved passages and pictures |
| `ragas_sample.json`, `ragas_sample.csv`, `ragas_sample_questions.csv` | RAGAS scores: raw, by group, and per question |

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
.venv/bin/python eval/run_generation.py      # needs OPENCODE_API_KEY
.venv/bin/python eval/ragas_eval.py sample
```
