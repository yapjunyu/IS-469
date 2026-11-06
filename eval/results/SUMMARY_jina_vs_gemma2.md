# Embedder and search comparison

**Embedder: Gemma 2 goes forward.** It wins on the dev rule (evidence recall@3, then MRR), and the test split supports that, mainly because it finds image evidence far better.

**Search: dense goes forward.** It wins on the same rule, with test evidence R@3 of 0.841 and MRR of 0.787. Its dev lead over hybrid + rerank is within noise, so simplicity also favours it. See [Search methods](#search-methods).

## Setup

| Setting | Value |
|---|---|
| Models | `google/embeddinggemma-2` (`gemma2`), `jinaai/jina-clip-v2` (`jinaclip2`) |
| Slides | One chunk per slide, plus image chunks |
| Documents | Fixed 200 words, 30-word overlap, plus image chunks |
| Search | Dense, top 3, text and image chunks in one index |
| Index | 1,136 text chunks, 515 image chunks |
| Questions | 268 (210 text, 58 image) |
| Winner rule | Best on dev: evidence recall@3, then MRR. Test is the reported result. |

## Overall results

Results are for all 268 questions unless the row names a split. The better value in each row is in bold.

| Metric | Gemma 2 | Jina CLIP v2 |
|---|---|---|
| **Dev R@3 (decides the winner)** | **0.765** | 0.759 |
| Dev MRR | 0.740 | **0.760** |
| **Test R@3 (reported result)** | **0.841** | 0.747 |
| Test MRR | **0.787** | 0.719 |
| R@3 | **0.812** | 0.752 |
| P@3 | **0.374** | 0.354 |
| F1@3 | **0.497** | 0.465 |
| R@5 | **0.883** | 0.817 |
| MRR | **0.769** | 0.734 |

## By question type

Image questions get two scores:
- **Answer:** a retrieved chunk holds the answer. This can be an image chunk on the gold page, or any chunk on a page listed in `also_gold`.
- **Figure:** an image chunk on the gold page was retrieved, ignoring `also_gold`.

| Group | Questions | Gemma 2 R@3 | Jina R@3 | Gemma 2 MRR | Jina MRR |
|---|---|---|---|---|---|
| Text questions | 210 | 0.855 | **0.864** | 0.844 | **0.860** |
| Image questions, answer | 58 | **0.655** | 0.345 | **0.497** | 0.279 |
| Image questions, figure | 58 | **0.466** | 0.052 | **0.253** | 0.029 |

## By format

| Format | Questions | Gemma 2 R@3 | Jina R@3 | Gemma 2 MRR | Jina MRR |
|---|---|---|---|---|---|
| Slides | 206 | **0.823** | 0.756 | **0.794** | 0.747 |
| Documents | 62 | **0.774** | 0.737 | 0.685 | **0.694** |

## Paired difference on dev

Jina minus Gemma 2 on evidence R@3, question by question, with a bootstrap 95% interval.

| Group | Questions | Gain (95% interval) | Reading |
|---|---|---|---|
| All questions | 102 | -0.006 (-0.074 to +0.059) | Tie |
| Text questions | 82 | +0.042 (-0.027 to +0.115) | Tie |
| Image questions, answer | 20 | **-0.200 (-0.375 to -0.050)** | Gemma 2 better |
| Image questions, figure | 20 | **-0.400 (-0.600 to -0.200)** | Gemma 2 better |

## Text-only index

The same chunking and text questions, with no image chunks in the index.

| Format | Questions | Gemma 2 R@3 | Jina R@3 | Gemma 2 MRR | Jina MRR |
|---|---|---|---|---|---|
| Slides | 161 | **0.861** | 0.849 | 0.858 | **0.861** |
| Documents | 49 | 0.888 | **0.912** | 0.813 | **0.857** |

## Why Jina finds so few figures

Jina's text-to-image similarity scores run much lower than its text-to-text scores, so figures rarely reach a shared top 3. Searching image chunks alone recovers much of its score, but Gemma 2 still leads.

| Image questions (58) | Gemma 2 | Jina CLIP v2 |
|---|---|---|
| Figure R@3, images-only search | **0.922** | 0.750 |
| Mean best image similarity | 0.745 | 0.372 |
| Mean third-best text similarity | 0.748 | 0.486 |

This test covers all 58 image questions, including non-dev ones. It explains the gap and was not used to pick the winner. The script for it was not added to the repo.

## Practical differences

| | Gemma 2 | Jina CLIP v2 |
|---|---|---|
| Parameters | 740M | 865M |
| Embedding time, text + images | 1,318 s | **432 s** |
| Average words in top 3 | **216.6** | 295.1 |
| Environment | `.venv` (transformers 5) | `.venv-jina` (transformers 4.46.2) |
| Custom remote code | No | Yes, from its Hugging Face repos |
| License | Open weights (check Google's terms before commercial use) | CC BY-NC 4.0, non-commercial |

Embedding times come from separate runs on the same laptop, so treat them as a rough guide. Jina returns more words because fewer image chunks reach its top 3.

## Embedder conclusion

- **Why Gemma 2 wins:** the dev margin is tiny (0.765 vs 0.759) and Jina's dev MRR is higher, so dev alone is close to a tie. The clear difference is image questions, where Gemma 2's lead holds up statistically and shows up on both dev and test.
- **Where Jina is better:** it is slightly ahead on text questions, though the R@3 difference could be chance, and it embeds about 3 times faster.
- **Practical advantage:** Gemma 2 runs in the main environment and needs no custom remote code.

**Limits:** dev has only 20 image questions, so those intervals are wide. Gemma 2 won on recall, not on every measure, so it is the better all-round model rather than better across the board.

## Search methods

The chunking and embedder winners are fixed: one chunk per slide, fixed 200 words with 30-word overlap for documents, and Gemma 2 with image chunks. Results use the same 268 questions and the same scoring as above.

| Method | How it ranks chunks |
|---|---|
| Dense | Embeddings and cosine similarity |
| Hybrid | Dense and BM25 rankings merged with RRF, 20 candidates from each |
| Hybrid + rerank | The top 20 text chunks from hybrid, reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`. The cross-encoder cannot score images, so image chunks keep their dense positions. |

### Overall results

| Metric | Dense | Hybrid | Hybrid + rerank |
|---|---|---|---|
| **Dev R@3 (decides the winner)** | **0.765** | 0.755 | 0.764 |
| Dev MRR | **0.740** | 0.722 | **0.740** |
| **Test R@3 (reported result)** | 0.841 | 0.783 | **0.855** |
| Test MRR | 0.787 | 0.740 | **0.821** |
| R@3 | 0.812 | 0.772 | **0.820** |
| P@3 | 0.374 | 0.356 | **0.381** |
| F1@3 | 0.497 | 0.473 | **0.503** |
| R@5 | 0.883 | 0.845 | **0.889** |
| MRR | 0.769 | 0.733 | **0.790** |
| Page R@3 | **0.870** | 0.848 | 0.868 |
| Average words in top 3 | **216.6** | 289.0 | 274.5 |
| Search time per question | **0.6 ms** | 3.3 ms | 84.5 ms |

Search times come from one run on a laptop and exclude embedding the question, because question vectors were cached.

### By question type

| Group | Questions | Dense R@3 | Hybrid R@3 | Hybrid + rerank R@3 | Dense MRR | Hybrid MRR | Hybrid + rerank MRR |
|---|---|---|---|---|---|---|---|
| Text questions | 210 | 0.855 | **0.881** | 0.871 | 0.844 | 0.845 | **0.879** |
| Image questions, answer | 58 | **0.655** | 0.379 | 0.638 | **0.497** | 0.328 | 0.471 |
| Image questions, figure | 58 | **0.466** | 0.069 | **0.466** | **0.253** | 0.026 | **0.253** |

Image questions are scored as in [By question type](#by-question-type) above.

### By format

| Format | Questions | Dense R@3 | Hybrid R@3 | Hybrid + rerank R@3 | Dense MRR | Hybrid MRR | Hybrid + rerank MRR |
|---|---|---|---|---|---|---|---|
| Slides | 206 | **0.823** | 0.787 | 0.820 | 0.794 | 0.756 | **0.808** |
| Documents | 62 | 0.774 | 0.723 | **0.820** | 0.685 | 0.659 | **0.731** |

### Paired difference from dense

Each method minus dense on evidence R@3, question by question, with a bootstrap 95% interval. Bold gains have intervals that exclude 0. Dev has 102 questions (82 text, 20 image), and test has 166 (128 text, 38 image).

| Group | Hybrid, dev | Hybrid, test | Hybrid + rerank, dev | Hybrid + rerank, test |
|---|---|---|---|---|
| All questions | -0.010 (-0.069 to +0.047) | **-0.058 (-0.110 to -0.010)** | -0.001 (-0.053 to +0.053) | +0.014 (-0.017 to +0.047) |
| Text questions | **+0.049 (+0.002 to +0.102)** | +0.010 (-0.031 to +0.053) | +0.011 (-0.051 to +0.075) | +0.018 (-0.022 to +0.061) |
| Image questions, answer | **-0.250 (-0.425 to -0.075)** | **-0.289 (-0.447 to -0.158)** | -0.050 (-0.150 to 0.000) | 0.000 (0.000 to 0.000) |
| Image questions, figure | **-0.400 (-0.600 to -0.200)** | **-0.395 (-0.553 to -0.237)** | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |

### Search conclusion

- **Why dense wins:** it has the best dev R@3 and ties for the best dev MRR. It is also the simplest method and needs no extra index or model.
- **How close hybrid + rerank is:** on dev the two are level (-0.001). It scores higher on test (0.855 against 0.841), but that gain is within noise and test does not pick the winner. It adds a keyword index, a second model and about 84 ms per question.
- **Why plain hybrid falls behind:** it gains on text questions, but finds the figure on only 6.9% of image questions. BM25 cannot see image chunks, and RRF rewards chunks that appear in both lists, so figures rarely reach the top 3.
- **Reranking does not help figures:** the figure score stays at 0.466, because images keep their dense positions.

**Other methods tried:** BM25 alone and three other variants were tested and left out to keep this comparison to three distinct methods. BM25 alone was clearly behind dense (dev R@3 0.647). Two variants scored slightly higher than dense on dev, both within noise: images at dense positions with hybrid text (0.784) and reranked dense (0.779). The third, an image slot reserved in the top 3, found the figure more often (0.741) but lowered text R@3 to 0.785.

**Limits:** dev has only 20 image questions, so those intervals are wide.

## Reproduce

```bash
# Search methods
.venv/bin/python eval/run_search.py

# Gemma 2
.venv/bin/python eval/run_text_embedders.py gemma2
.venv/bin/python eval/run_image_embedders.py gemma2

# Jina CLIP v2 needs its own environment
.venv/bin/python -m venv .venv-jina
.venv-jina/bin/python -m pip install -r requirements-jina.txt
.venv-jina/bin/python eval/run_text_embedders.py jinaclip2
.venv-jina/bin/python eval/run_image_embedders.py jinaclip2

# Paired comparison, from the saved results
.venv/bin/python eval/run_text_embedders.py --compare
.venv/bin/python eval/run_image_embedders.py --compare
```

Results are in `eval/results/text_embedder_<model>.csv`, `eval/results/image_embedder_<model>.csv` and `eval/results/search.csv`, each with a matching `_questions.csv`.
