# MiniLM chunking evaluation

**Setup:** `all-MiniLM-L6-v2`, dense search, top 3 chunks and no image chunks, on 210 text questions.

| Format | Dev | Test | Total |
|---|---|---|---|
| Slides | 58 | 103 | 161 |
| Documents | 24 | 25 | 49 |
| All text | 82 | 128 | 210 |

**Winner rule:** highest dev evidence recall@3, then MRR@3. Test scores are for reporting only.

**Reading the tables:** each cell is evidence recall@3 / precision@3 / F1@3 / MRR@3. Precision has a low ceiling: when one chunk holds the evidence, the most a question can score is 0.333.

**MRR source:** MRR@3 is the `ev_mrr` column family in the summary CSVs, averaged from `ev3_rr` in the `_questions.csv` files.

## Baseline

Slides use fixed 100/15 and documents use fixed 200/30.

| Questions | Dev | Test |
|---|---|---|
| Slides | 0.756 / 0.362 / 0.472 / 0.718 | 0.690 / 0.333 / 0.431 / 0.649 |
| Documents | 0.826 / 0.403 / 0.522 / 0.868 | 0.940 / 0.400 / 0.548 / 0.893 |
| All text | 0.776 / 0.374 / 0.486 / 0.762 | 0.739 / 0.346 / 0.454 / 0.697 |

## Chunking permutations

Slides and documents share one index, so all 6 combinations were run. Fixed and recursive use 100/15 on slides and 200/30 on documents. A format's score can change with the other format's method, because its questions compete with different chunks.

**Overall (all text questions):**

| Slides | Documents | Dev | Test |
|---|---|---|---|
| Fixed | Fixed (baseline) | 0.776 / 0.374 / 0.486 / 0.762 | 0.739 / 0.346 / 0.454 / 0.697 |
| Fixed | Recursive | 0.783 / 0.382 / 0.495 / 0.754 | 0.739 / 0.352 / 0.458 / 0.689 |
| Recursive | Fixed | 0.819 / 0.390 / 0.510 / 0.817 | 0.791 / 0.375 / 0.492 / 0.745 |
| Recursive | Recursive | 0.819 / 0.398 / 0.517 / 0.807 | 0.791 / 0.380 / 0.496 / 0.742 |
| **One per slide** | **Fixed (chosen)** | **0.831 / 0.407 / 0.526 / 0.854** | 0.818 / 0.370 / 0.495 / 0.803 |
| One per slide | Recursive | 0.822 / 0.407 / 0.525 / 0.831 | 0.814 / 0.372 / 0.495 / 0.801 |

**By format:**

| Slides | Documents | Slides dev | Slides test | Documents dev | Documents test |
|---|---|---|---|---|---|
| Fixed | Fixed (baseline) | 0.756 / 0.362 / 0.472 / 0.718 | 0.690 / 0.333 / 0.431 / 0.649 | 0.826 / 0.403 / 0.522 / 0.868 | 0.940 / 0.400 / 0.548 / 0.893 |
| Fixed | Recursive | 0.756 / 0.362 / 0.472 / 0.716 | 0.680 / 0.330 / 0.427 / 0.644 | 0.847 / 0.431 / 0.551 / 0.847 | 0.980 / 0.440 / 0.587 / 0.873 |
| Recursive | Fixed | 0.825 / 0.391 / 0.511 / 0.802 | 0.755 / 0.369 / 0.479 / 0.714 | 0.806 / 0.389 / 0.506 / 0.854 | 0.940 / 0.400 / 0.548 / 0.873 |
| Recursive | Recursive | 0.825 / 0.391 / 0.511 / 0.799 | 0.755 / 0.369 / 0.479 / 0.709 | 0.806 / 0.417 / 0.531 / 0.826 | 0.940 / 0.427 / 0.567 / 0.880 |
| **One per slide** | **Fixed (chosen)** | **0.842 / 0.414 / 0.534 / 0.856** | 0.788 / 0.362 / 0.482 / 0.786 | **0.806 / 0.389 / 0.506 / 0.847** | 0.940 / 0.400 / 0.548 / 0.873 |
| One per slide | Recursive | 0.838 / 0.408 / 0.529 / 0.848 | 0.788 / 0.362 / 0.482 / 0.782 | 0.785 / 0.403 / 0.514 / 0.792 | 0.920 / 0.413 / 0.551 / 0.880 |

**Document window sizes, with slides at one per slide:**

| Method and size | Words in top 3 | Chunks truncated | Dev | Test |
|---|---|---|---|---|
| Fixed 100/15 | 289 | 1.0% | 0.611 / 0.319 / 0.403 / 0.646 | 0.820 / 0.320 / 0.452 / 0.780 |
| Recursive 100/15 | 256 | 0.6% | 0.646 / 0.306 / 0.404 / 0.618 | 0.900 / 0.440 / 0.572 / 0.773 |
| Fixed 150/22 | 419 | 3.5% | 0.799 / 0.431 / 0.544 / 0.715 | 0.920 / 0.387 / 0.532 / 0.833 |
| Recursive 150/22 | 384 | 2.2% | 0.792 / 0.361 / 0.483 / 0.681 | 0.920 / 0.440 / 0.576 / 0.880 |
| **Fixed 200/30 (chosen)** | 529 | 38.6% | **0.806 / 0.389 / 0.506 / 0.847** | 0.940 / 0.400 / 0.548 / 0.873 |
| Recursive 200/30 | 480 | 19.8% | 0.785 / 0.403 / 0.514 / 0.792 | 0.920 / 0.413 / 0.551 / 0.880 |

## Best outcome

**One chunk per slide for slides, and fixed 200/30 for documents.**

This combination has the highest overall dev recall@3 (0.831) and MRR@3 (0.854). Against the baseline, it adds 0.055 to dev recall@3, 0.092 to dev MRR@3 and 0.040 to dev F1@3. Test recall@3 also rises, from 0.739 to 0.818.

**Slides: one chunk per slide (strong)**

- It scores highest on dev recall@3, precision@3, F1@3 and MRR@3, and it retrieves the least text: 202 words in the top 3, against 306 for fixed and 260 for recursive.
- A slide is usually one idea, so this avoids mixing topics across slides, which fixed and recursive chunks do.
- Its clearest gain over fixed is MRR@3: +0.138 on dev (95% interval +0.029 to +0.250). Its recall@3 lead over recursive is small and within noise (+0.017, interval -0.040 to +0.080).
- It stays the best slide method whichever document method is used, and it has the highest slide recall@3 and MRR@3 on test.

**Documents: fixed 200/30 (moderate)**

- It has the highest dev recall@3 and MRR@3 of the six document settings, so it wins the rule when slides use one chunk per slide.
- Fixed and recursive are effectively tied. At 200 words, recursive's dev recall@3 gain is -0.021 (interval -0.188 to +0.125), and recursive wins instead when slides use fixed chunks. Fixed was chosen as the simpler method.
- 100-word chunks are clearly too small: dev recall@3 is 0.194 lower than fixed 200/30 (interval -0.375 to -0.021).

**Limits of this choice**

- **Fixed 150/22 is the close alternative for documents.** Its dev recall@3 is about level, and it has higher dev F1@3 (0.544 vs 0.506) and retrieves 21% less text. Fixed 200/30 wins because of its higher MRR@3 (+0.132, interval +0.014 to +0.264), and its F1@3 lead on test is small (0.548 vs 0.532).
- **MiniLM truncates long chunks.** It reads up to 256 tokens, which cuts off 38.6% of fixed 200/30 chunks. Evidence is still scored against the full chunk text, so this could affect the document result.
- **Larger chunks can inflate recall.** They have more chances to contain an evidence quote.
- **Document results rest on few questions.** There are 24 on dev, and one single-part question moves dev recall@3 by about 0.04.
- **Document test scores are not fully untouched.** Eleven document questions moved from test to dev after aggregate test scores from a first run had been seen. All 11 eligible questions moved, with none chosen by score.
