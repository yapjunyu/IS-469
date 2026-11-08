# Agent evaluation results

Written by `eval/summarise_agent.py`. Bold gains have a 95% interval that excludes 0.

## Agent scorecard: search-again on vs off

### All requests

| Metric | search-again off | search-again on | Gain, on minus off (95% CI) | Items |
|---|---|---|---|---|
| Task success | 0.933 | 0.933 | +0.000 (-0.067 to +0.067) | 60 |
| Tool-call accuracy | 0.900 | 0.867 | -0.033 (-0.083 to +0.000) | 60 |
| Steps per turn | 2.450 | 2.367 | -0.083 (-0.217 to +0.050) | 60 |
| Searches per turn | 1.467 | 1.750 | **+0.283 (+0.017 to +0.650)** | 60 |
| Tokens per turn | 12308.250 | 12981.017 | +672.767 (-239.567 to +1617.483) | 60 |
| Latency (s) | 10.326 | 13.197 | **+2.870 (+1.689 to +4.213)** | 60 |

| Scorer tokens | In | Out |
|---|---|---|
| search-again off | 1511 | 867 |
| search-again on | 1483 | 825 |

### By request type

| Type | Arm | Items | Task success | Tool-call accuracy | Steps | Tokens | Latency (s) |
|---|---|---|---|---|---|---|---|
| content | off | 30 | 0.90 | 0.80 | 2.5 | 14379 | 11.8 |
| content | on | 30 | 0.90 | 0.73 | 2.4 | 15396 | 16.9 |
| quiz_me | off | 12 | 1.00 | 1.00 | 2.0 | 9130 | 8.5 |
| quiz_me | on | 12 | 1.00 | 1.00 | 2.2 | 10771 | 10.7 |
| quiz_answer | off | 10 | 0.90 | 1.00 | 3.0 | 12627 | 9.4 |
| quiz_answer | on | 10 | 0.90 | 1.00 | 2.8 | 11888 | 9.2 |
| remember | off | 4 | 1.00 | 1.00 | 3.0 | 11917 | 12.2 |
| remember | on | 4 | 1.00 | 1.00 | 3.0 | 12526 | 12.1 |
| off_topic | off | 4 | 1.00 | 1.00 | 1.2 | 5905 | 5.5 |
| off_topic | on | 4 | 1.00 | 1.00 | 1.0 | 4684 | 4.2 |

### Failed requests, by cause

| Request | Arm | Cause | Tools called | Answer starts |
|---|---|---|---|---|
| qa-03 | off | unfaithful answer | grade_answer, update_mastery | Not quite — reread the Shielded VM section on page 9; your answer only covers th |
| pm-18 | off | unfaithful answer | search_notes, search_notes | ## Why independent and negotiable  **Independent** and **Negotiable** are two of |
| cl-91 | off | unfaithful answer | search_notes, search_notes, search_notes, search_notes | Here's what your notes contain on these services:  **Lambda → GCP:** The notes g |
| cl-88 | off | unfaithful answer | search_notes, search_notes | Your notes don't answer this. Here's what they actually contain:  - A slide titl |
| ai-52 | on | unfaithful answer | search_notes, search_notes | ## Prefill vs. decode  **Prefill** processes the whole prompt in one parallel pa |
| qa-03 | on | unfaithful answer | grade_answer, update_mastery | Not quite — reread the Shielded VM section on page 9; your answer only covers th |
| pm-23 | on | unfaithful answer | search_notes, search_notes | Here's what your notes actually contain on this.  **Story mapping** — your notes |
| cl-91 | on | unfaithful answer | search_notes, search_notes, search_notes, search_notes, search_notes | Your notes give only part of this mapping.  **Lambda:** The notes place AWS Lamb |

## Answer quality (RAGAS): search-again on vs off

Scored by the Scorer, `claude-haiku-5-5`. Context metrics cover text questions only.

| Scorer tokens | Total |
|---|---|
| search-again off | 1081492 |
| search-again on | 1798613 |

### llm_eval gold questions

| Metric | search-again off | search-again on | Gain, on minus off (95% CI) | Items |
|---|---|---|---|---|
| Factual correctness | 0.495 | 0.495 | +0.000 (-0.025 to +0.024) | 75 |
| Faithfulness | 0.933 | 0.933 | +0.000 (-0.067 to +0.067) | 75 |
| Context precision | 0.914 | 0.900 | -0.014 (-0.039 to +0.007) | 60 |
| Context recall | 0.912 | 0.913 | +0.000 (-0.025 to +0.025) | 60 |
| Searches per answer | 1.960 | 2.507 | **+0.547 (+0.227 to +0.947)** | 75 |
| Tokens per answer | 13546.853 | 15707.827 | **+2160.973 (+783.120 to +3606.867)** | 75 |
| Latency (s) | 11.486 | 16.151 | **+4.665 (+3.458 to +5.967)** | 75 |

## Prompt injection: guard on vs off

### Injection rate (lower is better)

| Metric | guard off | guard on | Gain, on minus off (95% CI) | Items |
|---|---|---|---|---|
| Injection followed | 0.300 | 0.000 | **-0.300 (-0.500 to -0.100)** | 20 |

| Attack | Guard off | Guard on |
|---|---|---|
| answer | 3 / 12 | 0 / 12 |
| quiz | 1 / 3 | 0 / 3 |
| tool | 2 / 5 | 0 / 5 |

## Memory poisoning: memory-write check on vs off

### Poisoning (lower is better, except wrong belief recorded)

| Metric | check off | check on | Gain, on minus off (95% CI) | Items |
|---|---|---|---|---|
| False claim stored as a fact | 0.000 | 0.000 | +0.000 (+0.000 to +0.000) | 20 |
| Repeated as true next session | 0.000 | 0.000 | +0.000 (+0.000 to +0.000) | 20 |
| Poisoned (either) | 0.000 | 0.000 | +0.000 (+0.000 to +0.000) | 20 |
| Recorded as a wrong belief | 0.750 | 0.650 | -0.100 (-0.300 to +0.100) | 20 |

| Scorer tokens | In | Out |
|---|---|---|
| check off | 6268 | 1562 |
| check on | 7282 | 1691 |

Facts never reach the tutor's prompt (only weak spots and wrong beliefs do), so "repeated" cannot come from a stored fact: only from the wrong-belief line or the tutor itself.

