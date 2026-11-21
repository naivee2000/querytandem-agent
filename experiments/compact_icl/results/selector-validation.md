# Trained SQL selector: validation result

**Decision: retain the existing selector.** The trained ranker scored 77/100,
matching the existing selector and majority vote. It did not meet the preset
requirement of at least 79/100 and a score above majority vote, so the fresh
200-question benchmark was not run. No additional accuracy improvement is claimed.

This follow-up leaves the earlier four-way generator benchmark unchanged.
The optional ranker is disabled by default; its weights are published for
inspection and reproduction, not as a recommended replacement.

## Paired validation

All selectors used the same 100 cached validation questions and repaired SQL
candidates. These are validation questions from held-out training databases,
not the 200 BIRD dev questions in the original public benchmark.

| Selector | Correct / 100 | SQL overrides | Wrong answers fixed | Correct answers overturned |
| --- | ---: | ---: | ---: | ---: |
| Existing selector | 77 | 0 | — | — |
| Majority vote | 77 | 1 | 0 | 0 |
| Trained ranker, threshold 0.60 | 77 | 1 | 0 | 0 |
| Trained ranker, threshold 0.75 | 77 | 0 | 0 | 0 |
| Trained ranker, threshold 0.90 | 77 | 0 | 0 | 0 |

The selection rule first maximized correct answers, then minimized overrides;
the remaining tie selected 0.90. The one override at 0.60 did not change that
question's correctness. Each variant therefore has a paired difference of
zero and a database-bootstrap interval of [0, 0] on these saved outcomes.
This interval does not establish equivalence on unseen questions.

The ranker added **zero model calls and tokens** during validation replay.
Its recorded added local latency was approximately **3.1 ms median** and
**2.20 seconds at the 95th percentile**, mostly from executing candidates.
The existing selector still runs first; this extension does not remove its calls.

## Training and labeling

The initial 40-question pilot spent $0.53058661. Its shared three-second
candidate-execution limit yielded five informative questions. Giving each
unique candidate its own 30-second offline limit recovered eight. Adding
three locally generated candidates per question increased that pilot to 15.

| Training batch | Questions | Informative before augmentation | Informative after augmentation |
| --- | ---: | ---: | ---: |
| Original pilot | 40 | 8 | 15 |
| Additional batch 1 | 40 | 2 | 8 |
| Additional batch 2 | 40 | 3 | 7 |
| Additional batch 3 | 40 | 8 | 16 |
| Additional batch 4 | 40 | 11 | 16 |
| Total | **200** | **32** | **62** |

An informative question has at least one correct and one incorrect executable
answer group. Collection stopped after reaching the minimum of 60 such
questions. Regularized pairwise logistic regression (`C=1`, no intercept)
was fitted on CPU with equal total weight per informative question. Its 13
features describe agreement, result dimensions, empty results, SQL length,
word overlap, and simple question/SQL cues. Gold SQL supplies labels only.

The selected local Qwen adapter generated 600 extra candidate responses:
575 parsed successfully and 25 failed the required output format. Across
the augmented training pools, 1,025 unique-per-question candidate executions
succeeded, 78 failed SQL execution, and nine exceeded the 100,000-row limit.
Three gold queries also exceeded that limit; those questions remain in the
200-question collection count but supplied no fitting pairs.

Training questions were sampled with seed 42, excluded from the retrieval
index, and checked for separation from validation and evaluation questions.
Training and validation databases are disjoint. Generation used cached
inference-selected schemas, questions, and evidence; the local generation
process received no gold SQL. Runtime selection retains its shared three-second
deadline and fallback. Offline labeling failures are recorded separately.

## Spending and artifacts

Together reported **$2.37264653** for the whole selector experiment, including
the original pilot. A further **$0.044821** remains reserved for one request
without reported usage, giving **$2.41746753** of budget accounted for.
There were 3,540 Together requests and 800 local generations, including
retries in the request ledger. Recorded tokens totaled 13,564,358; tokens
for the unreported request are unknown. These are preparation/training costs,
not an estimate of per-question inference savings.

The cached validation pipeline had historical usage of its own, reported
separately in the JSON. It was replayed locally and was not paid for again.
The original $8 experiment ceiling was respected; each new paid batch stayed
under its $1 cap. The fresh benchmark's reserved funds were not spent.

- [Machine-readable results and usage](selector-validation.json)
- [Fitted ranker, marked `qualified: false`](selector-ranker.json)
- [Derived training features and labels](selector-training-features.json)
- [Experiment setup and commands](../README.md#trained-sql-selector-experiment)

Refit the published feature records and verify all coefficients without
downloading databases, loading MLX, or making API calls:

```sh
.venv/bin/python -m experiments.compact_icl.rerank_report --verify
```

The derived records include dataset row IDs and feature vectors, not question
text, SQL, database contents, or credentials. Original prompts, execution
diagnostics, and raw request logs remain in the ignored local workspace.

## Interpretation and limitations

Local augmentation increased the number of training contrasts, but the
resulting ranker did not recover any validation misses. The small training
set, simple features, and difference between raw augmented candidates and
repaired inference candidates are limitations, not established explanations
for the result. The shared validation set was also used in earlier project
experiments; it is not an independent final test.

The useful contribution from this follow-up is the optional selector,
reproducible training and budget gates, explicit failure accounting, and
an evidence-based decision to retain the working configuration.
