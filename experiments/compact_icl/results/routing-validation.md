# Adaptive routing validation

**Decision:** Keep the full tuned pipeline. Neither routing rule met the
predefined target of saving at least 10% of total recorded pipeline tokens
with no observed drop in execution accuracy. The planned fresh BIRD dev
evaluation was skipped.

| Approach | Correct / 100 | Pipeline tokens | Token savings | First answer used |
| --- | ---: | ---: | ---: | ---: |
| Full tuned pipeline | 77 | 5,282,977 | — | — |
| Always return the first answer | 62 | 3,898,060 | 26.2% | 100% |
| Single-table rule | 77 | 5,213,014 | 1.3% | 9% |
| One-join rule | 75 | 4,918,363 | 6.9% | 35% |

The 100-question validation split includes the initial 20-question pilot.
Each approach used the same first candidate and preprocessing. The routing
scores replay the two rules on the shared outputs. Token estimates remove
the generation, repair, and selection calls each rule would skip.

Compared with returning the first answer, the full pipeline rescued 21
wrong answers and replaced 6 correct answers with incorrect ones. The
one-join rule's paired accuracy difference versus the full pipeline was
−2 percentage points (95% database-bootstrap interval approximately −5.0
to 0.0).

Together reported $0.97 in usage for the shared experiment, including
the pilot. The total was $1.15 when conservatively reserving cost for
requests with unknown usage.
