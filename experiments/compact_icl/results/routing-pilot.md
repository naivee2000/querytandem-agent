# Adaptive ICL pilot results

| Approach | Correct | Pipeline tokens | Token savings | First answer used | Estimated API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 17/20 | 1,038,087 | 0.0% | 0.0% | $0.24 |
| first | 13/20 | 763,794 | 26.4% | 100.0% | $0.14 |
| single_table | 17/20 | 1,022,587 | 1.5% | 10.0% | $0.23 |
| one_join | 17/20 | 957,334 | 7.8% | 40.0% | $0.22 |

Recommendation for this phase: **full**.
Together reported $0.19 in usage for the shared experiment. Including reservations for unknown usage, the total is $0.24.
The full pipeline rescued 5 first answers and replaced 1 correct ones with incorrect ones.
The JSON report includes paired database bootstrap intervals and the question IDs for disagreements.
