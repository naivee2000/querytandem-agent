# QueryTandem Agent

Compact text-to-SQL generation on Apple Silicon.

I adapted one generator to run a locally fine-tuned 4B model with zero prompt
examples, then measured what that saved across the full text-to-SQL pipeline.

On 200 held-out BIRD dev questions, the tuned zero-example configuration
scored **70.0% execution accuracy vs. 67.0%** for the seven-example baseline.
It used **69.5% fewer local generator tokens** and **1.9% fewer total recorded
pipeline tokens**. The paired 95% database-bootstrap interval for the accuracy
difference was **−0.5 to +7.1 percentage points**.

| Generator | Prompt examples | Correct / 200 | Total recorded pipeline tokens |
| --- | ---: | ---: | ---: |
| Original Qwen3-4B | 7 | 134 (67.0%) | 11,436,511 |
| Original Qwen3-4B | 0 | 135 (67.5%) | 10,908,668 |
| LoRA-tuned Qwen3-4B | 7 | 137 (68.5%) | 11,507,773 |
| LoRA-tuned Qwen3-4B | 0 | **140 (70.0%)** | 11,221,739 |

With the zero-example prompt held fixed, fine-tuning got five more answers
right and used 2.9% more pipeline tokens. Prompt shortening alone used the
fewest tokens. The 200-question comparison used the same model family,
candidate budgets, schemas, and shared generator outputs across variants.
These results apply to this DeepEye configuration and evaluation sample.

## What I built

- Integrated a local MLX generator into DeepEye's existing pipeline and added
  optional settings for the generator model, prompt format, and example count.
- Trained a rank-8 LoRA adapter on 500 BIRD training examples on a MacBook Pro.
  I selected checkpoint 500 and the zero-example prompt on a separate
  100-question validation split.
- Built a reproducible four-way comparison with read-only SQLite execution
  scoring, paired uncertainty estimates, and a per-request API budget ledger.

## Adaptive routing follow-up

The optional router tries one local SQL answer and uses the full pipeline
when the answer fails a conservative screen. In the **20-question pilot**,
the one-join rule matched the full pipeline at **17/20** while saving
**7.8% of tokens**. That was below the 10% target. Always taking the first
answer saved 26.4% but scored 13/20. Validation is in progress. Routing is
off by default.

## Run or review

- [Experiment setup and reproduction](experiments/compact_icl/README.md)
- [Four-variant results](experiments/compact_icl/results/compact-icl.json)
  and [routing pilot](experiments/compact_icl/results/routing-pilot.md)
- [Trained MLX adapter and model card](experiments/compact_icl/checkpoint-500/README.md)
- [Original DeepEye-SQL documentation and citation](UPSTREAM.md)

Based on DeepEye-SQL commit [`252ab45`](https://github.com/HKUSTDial/DeepEye-SQL/commit/252ab45a7adf9f88fb93d1ff27ca019d20f0449a).
The upstream research pipeline is MIT licensed. The adapter uses the
[Apache 2.0 Qwen model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
