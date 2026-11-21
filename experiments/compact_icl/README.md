# Compact ICL experiment

This extension is based on HKUSTDial/DeepEye-SQL commit
`252ab45a7adf9f88fb93d1ff27ca019d20f0449a`. See the [upstream README](../../UPSTREAM.md)
and MIT license, the [filtered BIRD training data](https://huggingface.co/datasets/birdsql/bird23-train-filtered),
and the [quantized Qwen checkpoint](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit).

Run commands from the project root. Use the locked Python 3.12 environment for
DeepEye and install `mlx-lm[train]` in a separate environment on Apple Silicon.
For inference with the released weights, start with the [checkpoint 500 model card](checkpoint-500/README.md).

1. Download BIRD's official `train.zip` and `dev.zip` and the filtered JSONL to
   `data/bird/downloads/` as `train.zip`, `dev.zip`, and `train.jsonl`.
2. Run `.venv/bin/python -m experiments.compact_icl.unpack`.
3. Set `MODEL` to the downloaded MLX snapshot directory. Run
   `.venv/bin/python -m experiments.compact_icl.prepare --model "$MODEL" --output workspace/compact-icl/bird`.
   The manifest records original row indices, database IDs, token counts, and rejections.
   Training schemas use the upstream formatter with schema statistics and example values
   disabled. This is a training approximation to runtime linked schema profiles.
4. Try the 20-step run before full training:
   `workspace/mlx-env/bin/mlx_lm.lora -c experiments/compact_icl/lora.yaml --model "$MODEL" --data workspace/compact-icl/bird --iters 20 --steps-per-eval 20 --save-every 20 --adapter-path workspace/compact-icl/pilot`.
   Use `check_adapter.py` with the MLX Python environment, the adapter path, and
   the validation JSONL to check generation. The full training YAML uses 1,000 steps.
5. After validation selects a checkpoint and example count, run MLX servers
   on loopback ports 8080 (base) and 8081 (with `--adapter-path`). Set
   `TOGETHER_API_KEY` in your shell, then run
   `.venv/bin/python -m experiments.compact_icl.configure --examples 0`
   (use 1 if validation selects one example).
6. The generated configs are inputs to DeepEye's BIRD runner. Keep each
   variant's output in its own run directory. The template fixes data,
   budgets, endpoints, and decoding settings for the comparison. After the
   five-question pilot and checkpoint selection, run
   `bash experiments/compact_icl/run_benchmark.sh`. It projects the
   Together cost from the pilot, enforces a $10 API cap, reuses the baseline
   schema and other candidates, and writes `workspace/compact-icl/results.json`.
   The ledger retains a conservative charge for calls with unknown usage.

The core generator defaults to the upstream prompt, example retrieval, and
stage model. The optional settings under `[sql_generation]` are
`icl_llm_profile`, `icl_prompt_mode`, and `icl_example_count`.

## Adaptive ICL

The adaptive experiment starts with the compact tuned model's first SQL candidate.
Using `[sql_generation].icl_fast_path = "single_table"` or `"one_join"` enables
the corresponding gate for a compact, zero-example, one-candidate configuration.
Use a separate run name when changing the policy so old snapshots cannot be
reused. The default is `"off"`.
The `"first"` setting runs the first-answer control, including counting a
missing answer as an error. Use the final report's recommended setting.

From the repository root, give each run the current Together prepaid credit in
USD as its second argument. The script checks its forecast and reserves funds
for each request. Run the phases serially:

```sh
bash experiments/compact_icl/run_routing.sh pilot <available-credit>
bash experiments/compact_icl/run_routing.sh validation <available-credit>
bash experiments/compact_icl/run_routing.sh fresh <available-credit>
```

The pilot caps spending at $2. Validation combines the pilot's 20 questions
with the remaining 80 held-out training questions. The fresh phase requires a
frozen policy and evaluates 200 new dev questions. The full run supplies the
same first candidate and full-pipeline result for all comparisons. The report
replays the router against those outputs and removes only the requests it
would skip. Shared preprocessing and its retries count in each variant. The
listed actual expense includes a conservative reservation when usage is unknown.
Reports, manifests and request ledgers are in `workspace/adaptive-icl`.
Use `run_routing.sh all <available-credit>` to run the gated sequence in one
process. It budgets each phase from the verified opening credit minus recorded
requests and outstanding reservations. Use a dedicated credit allowance for
the sequence when other jobs are spending from the same account.

```sh
.venv/bin/python -m unittest experiments.compact_icl.test_routing
```

## Trained SQL selector experiment

**Completed result:** the ranker matched the current selector at 77/100 and
did not qualify for fresh evaluation. Keep the default selector. See the
[full report](results/selector-validation.md), [fitted artifact](results/selector-ranker.json),
and [derived training features](results/selector-training-features.json).
The artifact is marked `qualified: false`; enabling it manually is an
experimental opt-in and does not represent a demonstrated improvement.

To check the published fit without API calls or BIRD downloads:

```sh
.venv/bin/python -m experiments.compact_icl.rerank_report --verify
```

Recreating candidate generation requires the original data, local model,
selected checkpoint, retrieval index, and cached adaptive validation runs
described above. Use Python 3.12 and the existing lockfile's package index:
`uv sync --locked --default-index https://pypi.tuna.tsinghua.edu.cn/simple`.
The recorded CPU fit used scikit-learn 1.7.2. Provider-generated candidates can vary between runs,
so an independent run need not reproduce identical accuracy.

Run `.venv/bin/python -m experiments.compact_icl.rerank_experiment --credit AMOUNT`
from the repository root with the verified opening Together credit in USD.
The experiment reuses the tuned compact model and the existing BIRD retrieval
index. It starts with 40 questions and a $1 pilot cap. The overall cap is the
smaller of the supplied credit and $8. Training requires at least 60 questions
with both correct and incorrect answer groups across the 400 training questions.

Logs, reports and the JSON ranker artifact are in `workspace/sql-reranker`.
The validation gate uses the 100 saved adaptive-routing validation questions
and requires at least 79 correct answers and a higher score than majority vote.
Only a qualifying artifact advances to the fresh 200-question evaluation.

To use the optional SQLite selector set `[sql_selection].reranker_path` to the
JSON model file and use a new run name to avoid resuming an existing selection
snapshot. The ordinary selector runs first. The ranker can replace its answer
using the already repaired candidates, with no additional model request.
Run the targeted tests with
`.venv/bin/python -m unittest experiments.compact_icl.test_reranker`.

### Revised pilot: offline labels and local candidate augmentation

The first pilot spent $0.53058661 on 40 training questions. Its original shared
three-second candidate-checking limit yielded only five informative questions,
and the full-run cost forecast ($9.44 including headroom) exceeded the $8 cap.
Those pilot outputs and usage records are preserved.

The revised experiment gives each unique candidate and gold query an independent
30-second read-only execution limit during **offline training labeling**. It
retains other completed candidates when one fails and records SQL errors,
timeouts, row-limit failures, and gold-query failures explicitly. Live selection
still uses a shared three-second limit and its existing fallback behavior.

```bash
.venv/bin/python -u -m experiments.compact_icl.refine_reranker --credit 8
```

This command resumes the first pilot created by `rerank_experiment` above;
it requires that pilot's completed candidate snapshot and usage ledger.
It does not launch a new experiment when a completion report already exists.

`--credit` is the opening allowance for the whole selector experiment, including
the first pilot's spending, and is capped at $8. First, three additional candidates
per training question are generated locally from the cached inference schema,
question, and evidence, using the selected adapter, temperature 0.7, top-p 0.9,
512 output tokens, and reproducible seeds. The generation process receives no
gold SQL or labels. Actual tokenizer counts, output failures, runtime, and peak
MLX memory are logged. Raw local candidates are used as training augmentation;
this differs from the repaired candidate distribution used at inference and
must earn its benefit on unchanged validation candidates.

Only if augmentation increases the number of informative questions and the
forecast fits does collection advance to new, disjoint 40-question shards from
the original seed-42 training split. Each paid shard has a cumulative $1 cap,
including retries. Cost forecasts add 20% headroom and discount observed label
yield by 20%; at least $3.20 is reserved for the fresh benchmark. Collection
stops when 60 distinct informative training questions are available, or when
the data or budget gate fails. The original validation thresholds and fresh
200-question evaluation are unchanged. The local augmentation is training only
and adds no inference model calls.

Progress and outputs:

- `workspace/sql-reranker/refinement-status.json`: current phase and total usage.
- `workspace/sql-reranker/refinement.log`: overall progress and failures.
- `workspace/sql-reranker/augmentation/<phase>/progress.json`: local generation count.
- `workspace/sql-reranker/offline-labels/`: cached labels and execution diagnostics.
- `workspace/sql-reranker/refinement-forecast.json`: next-batch forecast and reserve.
- `workspace/sql-reranker/refinement-complete.json`: final gate decision or fresh results.

The same command resumes completed labels and individual local generations.
Frozen manifests reject changed prompts, model settings, or source code;
incomplete pipeline checkpoints are resumed rather than treated as finished.

Export completed local reports and derived training features without exposing
prompts, SQL, credentials, or databases:

```sh
.venv/bin/python -m experiments.compact_icl.rerank_report
.venv/bin/python -m unittest experiments.compact_icl.test_reranker experiments.compact_icl.test_local_index experiments.compact_icl.test_routing tests.test_schema_profile
```

Index construction now streams embedding batches to disk and uses memory-mapped
arrays for CPU retrieval. `runner/create_vector_db_parallel.py --parallelism 1`
limits concurrent database builds on a laptop. These changes preserve the
index format and retrieval calculation while avoiding whole-column embedding
accumulation in memory.
