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
