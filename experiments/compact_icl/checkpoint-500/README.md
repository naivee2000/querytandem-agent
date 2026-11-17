# Compact SQL generator adapter

This is the LoRA checkpoint selected at step 500. It adapts the query and
value projections in the last eight layers of
[mlx-community/Qwen3-4B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit/tree/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b),
a 4-bit conversion of [Qwen's instruct model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
The [adapter license](LICENSE) is Apache 2.0. The base model is downloaded
separately.

On Apple Silicon, start the MLX server with the downloaded model snapshot:

```sh
mlx_lm.server --model "$MODEL" \
  --adapter-path experiments/compact_icl/checkpoint-500 \
  --host 127.0.0.1 --port 8081
```

Use DeepEye's compact prompt with zero examples. The generator returns SQL
inside `<result>…</result>` tags. See the [experiment instructions](../README.md)
for full pipeline settings and BIRD setup.

The training runs used rank 8, a learning rate of 1e-5, a sequence limit of
2,048 tokens, and loss on answer tokens only. Training examples come from
[filtered BIRD](https://huggingface.co/datasets/birdsql/bird23-train-filtered).
The SHA-256 of `adapters.safetensors` is
`802df58831449af6ad015f5f89d1c2eb6d9a8861fb5ba604cc5876a42bc758b3`.
