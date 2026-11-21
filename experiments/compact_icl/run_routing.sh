#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

phase="${1:?Specify pilot, validation, or fresh}"
credit="${2:?Supply the current available Together credit in USD}"
out="$PWD/workspace/adaptive-icl"
python="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD" HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1

if [[ "$phase" == "all" ]]; then
  if [[ -n "${3:-}" ]]; then
    while [[ ! -s "$out/pilot.md" ]] && kill -0 "$3" 2>/dev/null; do sleep 15; done
    test -s "$out/pilot.md"
  else
    bash "$0" pilot "$credit"
  fi
  remaining=$("$python" -m experiments.compact_icl.routing remaining-credit --verified-credit "$credit")
  bash "$0" validation "$remaining"
  if [[ -s "$out/frozen-policy.json" ]]; then
    remaining=$("$python" -m experiments.compact_icl.routing remaining-credit --verified-credit "$credit")
    bash "$0" fresh "$remaining"
  else
    printf 'Validation did not qualify a routing policy. Fresh evaluation skipped.\n'
  fi
  exit 0
fi

experiment=adaptive-icl
if [[ "$phase" == pipeline ]]; then
  experiment="${3:?Specify the experiment directory name}"
  phase="${4:?Specify the dataset phase}"
  out="$PWD/workspace/$experiment"
  export DEEPEYE_API_CAP_USD="$credit"
else
  "$python" -m experiments.compact_icl.routing prepare "$phase"
fi
run="$PWD/workspace/runs/$experiment-$phase"
if [[ "$experiment" == adaptive-icl ]]; then
  export CONFIG_PATH="$PWD/config/local/compact-icl/adaptive-$phase.toml"
else
  export CONFIG_PATH="$PWD/config/local/compact-icl/$experiment-$phase.toml"
fi
export DEEPEYE_USAGE_LEDGER="$out/$phase-usage.jsonl" DEEPEYE_COUNT_NO_CANDIDATE=1
stage_complete() {
  "$python" -c 'import sys; from experiments.compact_icl.rerank_experiment import snapshot_complete; sys.exit(0 if snapshot_complete(sys.argv[1], sys.argv[2]) else 1)' "$run/$1.snapshot" "$1"
}
if stage_complete sql_selection; then
  if [[ "$experiment" == adaptive-icl ]]; then
    "$python" -m experiments.compact_icl.routing report "$phase"
  fi
  exit 0
fi
export DEEPEYE_API_CAP_USD
if [[ "$experiment" == adaptive-icl ]]; then
  DEEPEYE_API_CAP_USD=$("$python" -m experiments.compact_icl.routing budget "$phase" --verified-credit "$credit")
fi
test ! -e "$DEEPEYE_USAGE_LEDGER.cap-reached"
printf 'Phase %s has a cumulative request cap of $%s.\n' "$phase" "$DEEPEYE_API_CAP_USD"
curl --silent --show-error --max-time 15 --output /dev/null https://api.together.xyz/v1/models

run_stage() {
  export DEEPEYE_STAGE="$1"
  stage="$1"
  shift
  "$python" "runner/$stage.py" "$@"
  test ! -e "$DEEPEYE_USAGE_LEDGER.cap-reached"
}

if [[ ! -s "$run/dataset.snapshot" ]]; then
  run_stage preprocess_dataset
fi
if [[ ! -f "$out/$phase.vector-ready" ]]; then
  if [[ "$experiment" == sql-reranker ]]; then
    run_stage create_vector_db_parallel --parallelism 1
  else
    run_stage create_vector_db_parallel
  fi
  touch "$out/$phase.vector-ready"
fi
if ! stage_complete value_retrieval; then
  run_stage run_value_retrieval
fi
if ! stage_complete schema_linking; then
  export DEEPEYE_STAGE=run_few_shot_preparation
  "$python" runner/run_few_shot_preparation.py --index_path "$PWD/workspace/runs/_shared/bird/train_full_index"
  test ! -e "$DEEPEYE_USAGE_LEDGER.cap-reached"
  run_stage run_schema_linking
fi

if ! stage_complete sql_generation; then
  model=$(find .cache/huggingface/hub -type d -path '*/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*' -print -quit)
  adapter=$("$python" -c 'import json; print(json.load(open("workspace/compact-icl/selection.json"))["selected"]["adapter"])')
  test -d "$model"
  test -f "$adapter/adapters.safetensors"
  if curl -fsS http://127.0.0.1:8081/v1/models >/dev/null 2>&1; then
    printf 'Port 8081 is in use. Stop that server to ensure the selected adapter is loaded.\n' >&2
    exit 1
  fi
  workspace/mlx-env/bin/mlx_lm.server --model "$model" --adapter-path "$adapter" \
    --host 127.0.0.1 --port 8081 --prompt-cache-size 2 --prompt-cache-bytes 536870912 \
    --prefill-step-size 512 --decode-concurrency 1 --prompt-concurrency 1 > "$out/$phase-mlx.log" 2>&1 &
  server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
  for ((attempt=0; attempt<60; attempt++)); do
    kill -0 "$server_pid"
    if curl -fsS http://127.0.0.1:8081/v1/models >/dev/null 2>&1; then break; fi
    sleep 2
  done
  curl -fsS http://127.0.0.1:8081/v1/models >/dev/null
  run_stage run_sql_generation
fi
for stage in run_sql_revision run_sql_selection; do
  if ! stage_complete "${stage#run_}"; then
    run_stage "$stage"
  fi
done
if [[ "$experiment" == adaptive-icl ]]; then
  "$python" -m experiments.compact_icl.routing report "$phase"
fi
