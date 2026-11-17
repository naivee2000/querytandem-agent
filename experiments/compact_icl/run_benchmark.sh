#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

out="$PWD/workspace/compact-icl"
local_configs="$PWD/config/local/compact-icl"
export DEEPEYE_COUNT_NO_CANDIDATE=1
export PYTHONPATH="$PWD" HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
export DEEPEYE_USAGE_LEDGER="$out/benchmark-usage.jsonl" DEEPEYE_API_CAP_USD=10

test -e "$out/pilot-runs/compact-icl-pilot/sql_selection.snapshot"
.venv/bin/python -m experiments.compact_icl.estimate > "$out/estimate.json"
adapter=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["adapter"])' "$out/selection.json")
examples=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["examples"])' "$out/selection.json")
.venv/bin/python -m experiments.compact_icl.configure --examples "$examples"

model=$(find .cache/huggingface/hub -type d -path '*/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*' | head -1)
test -d "$model"
test -f "$adapter/adapters.safetensors"

run_stage() {
  export DEEPEYE_STAGE="$1"
  .venv/bin/python "runner/$1.py"
}

start_server() {
  port=$1
  shift
  workspace/mlx-env/bin/mlx_lm.server --model "$model" --host 127.0.0.1 --port "$port" --prompt-cache-size 2 --prompt-cache-bytes 536870912 --prefill-step-size 512 --decode-concurrency 1 --prompt-concurrency 1 "$@" > "$out/mlx-$port.log" 2>&1 &
  server_pid=$!
  for ((attempt=0; attempt<60; attempt++)); do
    kill -0 "$server_pid"
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done
  return 1
}

stop_server() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}
trap stop_server EXIT

export CONFIG_PATH="$local_configs/baseline.toml"
if [[ ! -s workspace/runs/compact-icl-baseline/schema_linking.snapshot ]]; then
  for stage in preprocess_dataset create_vector_db_parallel run_value_retrieval build_few_shot_index run_few_shot_preparation run_schema_linking; do
    run_stage "$stage"
  done
fi

export DEEPEYE_SHARED_CANDIDATES="$out/benchmark-candidates"
baseline_snapshot="$PWD/workspace/runs/compact-icl-baseline/schema_linking.snapshot"
# Point the other arms at the same immutable schema-linking output.
BASELINE_SCHEMA="$baseline_snapshot" .venv/bin/python - <<'PY'
import os
from pathlib import Path
import tomlkit

for label in ("prompt-only", "training-only", "tuned-compact"):
    path = Path("config/local/compact-icl") / f"{label}.toml"
    config = tomlkit.parse(path.read_text())
    config["schema_linking"]["save_path"] = os.environ["BASELINE_SCHEMA"]
    path.write_text(tomlkit.dumps(config))
PY

snapshots=()
for variant in baseline prompt-only training-only tuned-compact; do
  export CONFIG_PATH="$local_configs/$variant.toml"
  export DEEPEYE_ICL_METRICS="$out/benchmark-icl/$variant"
  export DEEPEYE_SHARED_CANDIDATES_MODE=replay
  if [[ "$variant" == baseline ]]; then
    export DEEPEYE_SHARED_CANDIDATES_MODE=record
  fi
  if [[ ! -s "workspace/runs/compact-icl-$variant/sql_generation.snapshot" ]]; then
    stop_server
    case "$variant" in
      baseline|prompt-only) start_server 8080 ;;
      *) start_server 8081 --adapter-path "$adapter" ;;
    esac
  fi
  for stage in run_sql_generation run_sql_revision run_sql_selection; do
    snapshot="workspace/runs/compact-icl-$variant/${stage#run_}.snapshot"
    if [[ ! -s "$snapshot" ]]; then
      run_stage "$stage"
    fi
  done
  snapshots+=("workspace/runs/compact-icl-$variant/sql_selection.snapshot")
done
stop_server
.venv/bin/python -m experiments.compact_icl.report --icl-metrics "$out/benchmark-icl" "${snapshots[@]}" > "$out/results.pending.json"
mv "$out/results.pending.json" "$out/results.json"
