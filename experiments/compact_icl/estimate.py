"""Estimate the full benchmark's Together cost from the completed pilot."""
import json
from pathlib import Path


ROOT = Path("workspace/compact-icl")
PREPARATION = {"run_value_retrieval", "run_few_shot_preparation", "run_schema_linking", "run_sql_generation"}
DOWNSTREAM = {"run_sql_revision", "run_sql_selection"}


def main():
    rows = [json.loads(line) for line in (ROOT / "pilot-usage.jsonl").read_text().splitlines()]
    stages = {}
    for row in rows:
        if row.get("provider") != "together" or row.get("prompt_tokens") is None:
            continue
        cost = (row["prompt_tokens"] * 0.17 + row["completion_tokens"] * 0.25) / 1_000_000
        stages[row["stage"]] = stages.get(row["stage"], 0) + cost
    if not PREPARATION.issubset(stages):
        raise ValueError("The pilot has incomplete API usage.")
    forecast = (
        stages.get("build_few_shot_index", 0) * 25
        + sum(stages[key] for key in PREPARATION) * 40
        + sum(stages.get(key, 0) for key in DOWNSTREAM) * 160
    )
    # The five-question pilot can miss repair and selection calls entirely.
    # Reserve two extra dollars for that case and double observed spending.
    decision = {"pilot_usd": sum(stages.values()), "projected_usd": forecast,
                "projected_with_margin_usd": forecast * 2 + 2, "stages_usd": stages}
    print(json.dumps(decision, indent=2))
    if decision["projected_with_margin_usd"] > 10:
        raise SystemExit("Estimated full run exceeds the initial $10 benchmark allowance.")


if __name__ == "__main__":
    main()
