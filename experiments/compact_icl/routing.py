"""Prepare and score the paired adaptive-routing experiment."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3

import numpy as np
import tomlkit

from app.pipeline.sql_generation.fast_path import accepts_sql
from experiments.compact_icl.report import execute_result


ROOT = Path("workspace/adaptive-icl")
ORIGINAL = Path("workspace/compact-icl")
POLICIES = ("single_table", "one_join")
PRICE = (0.17, 0.25)
POLICY_SOURCE = Path("app/pipeline/sql_generation/fast_path.py")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(path)


def identity(row):
    return row["db_id"], " ".join(row["question"].lower().split())


def stratify(pool, count, key):
    groups = defaultdict(list)
    for row in pool:
        groups[key(row)].append(row)
    rng = random.Random(42)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while keys and len(result) < count:
        for group in keys[:]:
            result.append(groups[group].pop())
            if not groups[group]:
                keys.remove(group)
            if len(result) == count:
                break
    if len(result) != count:
        raise ValueError("Insufficient questions for the frozen split.")
    return result


def prepare(phase):
    manifest = read_json(ORIGINAL / "bird/manifest.json")
    if phase == "fresh":
        if frozen_policy() not in POLICIES:
            raise ValueError("Validation did not qualify a routing policy.")
        dev = read_json(Path("data/bird/dev/dev.json"))
        excludes = set(manifest["dev_indices"])
        seen = {identity(dev[index]) for index in excludes}
        seen.update(map(identity, read_json(ORIGINAL / "pilot-benchmark/dev/dev.json")))
        pool = [(index, row) for index, row in enumerate(dev)
                if index not in excludes and identity(row) not in seen]
        rows = stratify(pool, 200, lambda entry: (entry[1]["db_id"], entry[1].get("difficulty") or ""))
        source = Path("data/bird/dev/dev_databases")
    else:
        train = [json.loads(line) for line in Path("data/bird/downloads/train.jsonl").read_text().splitlines()]
        validation = [(entry["index"], train[entry["index"]]) for entry in manifest["valid"]]
        index = Path("workspace/runs/_shared/bird/train_full_index/examples.jsonl")
        index_databases = {json.loads(line)["db_id"] for line in index.read_text().splitlines()}
        if index_databases.intersection(row["db_id"] for _, row in validation):
            raise ValueError("Validation databases appear in the retrieval index.")
        pilot = stratify(validation, 20, lambda entry: entry[1]["db_id"])
        pilot_ids = {index for index, _ in pilot}
        rows = pilot if phase == "pilot" else [entry for entry in validation if entry[0] not in pilot_ids]
        source = Path("data/bird/train/train_databases")
    destination = ROOT / phase / "dev"
    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = destination / "dev.json"
    records = [dict(row, question_id=index, evidence=row.get("evidence") or "",
                    difficulty=row.get("difficulty") or "") for index, row in rows]
    if dataset_path.exists() and read_json(dataset_path) != records:
        raise ValueError("The dataset changed. Use a new run directory.")
    write_json(dataset_path, records)
    databases = destination / "dev_databases"
    if not databases.is_symlink():
        databases.symlink_to(source.resolve(), target_is_directory=True)
    if databases.resolve() != source.resolve():
        raise ValueError("Unexpected database path for the frozen split.")
    if any(not (databases / row["db_id"] / (row["db_id"] + ".sqlite")).is_file() for row in records):
        raise ValueError("A selected database is missing.")

    config = tomlkit.parse(Path(__file__).with_name("bird.toml").read_text())
    config["run"]["exp_name"] = f"adaptive-icl-{phase}"
    config["run"]["checkpoint_interval"] = 5
    config["dataset"]["root_path"] = str(destination.parent)
    config["sql_generation"]["icl_prompt_mode"] = "compact"
    config["sql_generation"]["icl_example_count"] = 0
    config["llm_profiles"]["local_icl"]["base_url"] = "http://127.0.0.1:8081/v1"
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        key = Path("config/local/compact-icl/together.key").read_text().strip()
    config["llm_profiles"]["qwen3_coder_instruct"]["api_key"] = key
    target = Path(f"config/local/compact-icl/adaptive-{phase}.toml")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(tomlkit.dumps(config))


def assess(item):
    try:
        expected = execute_result(item.database_path, item.gold_sql)
    except (sqlite3.Error, ValueError):
        expected = None

    def correct(sql):
        if expected is None or not sql:
            return 0
        try:
            return int(execute_result(item.database_path, sql) == expected)
        except (sqlite3.Error, ValueError):
            return 0

    return {
        "id": item.get_item_id(), "database": item.database_id,
        "full": correct(item.final_selected_sql), "first": correct(item.first_icl_sql),
        "question": item.question, "first_sql": item.first_icl_sql, "full_sql": item.final_selected_sql,
        **{policy: accepts_sql(item.question, item.first_icl_sql, item.database_path, policy)
           for policy in POLICIES},
    }


def request_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    allocations = {row["request"]: row for row in rows if "reserve" in row}
    completions = {row["request"]: row for row in rows if "ok" in row}
    return [{**row, **completions.get(number, {})} for number, row in allocations.items()]


def cost(row):
    if row["provider"] != "together":
        return 0
    if row.get("prompt_tokens") is None:
        return row["reserve"] / 1_000_000
    return sum(row[key] * price for key, price in zip(("prompt_tokens", "completion_tokens"), PRICE)) / 1_000_000


def collect(phase):
    from app.dataset import load_dataset

    items = list(load_dataset(f"workspace/runs/adaptive-icl-{phase}/sql_selection.snapshot"))
    expected_ids = {str(row["question_id"]) for row in read_json(ROOT / phase / "dev/dev.json")}
    if len(items) != len(expected_ids) or {item.get_item_id() for item in items} != expected_ids:
        raise ValueError("Snapshot does not match the frozen split.")
    if (ROOT / f"{phase}-usage.jsonl.cap-reached").exists():
        raise ValueError("The budget interrupted this run.")
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(assess, items))
    requests = request_rows(ROOT / f"{phase}-usage.jsonl")
    if not any(row.get("stage") == "run_sql_generation" and row["provider"] == "local" for row in requests):
        raise ValueError("Local generation usage is missing.")
    return results, requests


def summarize(results, requests, labels):
    report = {}
    baseline = np.array([row["full"] for row in results])
    databases = sorted({row["database"] for row in results})
    groups = [np.array([i for i, row in enumerate(results) if row["database"] == db]) for db in databases]
    rng = np.random.default_rng(42)
    samples = [np.concatenate([groups[i] for i in draw])
               for draw in rng.integers(len(groups), size=(5000, len(groups)))]
    full_stages = {"run_sql_generation", "run_sql_revision", "run_sql_selection"}
    optional_requests = [
        request for request in requests
        if request.get("stage") in full_stages and request["provider"] == "together"
    ]
    if any(request.get("item_id") is None for request in optional_requests):
        raise ValueError("Paid downstream usage is missing question attribution.")
    for label in labels:
        first = [label == "first" or (label in POLICIES and row[label]) for row in results]
        selected_ids = {row["id"] for row, use_first in zip(results, first) if use_first}
        scored = np.array([row["first" if use_first else "full"] for row, use_first in zip(results, first)])
        delta = scored - baseline
        used = [row for row in requests if not (
            row["provider"] == "together" and row.get("stage") in full_stages and row.get("item_id") in selected_ids
        )]
        counts = {key: sum(row.get(key) or 0 for row in used) for key in ("prompt_tokens", "completion_tokens")}
        report[label] = {
            "questions": len(results), "correct": int(scored.sum()), "accuracy": float(scored.mean()),
            "fast_path_fraction": float(np.mean(first)), **counts, "tokens": sum(counts.values()),
            "api_usd_with_unknown_usage_reserved": sum(map(cost, used)),
            "requests": len(used), "unknown_usage_requests": sum(row.get("prompt_tokens") is None for row in used),
            "failed_or_unfinished_requests": sum(row.get("ok") is not True for row in used),
            "accuracy_delta_ci95": np.quantile([delta[sample].mean() for sample in samples], [.025, .975]).tolist(),
            "regressions": [row["id"] for row, diff in zip(results, delta) if diff < 0],
            "improvements": [row["id"] for row, diff in zip(results, delta) if diff > 0],
        }
    for result in report.values():
        result["token_savings"] = 1 - result["tokens"] / report["full"]["tokens"]
        result["meets_target"] = result["token_savings"] >= .10 and result["correct"] >= report["full"]["correct"]
    return report


def frozen_policy():
    frozen = read_json(ROOT / "frozen-policy.json")
    if frozen["source_sha256"] != hashlib.sha256(POLICY_SOURCE.read_bytes()).hexdigest():
        raise ValueError("The routing rules changed after validation.")
    if frozen["validation_sha256"] != hashlib.sha256((ROOT / "validation.json").read_bytes()).hexdigest():
        raise ValueError("Validation results changed after routing was frozen.")
    return frozen["policy"]


def report(phase):
    results, requests = collect(phase)
    if phase == "validation":
        pilot_results, pilot_requests = collect("pilot")
        results = pilot_results + results
        requests = pilot_requests + requests
    policies = (frozen_policy(),) if phase == "fresh" else POLICIES
    summary = summarize(results, requests, ("full", "first", *policies))
    selected = min((policy for policy in policies if summary[policy]["meets_target"]),
                   key=lambda policy: summary[policy]["tokens"], default=None)
    recommendation = selected or "full"
    if phase == "fresh" and summary["first"]["meets_target"] and (
        summary["first"]["correct"] >= summary[policies[0]]["correct"]
        and summary["first"]["tokens"] <= summary[policies[0]]["tokens"]
    ):
        recommendation = "first"
    output = ROOT / f"{phase}.json"
    result = {
        "selected_policy": selected, "recommendation": recommendation, "variants": summary,
        "actual_api_usd_with_unknown_usage_reserved": sum(map(cost, requests)),
        "provider_reported_api_usd": sum(cost(request) for request in requests
                                         if request.get("prompt_tokens") is not None),
        "full_rescued": [row["id"] for row in results if row["full"] > row["first"]],
        "full_replaced_correct": [row["id"] for row in results if row["full"] < row["first"]],
    }
    result["examples"] = {
        direction: [
            {key: row[key] for key in ("id", "database", "question", "first_sql", "full_sql")}
            for row in results if row["id"] in ids[:3]
        ]
        for direction, ids in (("full_rescued", result["full_rescued"]),
                               ("full_replaced_correct", result["full_replaced_correct"]))
    }
    if phase == "fresh":
        result["recommended_icl_fast_path"] = "off" if recommendation == "full" else recommendation
    if phase == "validation" and (ROOT / "frozen-policy.json").exists():
        frozen_policy()
        if result != read_json(output):
            raise ValueError("The new result differs from the frozen validation result.")
    else:
        write_json(output, result)
    if phase == "validation" and selected and not (ROOT / "frozen-policy.json").exists():
        write_json(ROOT / "frozen-policy.json", {
            "policy": selected,
            "source_sha256": hashlib.sha256(POLICY_SOURCE.read_bytes()).hexdigest(),
            "validation_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        })
    lines = [
        f"# Adaptive ICL {phase} results", "",
        "| Approach | Correct | Pipeline tokens | Token savings | First answer used | Estimated API cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, variant in summary.items():
        lines.append(f"| {label} | {variant['correct']}/{variant['questions']} | {variant['tokens']:,} | "
                     f"{variant['token_savings']:.1%} | {variant['fast_path_fraction']:.1%} | "
                     f"${variant['api_usd_with_unknown_usage_reserved']:.2f} |")
    lines.extend([
        "", f"Recommendation for this phase: **{recommendation}**.",
        f"Together reported ${result['provider_reported_api_usd']:.2f} in usage for the shared experiment. "
        f"Including reservations for unknown usage, the total is "
        f"${result['actual_api_usd_with_unknown_usage_reserved']:.2f}.",
        f"The full pipeline rescued {len(result['full_rescued'])} first answers and replaced "
        f"{len(result['full_replaced_correct'])} correct ones with incorrect ones.",
        "The JSON report includes paired database bootstrap intervals and the question IDs for disagreements.",
    ])
    if phase == "fresh":
        lines.append(f'Apply with `[sql_generation].icl_fast_path = "{result["recommended_icl_fast_path"]}"`.')
    (ROOT / f"{phase}.md").write_text("\n".join(lines) + "\n")


def budget(phase, verified_credit):
    counts = {"pilot": 20, "validation": 80, "fresh": 200}
    prior = sum(cost(row) for path in (ORIGINAL / "benchmark-usage.jsonl", ORIGINAL / "pilot-usage.jsonl")
                for row in request_rows(path))
    prior += sum(cost(row) for name in counts if name != phase
                 if (path := ROOT / f"{name}-usage.jsonl").exists() for row in request_rows(path))
    current_path = ROOT / f"{phase}-usage.jsonl"
    current = sum(map(cost, request_rows(current_path))) if current_path.exists() else 0
    if phase == "pilot":
        # The previous benchmark includes multiple variants and one-time index work.
        previous = sum(map(cost, request_rows(ORIGINAL / "benchmark-usage.jsonl")))
        estimate = previous * counts[phase] / 200 * 2
    else:
        if phase == "fresh":
            result = read_json(ROOT / "validation.json")
            if frozen_policy() not in POLICIES:
                raise ValueError("Validation did not qualify a policy.")
            estimate = 2 * result["actual_api_usd_with_unknown_usage_reserved"] * counts[phase] / 100
        else:
            estimate = 2 * read_json(ROOT / "pilot.json")["actual_api_usd_with_unknown_usage_reserved"] * counts[phase] / 20
    cap = min(verified_credit + current, 100 - prior, 2 if phase == "pilot" else estimate)
    if not np.isfinite(cap) or cap <= current or estimate > cap:
        raise ValueError(f"Projected cost ${estimate:.2f} does not fit this batch's ${cap:.2f} cap.")
    print(cap)


def remaining_credit(verified_credit):
    used = sum(cost(row) for path in ROOT.glob("*-usage.jsonl")
               if path.name in {"pilot-usage.jsonl", "validation-usage.jsonl", "fresh-usage.jsonl"}
               for row in request_rows(path))
    print(max(0, verified_credit - used))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "report", "budget", "remaining-credit"))
    parser.add_argument("phase", choices=("pilot", "validation", "fresh"), nargs="?")
    parser.add_argument("--verified-credit", type=float)
    args = parser.parse_args()
    if args.action != "remaining-credit" and args.phase is None:
        parser.error("A phase is required.")
    if args.action in ("budget", "remaining-credit"):
        if args.verified_credit is None or not np.isfinite(args.verified_credit):
            parser.error("Provide the current available prepaid credit.")
        if args.action == "budget":
            budget(args.phase, args.verified_credit)
        else:
            remaining_credit(args.verified_credit)
    else:
        (prepare if args.action == "prepare" else report)(args.phase)


if __name__ == "__main__":
    main()
