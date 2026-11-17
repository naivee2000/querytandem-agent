"""Report paired BIRD execution accuracy and token use from four run snapshots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from itertools import groupby
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from app.dataset import load_dataset



VARIANTS = ("baseline", "prompt-only", "training-only", "tuned-compact")


def execute_result(db_path, sql):
    deadline = time.monotonic() + 30
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(100_001)
        if len(rows) > 100_000:
            raise ValueError("Result exceeds the evaluation row limit.")
        return set(rows)


def score(item):
    if not item.final_selected_sql:
        return 0
    try:
        return int(execute_result(item.database_path, item.final_selected_sql)
                   == execute_result(item.database_path, item.gold_sql))
    except (sqlite3.Error, ValueError):
        return 0


def summarize(items, scores):
    def totals(field):
        result = {}
        for token in ("prompt_tokens", "completion_tokens", "total_tokens"):
            result[token] = sum((getattr(item, field, None) or {}).get(token, 0) for item in items)
        return result

    times = [item.total_time for item in items]
    return {
        "questions": len(items),
        "correct": sum(scores),
        "execution_accuracy": float(np.mean(scores)),
        "generation_tokens": totals("sql_generation_llm_cost"),
        "pipeline_tokens": totals("total_llm_cost"),
        "median_recorded_seconds": float(np.median(times)),
    }


def recommend_variant(report):
    baseline = report["baseline"]
    base_tokens = baseline["pipeline_tokens"]["total_tokens"]
    eligible = [
        label for label in ("prompt-only", "tuned-compact")
        if report[label]["correct"] >= baseline["correct"]
        and report[label]["pipeline_tokens"]["total_tokens"] < base_tokens
    ]
    if not eligible:
        return "baseline"
    # Prefer the best observed accuracy, then fewer tokens, then the simpler setup.
    return min(eligible, key=lambda label: (
        -report[label]["correct"],
        report[label]["pipeline_tokens"]["total_tokens"],
        label != "prompt-only",
    ))


def write_summary(path, report):
    baseline = report["baseline"]
    tuned = report["tuned-compact"]
    prompt = report["prompt-only"]
    base_tokens = baseline["pipeline_tokens"]["total_tokens"]
    prompt_tokens = prompt["pipeline_tokens"]["total_tokens"]
    tuned_tokens = tuned["pipeline_tokens"]["total_tokens"]
    prompt_savings = (base_tokens - prompt_tokens) / base_tokens
    tuned_savings = (base_tokens - tuned_tokens) / base_tokens
    fine_tuning_delta = tuned["execution_accuracy"] - prompt["execution_accuracy"]
    fine_tuning_token_delta = (tuned_tokens - prompt_tokens) / prompt_tokens
    recommendation = recommend_variant(report)
    actions = {
        "baseline": "Keep the baseline. Document both compact variants as experiments.",
        "prompt-only": "Lead with the compact prompt and original model. Document the adapter experiment.",
        "tuned-compact": "Keep the compact prompt and fine-tuned adapter as the contribution.",
    }
    chosen = report[recommendation]
    chosen_tokens = chosen["pipeline_tokens"]["total_tokens"]
    lines = [
        "# Compact ICL results",
        "",
        f"**Recommendation:** {actions[recommendation]}",
        "",
        f"The recommended configuration answered {chosen['correct']} of {chosen['questions']} "
        f"questions correctly and used {chosen_tokens:,} pipeline tokens. "
        "The decision prioritizes observed execution accuracy among configurations "
        "that match baseline accuracy or better and use fewer tokens. "
        "At equal accuracy, prefer fewer tokens, then the original model.",
        "",
        "All four variants used the same 200 held-out BIRD questions. "
        "Execution accuracy compares SQLite result sets. The scorer counts "
        "unanswered queries, SQL errors, timeouts, and results exceeding the row limit as incorrect.",
        "",
        "| Variant | Correct | Accuracy | Pipeline tokens | Local ICL tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, variant in report.items():
        lines.append(
            f"| {label} | {variant['correct']}/{variant['questions']} | "
            f"{variant['execution_accuracy']:.1%} | "
            f"{variant['pipeline_tokens']['total_tokens']:,} | "
            f"{variant['icl_tokens']['total_tokens']:,} |"
        )
    low, high = tuned["paired_accuracy_delta_ci95"]
    lines.extend([
        "",
        f"The compact prompt with the original model changed observed accuracy from "
        f"{baseline['execution_accuracy']:.1%} to {prompt['execution_accuracy']:.1%} "
        f"and saved {prompt_savings:+.1%} of pipeline tokens.",
        "",
        f"The fine-tuned model with the compact prompt saved {tuned_savings:+.1%} "
        f"of pipeline tokens versus baseline. The paired accuracy difference was "
        f"{100 * tuned['paired_accuracy_delta']:+.1f} percentage points "
        f"(95% database bootstrap interval: {100 * low:+.1f} to {100 * high:+.1f}). "
        f"The measured target was {'met' if tuned['meets_target'] else 'missed'}.",
        "",
        f"With the compact prompt held fixed, fine-tuning changed observed accuracy by "
        f"{100 * fine_tuning_delta:+.1f} percentage points and pipeline token use by "
        f"{fine_tuning_token_delta:+.1%}.",
        "",
        "## Résumé draft",
        "",
        f"Fine-tuned a 4B Qwen model locally with MLX and LoRA for DeepEye-SQL, "
        f"then built a four-variant evaluation on 200 held-out BIRD questions. "
        + (
            f"The recommended configuration saved {(base_tokens - chosen_tokens) / base_tokens:.1%} "
            f"of pipeline tokens at {chosen['execution_accuracy']:.1%} accuracy "
            f"versus {baseline['execution_accuracy']:.1%} for the baseline."
            if recommendation != "baseline"
            else "Measured the accuracy and token tradeoffs of compact prompting and fine-tuning."
        ),
        "",
    ])
    pending = path.with_suffix(".pending")
    pending.write_text("\n".join(lines))
    pending.replace(path)


def successful_local_usage(path):
    """Recover successful model calls, including empty responses, from this run's ledger."""
    rows = (json.loads(line) for line in path.read_text().splitlines())
    calls = (row for row in rows if "stage" in row)
    totals = []
    for stage, group in groupby(calls, key=lambda row: row["stage"]):
        local = [row for row in group if row["provider"] == "local"]
        if stage != "run_sql_generation" or len(local) < 200:
            continue
        if any(not row["ok"] or row["prompt_tokens"] is None for row in local):
            continue
        prompt = sum(row["prompt_tokens"] for row in local)
        completion = sum(row["completion_tokens"] for row in local)
        totals.append(dict(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion))
    if len(totals) != len(VARIANTS):
        raise ValueError("Expected four complete local generation groups in the usage ledger.")
    return dict(zip(VARIANTS, totals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs=4, type=Path, metavar="SNAPSHOT")
    parser.add_argument("--icl-metrics", type=Path)
    parser.add_argument("--recover-local-usage", type=Path)
    args = parser.parse_args()
    recovered = successful_local_usage(args.recover_local_usage) if args.recover_local_usage else {}
    report, paired = {}, {}
    reference_ids = None
    with ProcessPoolExecutor(max_workers=2) as pool:
        for label, snapshot in zip(VARIANTS, args.snapshots):
            items = sorted(load_dataset(str(snapshot)), key=lambda item: item.question_id)
            ids = [(item.question_id, item.database_id) for item in items]
            if not ids or len(ids) != len(set(ids)) or (reference_ids is not None and ids != reference_ids):
                raise ValueError(f"{label} has missing, duplicated, or unpaired questions")
            reference_ids = ids
            scores = list(pool.map(score, items))
            report[label] = summarize(items, scores)
            if args.icl_metrics:
                branch = [
                    json.loads((args.icl_metrics / label / f"{question}.json").read_text())
                    for question, _ in ids
                ]
                if label not in recovered and any(row["total_tokens"] <= 0 for row in branch):
                    raise ValueError(f"{label} has questions without recorded local model usage")
                report[label]["icl_tokens"] = {
                    key: sum(row[key] for row in branch)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
                if label in recovered:
                    recorded = report[label]["icl_tokens"]
                    for key, value in recovered[label].items():
                        difference = value - recorded[key]
                        if difference < 0:
                            raise ValueError(f"{label} ledger has fewer {key} than the item records")
                        for field in ("generation_tokens", "pipeline_tokens"):
                            report[label][field][key] += difference
                    report[label]["icl_tokens"] = recovered[label]
            paired[label] = np.asarray(scores)
    # Resample whole databases to account for dependent questions within a schema.
    groups = [
        np.array([index for index, (_, actual) in enumerate(reference_ids) if actual == db_id])
        for db_id in sorted({db_id for _, db_id in reference_ids})
    ]
    rng = np.random.default_rng(42)
    samples = [np.concatenate([groups[index] for index in indices])
               for indices in rng.integers(len(groups), size=(5000, len(groups)))]
    for label in VARIANTS[1:]:
        delta = paired[label] - paired["baseline"]
        low, high = np.quantile([delta[indices].mean() for indices in samples], [0.025, 0.975])
        report[label]["paired_accuracy_delta"] = float(delta.mean())
        report[label]["paired_accuracy_delta_ci95"] = [float(low), float(high)]
    baseline = report["baseline"]
    tuned = report["tuned-compact"]
    initial_tokens = baseline["pipeline_tokens"]["total_tokens"]
    final_tokens = tuned["pipeline_tokens"]["total_tokens"]
    tuned["pipeline_token_savings_fraction"] = (initial_tokens - final_tokens) / initial_tokens if initial_tokens else None
    tuned["meets_target"] = final_tokens < initial_tokens and tuned["correct"] >= baseline["correct"]
    ledger = os.environ.get("DEEPEYE_USAGE_LEDGER")
    if args.icl_metrics and ledger:
        write_summary(Path(ledger).with_name("findings.md"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
