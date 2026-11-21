"""Export verified recorded examples; stdlib only, with no model/API calls."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/demo/demo-data.json"
SOURCE_COMMIT = "ef2e6dbb8610835a4396167215cf21eba2928104"
REPOSITORY = "https://github.com/naivee2000/querytandem-agent"
REPORT = "experiments/compact_icl/results/compact-icl.json"
SELECTOR = "experiments/compact_icl/results/selector-validation.json"
VARIANTS = ("baseline", "tuned-compact")
CASES = (
    (358, "both_correct", (True, True), "Same correct answer, different token costs",
     "Both pipelines return black. The compact configuration uses fewer local-generator tokens, "
     "but its recorded whole-pipeline token count is slightly higher for this question."),
    (507, "regression", (True, False), "A regression: counting cards instead of sets",
     "The compact configuration's final query joins individual cards and changes the denominator. "
     "The reference percentage is about 0.8264%; the compact answer is about 1.0920%."),
    (341, "both_incorrect", (False, False), "Shared failure: extra rows despite a similar preview",
     "Both pipelines select the same SQL and return 72 rows rather than the reference's 67. "
     "Matching preview rows do not make the complete result correct."),
)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def authorize(action, _name, _column, _database, _source):
    allowed = (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE)
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def execute_readonly(path, sql, *, timeout=30., max_rows=100000, preview_limit=10):
    """Compare complete unordered row sets, separately from the display preview."""
    result = dict(status="missing_sql", columns=[], rows_preview=[], row_count=None,
                  distinct_row_count=None, preview_limit=preview_limit, truncated=False,
                  preview_order="canonical_display_order")
    if not sql:
        return result, None
    deadline = time.monotonic() + timeout
    try:
        if time.monotonic() >= deadline:
            result["status"] = "timeout"
            return result, None
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only = ON")
            conn.set_authorizer(authorize)
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
            if time.monotonic() >= deadline:
                result["status"] = "timeout"
                return result, None
            if len(rows) > max_rows:
                result["status"] = "row_limit"
                return result, None
            row_set = frozenset(rows)
            preview = sorted(rows, key=canonical)[:preview_limit]
            result.update(status="ok" if rows else "empty", columns=[col[0] for col in cursor.description],
                          rows_preview=[list(row) for row in preview], row_count=len(rows),
                          distinct_row_count=len(row_set), truncated=len(rows) > preview_limit)
            return result, row_set
    except sqlite3.Error:
        result["status"] = "timeout" if time.monotonic() >= deadline else "sql_error"
        return result, None


def selected_records(root, variant, sources):
    name = f"workspace/runs/compact-icl-{variant}/sql_selection.snapshot"
    path = root / name
    manifest = read_json(path)
    if manifest["format"] != "structured_dataset_snapshot" or manifest["version"] != 1:
        raise ValueError("Unsupported snapshot format.")
    items = path.parent / manifest["snapshot_root"] / "items.jsonl"
    sources[name] = sha256(path)
    sources[items.relative_to(root).as_posix()] = sha256(items)
    records = {}
    with items.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row["input"]["question_id"]
            if key in {case[0] for case in CASES}:
                if key in records:
                    raise ValueError("Duplicate source question.")
                records[key] = row
    if len(records) != len(CASES):
        raise ValueError("A selected source question is missing.")
    return records


def schema_summary(record):
    schema = record["pipeline_artifacts"]["schema_linking"]["database_schema_after_schema_linking"]
    return [dict(name=name, columns=[dict(name=column, type=value["column_type"])
                                    for column, value in sorted(table["columns"].items())])
            for name, table in sorted(schema["tables"].items())]


def export(root=ROOT):
    sources = {REPORT: sha256(root / REPORT), SELECTOR: sha256(root / SELECTOR)}
    records = {variant: selected_records(root, variant, sources) for variant in VARIANTS}
    report = read_json(root / REPORT)
    selector = read_json(root / SELECTOR)
    cases = []
    for question_id, category, expected_scores, title, explanation in CASES:
        first = records["baseline"][question_id]
        reference = first["input"]
        if reference["database_id"] != "card_games":
            raise ValueError("The selected example's database changed.")
        database = Path(reference["database_path"])
        if not database.is_absolute():
            database = root / database
        db_source = "data/bird/dev/dev_databases/card_games/card_games.sqlite"
        if db_source not in sources:
            sources[db_source] = sha256(database)
        expected, expected_rows = execute_readonly(database, reference["gold_sql"])
        if expected_rows is None:
            raise ValueError(f"Reference execution failed for question {question_id}.")
        variants = {}
        for variant, should_match in zip(VARIANTS, expected_scores):
            record = records[variant][question_id]
            for key in ("question", "evidence", "database_id", "gold_sql"):
                if record["input"][key] != reference[key]:
                    raise ValueError(f"Unpaired {key} for question {question_id}.")
            if schema_summary(record) != schema_summary(first):
                raise ValueError("Variants did not use the same linked schema.")
            artifacts = record["pipeline_artifacts"]
            sql = artifacts["sql_selection"]["final_selected_sql"]
            answer, rows = execute_readonly(database, sql)
            correct = rows is not None and rows == expected_rows
            if correct != should_match:
                raise ValueError(f"Unexpected outcome for {variant}, question {question_id}.")
            metrics_path = f"workspace/compact-icl/benchmark-icl/{variant}/{question_id}.json"
            sources[metrics_path] = sha256(root / metrics_path)
            variants[variant] = dict(final_pipeline_sql=sql, result=answer, correct=correct,
                                     tokens=dict(local_generator=read_json(root / metrics_path),
                                                 pipeline=artifacts["metrics"]["total_llm_cost"]))
        cases.append(dict(id=f"bird-dev-{question_id}", question_id=question_id, database_id="card_games",
                          category=category, title=title, explanation=explanation,
                          question=reference["question"], evidence=reference["evidence"],
                          available_schema=schema_summary(first), reference_result=expected,
                          gold_sql_sha256=hashlib.sha256(reference["gold_sql"].encode()).hexdigest(),
                          variants=variants))
    benchmark_variants = {key: {field: row[field] for field in (
        "questions", "correct", "execution_accuracy", "pipeline_tokens", "icl_tokens"
    )} for key, row in report.items()}
    base, tuned = report["baseline"], report["tuned-compact"]
    bundle = dict(
        schema_version=1, mode="recorded_examples", title="QueryTandem Agent",
        source=dict(repository=REPOSITORY, reference_commit=SOURCE_COMMIT,
                    reference_commit_meaning="Published experiment reference, not the demo-package commit.",
                    artifact_sha256=sources,
                    exporter_sha256=sha256(Path(__file__)),
                    result_provenance="Saved final SQL predictions, locally reexecuted against the benchmark database; no new model calls."),
        benchmark=dict(dataset="BIRD dev", questions=200, variants=benchmark_variants,
                       local_generator_token_savings_fraction=1-tuned["icl_tokens"]["total_tokens"]/base["icl_tokens"]["total_tokens"],
                       pipeline_token_savings_fraction=tuned["pipeline_token_savings_fraction"],
                       accuracy_delta=tuned["paired_accuracy_delta"], accuracy_delta_ci95=tuned["paired_accuracy_delta_ci95"],
                       confidence_interval_method="95% paired database bootstrap; interval includes zero."),
        selector_followup=dict(questions=100, split="Separate validation set, not the 200-question benchmark",
                               current_correct=selector["validation"]["variants"]["current"]["correct"],
                               trained_correct=selector["validation"]["variants"]["0.9"]["correct"],
                               adopted=False, fresh_benchmark_run=selector["fresh_benchmark_run"]),
        variant_labels={"baseline": "Original model · 7 prompt examples", "tuned-compact": "Fine-tuned model · 0 prompt examples"},
        metric_notes=dict(
            sql="Final answer selected by the full DeepEye pipeline; not necessarily written by the local generator.",
            local_generator="Per-example ICL metrics from saved successful stage records; input plus output tokens.",
            pipeline="Per-example pipeline snapshot totals, including the shared preprocessing/generator attribution.",
            reconciliation="Aggregate benchmark counts were reconciled with the request ledger, including retries; per-example records are not a complete allocation of all retry spending.",
            correctness="Complete unordered sets of result rows match gold SQL; row order, duplicate multiplicity and column aliases are ignored, following this experiment's scorer.",
            previews="At most 10 rows, sorted for stable display; correctness is checked before preview truncation.",
            selection="Three deliberately selected illustrations from one database; do not calculate benchmark accuracy from them.",
            latency="No per-example runtime, API cost, or live-response-speed claim is supplied."),
        attribution=dict(project="QueryTandem Agent", upstream="HKUSTDial/DeepEye-SQL",
                         upstream_url="https://github.com/HKUSTDial/DeepEye-SQL", code_license="MIT",
                         model="Qwen3-4B-Instruct-2507; Apache-2.0", model_url="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
                         dataset="BIRD benchmark (BIRD team)", dataset_url="https://bird-bench.github.io/",
                         example_data_license="CC BY-SA 4.0", example_data_license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                         changes="Selected three BIRD dev examples, reduced schema metadata, added model predictions and evaluation labels, and truncated/sorted result previews."),
        examples=cases,
    )
    validate(bundle)
    return bundle


def validate(bundle):
    if bundle["schema_version"] != 1 or bundle["mode"] != "recorded_examples":
        raise ValueError("Unsupported demo data format.")
    if [row["question_id"] for row in bundle["examples"]] != [case[0] for case in CASES]:
        raise ValueError("Example IDs or their order differ from the handoff.")
    benchmark = bundle["benchmark"]
    for counts in benchmark["variants"].values():
        if counts["questions"] != 200 or counts["execution_accuracy"] != counts["correct"] / counts["questions"]:
            raise ValueError("Benchmark accuracy does not match its counts.")
    base, tuned = (benchmark["variants"][key] for key in VARIANTS)
    for name, metric in (("local_generator_token_savings_fraction", "icl_tokens"), ("pipeline_token_savings_fraction", "pipeline_tokens")):
        expected = 1-tuned[metric]["total_tokens"]/base[metric]["total_tokens"]
        if abs(benchmark[name] - expected) > 1e-12:
            raise ValueError("Savings do not match token counts.")
    for example, definition in zip(bundle["examples"], CASES):
        if example["category"] != definition[1]:
            raise ValueError("Example category differs from its verified outcome.")
        for variant, expected_correct in zip(VARIANTS, definition[2]):
            row = example["variants"][variant]
            if row["correct"] is not expected_correct:
                raise ValueError("Unexpected correctness label.")
            for tokens in row["tokens"].values():
                if tokens is not None and (any(type(tokens[key]) is not int or tokens[key] < 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                                          or tokens["total_tokens"] != tokens["prompt_tokens"] + tokens["completion_tokens"]):
                    raise ValueError("Invalid token counts.")
        for result in [example["reference_result"], *(row["result"] for row in example["variants"].values())]:
            if result["status"] in ("ok", "empty"):
                count = result["row_count"]
                if (len(result["rows_preview"]) != min(count, 10) or result["truncated"] != (count > 10)
                        or any(len(row) != len(result["columns"]) for row in result["rows_preview"])
                        or (result["status"] == "empty") != (count == 0)):
                    raise ValueError("Invalid result preview.")
            elif result["status"] not in ("missing_sql", "timeout", "row_limit", "sql_error") or result["rows_preview"]:
                raise ValueError("Invalid execution failure.")
    if "/Users/" in canonical(bundle) or len(canonical(bundle).encode()) > 250000:
        raise ValueError("The bundle contains a local path or exceeds its size budget.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", action="store_true", help="Validate the portable JSON only; no database/workspace needed")
    group.add_argument("--check", action="store_true", help="Reexecute saved SQL locally and compare with the published bundle")
    args = parser.parse_args()
    if args.verify:
        validate(read_json(args.output))
    else:
        bundle = export()
        if args.check:
            if bundle != read_json(args.output):
                raise ValueError("Export differs from the saved bundle.")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pending = args.output.with_suffix(".pending")
            pending.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            pending.replace(args.output)
    print("Verified three recorded examples and benchmark metric consistency.")


if __name__ == "__main__":
    main()
