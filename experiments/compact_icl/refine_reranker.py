"""Resume the selector experiment with offline labels and local augmentation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from app.prompt import PromptFactory
from app.services import get_schema_service
from app.services.schema_service import PROFILE_STRIPPING_LEVELS
from .rerank_data import ROOT, split
from .rerank_experiment import (
    MODEL, RUNTIME_SOURCE, batch_items, exclusive_run, fit, labeled, ledger,
    report_fresh, report_validation, run_batch, snapshot, usage,
)
from .routing import read_json, request_rows, write_json

STATUS = ROOT / "refinement-status.json"
PLAN = ROOT / "refinement-plan.json"
AUGMENT_SCRIPT = Path("experiments/compact_icl/augment_reranker.py")
SHARDS = tuple(f"refine-{index:02d}" for index in range(1, 10))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frozen_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Frozen experiment input changed: {path}")
    if not path.exists():
        write_json(path, value)


def spending():
    rows = [row for path in sorted(ROOT.glob("*-usage.jsonl")) for row in request_rows(path)]
    return usage(rows)


def status(stage, **details):
    value = dict(stage=stage, updated_at=time.time(), usage=spending(), **details)
    write_json(STATUS, value)
    print(json.dumps(value), flush=True)


def freeze_plan(cap):
    models = sorted(Path(".cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots").glob("*"))
    if len(models) != 1:
        raise ValueError("Expected one cached Qwen3-4B model snapshot.")
    adapter = read_json(Path("workspace/compact-icl/selection.json"))["selected"]["adapter"]
    generation = dict(model=str(models[0]), adapter=adapter, samples=3, temperature=0.7,
                      top_p=0.9, max_tokens=512, context_tokens=8192,
                      adapter_sha256=hashlib.sha256((Path(adapter) / "adapters.safetensors").read_bytes()).hexdigest())
    sources = [RUNTIME_SOURCE, AUGMENT_SCRIPT, Path(__file__),
               Path("experiments/compact_icl/rerank_experiment.py"),
               Path("experiments/compact_icl/rerank_data.py"),
               Path("experiments/compact_icl/run_routing.sh"),
               Path("experiments/compact_icl/bird.toml")]
    plan = dict(version=1, total_cap_usd=cap, revised_pilot_cap_usd=1.0,
                minimum_fresh_reserve_usd=3.2, label_timeout_per_query_seconds=30,
                runtime_shared_timeout_seconds=3, minimum_informative=60,
                validation_minimum_correct=79, thresholds=[.60, .75, .90],
                generation=generation, train_phases=["pilot", *SHARDS],
                source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    frozen_json(PLAN, plan)
    return plan


def assert_training_only(items):
    permitted = {str(index): row for phase in ("pilot", "train") for index, row in split(phase)[0]}
    seen = set()
    for item in items:
        key = item.get_item_id()
        row = permitted.get(key)
        if key in seen or row is None or (item.database_id, item.question, item.gold_sql.strip()) != (
                row["db_id"], row["question"], row["SQL"].strip()):
            raise ValueError("Training item is duplicated or outside the frozen training split.")
        seen.add(key)


def label_items(items, phase, kind):
    directory = ROOT / "offline-labels" / phase / kind
    directory.mkdir(parents=True, exist_ok=True)
    label_sources = [RUNTIME_SOURCE, Path("experiments/compact_icl/rerank_experiment.py")]
    code_hash = digest([hashlib.sha256(path.read_bytes()).hexdigest() for path in label_sources])

    def one(item):
        db = Path(item.database_path).stat()
        fingerprint = digest(dict(code=code_hash, id=item.get_item_id(), db=item.database_id,
                                  database_size=db.st_size, database_mtime_ns=db.st_mtime_ns,
                                  question=item.question, evidence=item.evidence, gold=item.gold_sql,
                                  candidates=item.sql_candidates_after_revision))
        path = directory / f"{item.get_item_id()}.json"
        if path.exists():
            cached = read_json(path)
            if cached["fingerprint"] != fingerprint:
                raise ValueError("Offline labels differ from their frozen candidates or database.")
            return cached
        diagnostics = {}
        good, bad = labeled(item, diagnostics=diagnostics)
        record = dict(fingerprint=fingerprint, good=good, bad=bad, diagnostics=diagnostics)
        write_json(path, record)
        print(f'Offline labels [{phase}/{kind}] question {item.get_item_id()}: '
              f'{len(good)} correct groups, {len(bad)} incorrect groups', flush=True)
        return record

    with ThreadPoolExecutor(max_workers=2) as workers:
        return list(workers.map(one, items))


def prompt_records(items):
    service = get_schema_service()
    rows = []
    for item in items:
        schema = item.database_schema_after_schema_linking
        if not schema:
            raise ValueError("Local augmentation requires the cached inference schema.")
        prompts = [PromptFactory.format_compact_icl_prompt(
            [], service.build_schema_profile(schema, **level), item.question, item.evidence or "", "SQLite"
        ).strip() for level in PROFILE_STRIPPING_LEVELS]
        rows.append(dict(item_id=item.get_item_id(), database_id=item.database_id, prompts=prompts))
    return rows


def augment(items, phase, plan):
    directory = ROOT / "augmentation" / phase
    manifest = directory / "manifest.json"
    frozen_json(manifest, dict(generation=plan["generation"], items=prompt_records(items)))
    status("local_augmentation", phase=phase, questions=len(items),
           progress_file=str(directory / "progress.json"))
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HOME=str(Path(".cache/huggingface").resolve()))
    with (ROOT / f"{phase}-augmentation.log").open("a") as log:
        subprocess.run([
            "workspace/mlx-env/bin/python", str(AUGMENT_SCRIPT), "--manifest", str(manifest),
            "--output", str(directory / "samples"), "--ledger", str(ROOT / f"{phase}-augmentation-usage.jsonl"),
        ], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    augmented = []
    for item in items:
        candidates = list(item.sql_candidates_after_revision or [])
        for sample in range(plan["generation"]["samples"]):
            row = read_json(directory / "samples" / f"{item.get_item_id()}-{sample}.json")
            if row["manifest_sha256"] != manifest_hash or row["database_id"] != item.database_id:
                raise ValueError("Generated candidates belong to another input.")
            if row["sql"]:
                candidates.append(row["sql"])
        augmented.append(item.model_copy(update={"sql_candidates_after_revision": candidates}))
    return augmented


def summarize_labels(records):
    counts = Counter(row["status"] for record in records for row in record["diagnostics"]["candidates"])
    return dict(questions=len(records), informative=sum(bool(row["good"] and row["bad"]) for row in records),
                candidate_execution_statuses=dict(counts),
                gold_execution_statuses=dict(Counter(row["diagnostics"]["gold"]["status"] for row in records)))


def process_phase(phase, plan):
    items = batch_items(phase)
    assert_training_only(items)
    status("offline_labeling", phase=phase, questions=len(items))
    original = label_items(items, phase, "original")
    augmented = augment(items, phase, plan)
    status("offline_labeling_augmented", phase=phase, questions=len(items))
    enriched = label_items(augmented, phase, "augmented")
    report = dict(original=summarize_labels(original), augmented=summarize_labels(enriched))
    write_json(ROOT / f"{phase}-refinement.json", report)
    return items, enriched, report


def forecast(questions, informative, paid_pipeline_usd, cap, spent):
    per_question = 1.2 * paid_pipeline_usd / questions
    fresh_reserve = max(3.2, per_question * 200)
    # Discount the observed label yield as well as buffering paid cost.
    yield_rate = 0.8 * informative / questions
    missing = max(0, 60 - informative)
    additional = math.ceil(missing / yield_rate / 40) * 40 if yield_rate else 400
    remaining = cap - spent
    projected = spent + per_question * additional + fresh_reserve
    return dict(additional_questions=additional, fresh_reserve_usd=fresh_reserve,
                projected_total_usd=projected, expected_batch_usd=per_question * 40,
                feasible=questions + additional <= 400 and projected <= cap,
                remaining_training_allowance_usd=max(0., remaining - fresh_reserve))


def run(cap):
    plan = freeze_plan(cap)
    if (ROOT / "refinement-complete.json").exists():
        status("complete", report=str(ROOT / "refinement-complete.json"))
        return
    if spending()["api_usd_with_reservations"] >= cap:
        raise ValueError("No experiment allowance remains.")
    items, records, pilot = process_phase("pilot", plan)
    if pilot["augmented"]["informative"] <= pilot["original"]["informative"]:
        status("stopped_no_label_gain", reason="Local candidates did not add informative training questions.")
        return
    phases = ["pilot"]
    for phase in SHARDS:
        totals = summarize_labels(records)
        if totals["informative"] >= 60:
            break
        paid = sum(usage(ledger(p))["api_usd_with_reservations"] for p in phases)
        spent = spending()["api_usd_with_reservations"]
        projection = forecast(len(items), totals["informative"], paid, cap, spent)
        write_json(ROOT / "refinement-forecast.json", projection)
        if not projection["feasible"]:
            status("stopped_budget_forecast", **projection, informative=totals["informative"])
            return
        # The first new 40-question pilot has an absolute $1 cap, including retries.
        own_spend = usage(ledger(phase))["api_usd_with_reservations"]
        batch_cap = min(1., cap - (spent - own_spend) - projection["fresh_reserve_usd"])
        if projection["expected_batch_usd"] > batch_cap:
            status("stopped_batch_forecast", phase=phase, batch_cap_usd=batch_cap, **projection)
            return
        status("paid_training_batch", phase=phase, batch_cap_usd=batch_cap,
               informative=totals["informative"], **projection)
        run_batch(phase, batch_cap)
        more_items, more_records, _ = process_phase(phase, plan)
        items.extend(more_items)
        records.extend(more_records)
        phases.append(phase)
        assert_training_only(items)
        # Every completed shard refreshes the forecast before more API spending.

    training = summarize_labels(records)
    training.update(phases=phases, generation=plan["generation"], usage=spending(),
                    augmentation="Original repaired candidates plus locally generated executable candidates; training only.")
    training_hash = digest([row["fingerprint"] for row in records])
    write_json(ROOT / "refinement-training.json", training)
    weights = fit([(row["good"], row["bad"]) for row in records])
    if weights is None:
        status("stopped_insufficient_training", informative=training["informative"])
        return
    status("validation", informative=training["informative"], questions=len(items))
    if MODEL.exists():
        artifact = read_json(MODEL)
        if artifact.get("training_sha256") != training_hash or artifact.get("refinement_plan_sha256") != digest(plan):
            raise ValueError("Existing ranker was fitted on different data.")
        qualified = artifact["qualified"]
    else:
        qualified = report_validation(weights, items, training["informative"], metadata={
            "training_sha256": training_hash, "refinement_plan_sha256": digest(plan),
        })
    if not qualified:
        result = dict(recommendation="current", reason="The trained selector did not pass the validation gate.",
                      validation=read_json(ROOT / "validation.json"), usage=spending())
        write_json(ROOT / "refinement-complete.json", result)
        status("complete_validation_not_qualified", report=str(ROOT / "refinement-complete.json"))
        return
    if read_json(MODEL)["runtime_sha256"] != hashlib.sha256(RUNTIME_SOURCE.read_bytes()).hexdigest():
        raise ValueError("Frozen ranker runtime changed after validation.")
    own_spend = usage(ledger("fresh"))["api_usd_with_reservations"]
    remaining = cap - (spending()["api_usd_with_reservations"] - own_spend)
    paid = sum(usage(ledger(p))["api_usd_with_reservations"] for p in phases)
    reserve = max(3.2, 1.2 * paid / len(items) * 200)
    if reserve > remaining:
        status("stopped_fresh_forecast", projected_fresh_usd=reserve, remaining_usd=remaining)
        return
    status("fresh_benchmark", questions=200, cumulative_batch_cap_usd=remaining)
    run_batch("fresh", remaining)
    report_fresh(items)
    result = read_json(ROOT / "fresh.json")
    result["total_experiment_usage"] = spending()
    write_json(ROOT / "fresh.json", result)
    write_json(ROOT / "refinement-complete.json", result)
    status("complete", report=str(ROOT / "refinement-complete.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credit", type=float, required=True,
                        help="Opening experiment allowance, including the existing $0.53 pilot spend")
    args = parser.parse_args()
    if not math.isfinite(args.credit) or args.credit <= 0:
        parser.error("Supply a positive allowance.")
    with exclusive_run():
        try:
            run(min(8., args.credit))
        except Exception as error:
            status("failed", error_type=type(error).__name__, message=str(error))
            raise
