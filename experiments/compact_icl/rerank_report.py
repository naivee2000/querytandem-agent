"""Export selector results without prompts, SQL, credentials, or database files."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .rerank_data import ROOT
from .rerank_experiment import FEATURES, fit, usage
from .routing import read_json, request_rows, write_json

OUTPUT = Path("experiments/compact_icl/results")


def verify(output):
    import numpy as np

    features = read_json(output / "selector-training-features.json")
    model = read_json(output / "selector-ranker.json")
    if features["features"] != list(FEATURES) or model["features"] != list(FEATURES):
        raise ValueError("Feature definitions do not match this code.")
    rows = features["questions"]
    if len({(row["database_id"], row["item_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate training question.")
    weights = fit([(row["good"], row["bad"]) for row in rows])
    if weights is None or not np.allclose(weights, model["weights"], rtol=1e-9, atol=1e-9):
        raise ValueError("Refitted coefficients differ from the published model.")
    result = read_json(output / "selector-validation.json")
    informative = sum(bool(row["good"] and row["bad"]) for row in rows)
    if (len(rows), informative) != (result["training"]["questions"], result["training"]["informative"]):
        raise ValueError("Training counts differ from the published report.")
    print(f"Verified {len(rows)} training questions, {informative} informative questions, and all fitted weights.")


def export(output):
    training = read_json(ROOT / "refinement-training.json")
    completed = read_json(ROOT / "refinement-complete.json")
    plan = read_json(ROOT / "refinement-plan.json")
    model = read_json(ROOT / "ranker.json")
    runtime = Path("app/pipeline/sql_selection/reranker.py")
    if model["runtime_sha256"] != hashlib.sha256(runtime.read_bytes()).hexdigest():
        raise ValueError("The selector runtime changed since validation.")

    questions, batches, samples = [], [], []
    for phase in training["phases"]:
        manifest = read_json(ROOT / phase / "dev/dev.json")
        for row in manifest:
            record = read_json(ROOT / "offline-labels" / phase / "augmented" / f'{row["question_id"]}.json')
            questions.append(dict(item_id=str(row["question_id"]), database_id=row["db_id"],
                                  good=record["good"], bad=record["bad"]))
        batches.append(dict(phase=phase, **read_json(ROOT / f"{phase}-refinement.json"),
                            pipeline_usage=usage(request_rows(ROOT / f"{phase}-usage.jsonl"))))
        samples.extend(read_json(path) for path in sorted((ROOT / "augmentation" / phase / "samples").glob("*.json")))

    rows = [row for path in sorted(ROOT.glob("*-usage.jsonl")) for row in request_rows(path)]
    actual_usage = usage(rows)
    if actual_usage != completed["usage"]:
        raise ValueError("Usage changed after the completed experiment.")
    result = dict(
        experiment="QueryTandem optional SQL selector", recommendation=completed["recommendation"],
        reason=completed["reason"], fresh_benchmark_run=(ROOT / "fresh-usage.jsonl").exists(),
        training={key: training[key] for key in ("questions", "informative", "candidate_execution_statuses",
                                                 "gold_execution_statuses", "augmentation")},
        validation=completed["validation"], batches=batches, experiment_usage=actual_usage,
        usage_by_provider={provider: usage([row for row in rows if row["provider"] == provider])
                           for provider in ("together", "local")},
        local_augmentation=dict(samples=len(samples), statuses=dict(sorted(Counter(row["status"] for row in samples).items())),
                                peak_memory_gb=max(row.get("peak_memory_gb", 0) for row in samples),
                                summed_generation_seconds=sum(row.get("seconds", 0) for row in samples)),
        generation=training["generation"],
        execution_limits=dict(offline_per_query_seconds=30, runtime_shared_seconds=3, max_rows=100000),
        validation_usage_note="Cached validation pipeline usage is historical and is not added to this experiment's spending.",
        source_sha256={str(Path(key).relative_to(Path.cwd())) if Path(key).is_absolute() else key: value
                       for key, value in plan["source_sha256"].items()},
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selector-validation.json", result)
    write_json(output / "selector-training-features.json", dict(features=list(FEATURES), questions=questions))
    write_json(output / "selector-ranker.json", model)
    verify(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true", help="Refit published features; no dataset or API required")
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        export(args.output)
