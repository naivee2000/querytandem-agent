"""Run the budget-capped, paired SQL reranker experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.dataset import load_dataset
from app.pipeline.sql_selection.reranker import FEATURES, SQLReranker, group_answers, majority_vote, read_answer
from .rerank_data import COUNTS, EXPERIMENT, ROOT, prepare
from .routing import cost, read_json, request_rows, write_json

THRESHOLDS = (0.60, 0.75, 0.90)
MODEL = ROOT / "ranker.json"
RUNTIME_SOURCE = Path("app/pipeline/sql_selection/reranker.py")
PIPELINE = Path("experiments/compact_icl/run_routing.sh")


def snapshot(phase, experiment=EXPERIMENT):
    return Path("workspace/runs") / f"{experiment}-{phase}/sql_selection.snapshot"


def batch_items(phase):
    items = list(load_dataset(str(snapshot(phase))))
    expected = read_json(ROOT / phase / "dev/dev.json")
    expected_ids = {(str(row["question_id"]), row["db_id"]) for row in expected}
    if len(items) != len(expected) or {(item.get_item_id(), item.database_id) for item in items} != expected_ids:
        raise ValueError(f"{phase} snapshot differs from the frozen questions.")
    if not all(item.is_stage_complete("sql_selection") for item in items):
        raise ValueError(f"{phase} selection checkpoint is incomplete.")
    return items


def snapshot_complete(path, stage):
    if not Path(path).is_file():
        return False
    items = list(load_dataset(str(path)))
    return bool(items) and all(item.is_stage_complete(stage) for item in items)


def ledger(phase, root=ROOT):
    path = root / f"{phase}-usage.jsonl"
    return request_rows(path) if path.exists() else []


def usage(rows):
    return {
        "requests": len(rows),
        "tokens": sum((row.get("prompt_tokens") or 0) + (row.get("completion_tokens") or 0) for row in rows),
        "unknown_usage_requests": sum(row.get("prompt_tokens") is None for row in rows),
        "reported_api_usd": sum(cost(row) for row in rows if row.get("prompt_tokens") is not None),
        "api_usd_with_reservations": sum(map(cost, rows)),
    }


def run_batch(phase, cap):
    prepare(phase)
    if snapshot_complete(snapshot(phase), "sql_selection"):
        return
    if cap <= usage(ledger(phase))["api_usd_with_reservations"]:
        raise ValueError(f"The {phase} request cap leaves no room to resume.")
    with (ROOT / f"{phase}.log").open("a") as log:
        subprocess.run(
            ["bash", str(PIPELINE), "pipeline", str(cap), EXPERIMENT, phase],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )


def labeled(item, *, diagnostics=None):
    executions = []
    groups = group_answers(item, timeout=30, per_query_timeout=True, diagnostics=executions)
    gold = {}
    expected = read_answer(item.database_path, item.gold_sql, time.monotonic() + 30, diagnostics=gold)
    if diagnostics is not None:
        diagnostics.update(item_id=item.get_item_id(), database_id=item.database_id,
                           candidates=executions, gold=gold, groups=len(groups))
    if expected is None:
        return [], []
    good = [group.features for group in groups if group.answer.rows == expected.rows]
    bad = [group.features for group in groups if group.answer.rows != expected.rows]
    return good, bad


def collect_labels(items):
    with ThreadPoolExecutor(max_workers=4) as workers:
        return list(workers.map(labeled, items))


def fit(examples):
    contrasts = [(good, bad) for good, bad in examples if good and bad]
    if len(contrasts) < 60:
        return None
    inputs, labels, sample_weights = [], [], []
    for good, bad in contrasts:
        weight = 1 / (2 * len(good) * len(bad))
        for correct in good:
            for incorrect in bad:
                difference = np.asarray(correct) - np.asarray(incorrect)
                inputs.extend((difference, -difference))
                labels.extend((1, 0))
                sample_weights.extend((weight, weight))
    classifier = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000, random_state=42)
    classifier.fit(inputs, labels, sample_weight=sample_weights)
    return tuple(classifier.coef_[0])


def evaluate(items, rankers, requests):
    def evaluate_item(item):
        incumbent = item.final_selected_sql
        started = time.perf_counter()
        groups = group_answers(item)
        group_ms = (time.perf_counter() - started) * 1000
        choices = {"current": incumbent, "majority": majority_vote(groups, incumbent)}
        timings = {}
        for label, ranker in rankers.items():
            started = time.perf_counter()
            choices[label] = ranker.choose(groups, incumbent)
            timings[label] = group_ms + (time.perf_counter() - started) * 1000

        expected = read_answer(item.database_path, item.gold_sql, time.monotonic() + 30)
        outputs = {sql: group.answer for group in groups for sql in group.sqls}
        if incumbent not in outputs:
            outputs[incumbent] = read_answer(item.database_path, incumbent, time.monotonic() + 30)
        scores = {
            label: int(expected is not None and outputs[sql] is not None and outputs[sql].rows == expected.rows)
            for label, sql in choices.items()
        }
        return item.get_item_id(), item.database_id, scores, choices, timings

    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(evaluate_item, items))
    baseline = np.array([row[2]["current"] for row in results])
    dbs = sorted({row[1] for row in results})
    groups = [np.array([index for index, row in enumerate(results) if row[1] == db]) for db in dbs]
    rng = np.random.default_rng(42)
    variants = {}
    for label in ("current", "majority", *rankers):
        scores = np.array([row[2][label] for row in results])
        delta = scores - baseline
        bootstraps = np.array([delta[group].sum() for group in groups])
        sample = rng.integers(len(groups), size=(5000, len(groups)))
        means = bootstraps[sample].sum(axis=1) / np.array([len(g) for g in groups])[sample].sum(axis=1)
        variants[label] = {
            "correct": int(scores.sum()), "accuracy": float(scores.mean()),
            "wins": [row[0] for row, diff in zip(results, delta) if diff > 0],
            "losses": [row[0] for row, diff in zip(results, delta) if diff < 0],
            "overrides": sum(row[3][label] != row[3]["current"] for row in results),
            "delta_ci95": np.quantile(means, (0.025, 0.975)).tolist(),
        }
        if label in rankers:
            median, p95 = np.quantile([row[4][label] for row in results], (0.5, 0.95))
            variants[label].update({"added_median_ms": float(median), "added_p95_ms": float(p95)})
    return {"questions": len(items), "variants": variants, "shared_model_usage": usage(requests),
            "extra_model_tokens": 0, "extra_model_calls": 0}


def report_validation(weights, train_items, informative, *, metadata=None):
    cached = ROOT.with_name("adaptive-icl")
    items = [item for phase in ("pilot", "validation") for item in load_dataset(str(snapshot(phase, "adaptive-icl")))]
    if len(items) != 100 or len({item.get_item_id() for item in items}) != 100:
        raise ValueError("Expected the 100 cached validation questions.")
    if {item.database_id for item in items} & {item.database_id for item in train_items}:
        raise ValueError("Training and validation databases overlap.")
    report = evaluate(items, {str(t): SQLReranker(weights, t) for t in THRESHOLDS},
                      ledger("pilot", cached) + ledger("validation", cached))
    scores = report["variants"]
    if scores["current"]["correct"] != 77:
        raise ValueError("Cached baseline did not reproduce the reported 77 correct answers.")
    threshold = min(THRESHOLDS, key=lambda t: (
        -scores[str(t)]["correct"], scores[str(t)]["overrides"], -t
    ))
    qualifies = scores[str(threshold)]["correct"] >= 79 and scores[str(threshold)]["correct"] > scores["majority"]["correct"]
    report.update({"informative_training_questions": informative, "threshold": threshold,
                   "qualified": qualifies})
    write_json(ROOT / "validation.json", report)
    write_json(MODEL, {
        "version": 1, "features": FEATURES, "weights": weights,
        "threshold": threshold, "qualified": qualifies,
        "runtime_sha256": hashlib.sha256(RUNTIME_SOURCE.read_bytes()).hexdigest(),
        **(metadata or {}),
    })
    return qualifies


def report_fresh(train_items):
    artifact = read_json(MODEL)
    if not artifact["qualified"] or artifact["runtime_sha256"] != hashlib.sha256(RUNTIME_SOURCE.read_bytes()).hexdigest():
        raise ValueError("The qualified reranker must stay frozen for the fresh benchmark.")
    items = batch_items("fresh")
    if {item.database_id for item in train_items} & {item.database_id for item in items}:
        raise ValueError("Training and fresh evaluation databases overlap.")
    report = evaluate(items, {"ranker": SQLReranker.load(MODEL)}, ledger("fresh"))
    ranker = report["variants"]["ranker"]
    difference = ranker["correct"] - report["variants"]["current"]["correct"]
    report["meets_target"] = difference >= 4
    report["recommendation"] = (
        "current" if not report["meets_target"] else
        "opt-in" if ranker["delta_ci95"][0] <= 0 else "ranker"
    )
    report["total_experiment_usage"] = usage([row for phase in COUNTS for row in ledger(phase)])
    write_json(ROOT / "fresh.json", report)


@contextmanager
def exclusive_run():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def wait_for_pilot(pilot_pid):
    while not snapshot("pilot").is_file():
        try:
            os.kill(pilot_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(15)


def run(cap, pilot_pid):
    if pilot_pid:
        wait_for_pilot(pilot_pid)
    run_batch("pilot", min(1.0, cap))
    pilot_items = batch_items("pilot")
    pilot_examples = collect_labels(pilot_items)
    pilot_informative = sum(bool(good and bad) for good, bad in pilot_examples)
    spending = usage(ledger("pilot"))
    if spending["requests"] == 0 or spending["reported_api_usd"] <= 0:
        raise ValueError("Pilot has no recorded paid usage.")
    estimate_per_question = 1.2 * spending["api_usd_with_reservations"] / len(pilot_items)
    train_budget = cap - spending["api_usd_with_reservations"] - estimate_per_question * COUNTS["fresh"]
    feasible = train_budget >= estimate_per_question * COUNTS["train"]
    write_json(ROOT / "pilot.json", {
        "questions": len(pilot_items), "informative": pilot_informative,
        "usage": spending,
        "projected_total_usd": spending["api_usd_with_reservations"] + estimate_per_question * (COUNTS["train"] + COUNTS["fresh"]),
        "within_budget": feasible,
    })
    print(f"Pilot produced {pilot_informative}/{len(pilot_items)} informative questions.", flush=True)
    if not pilot_informative or not feasible:
        print("Stopped at the pilot gate. See pilot.json.", flush=True)
        return

    run_batch("train", train_budget)
    train_items = pilot_items + batch_items("train")
    if len({item.get_item_id() for item in train_items}) != 400:
        raise ValueError("Pilot and training question IDs overlap.")
    examples = pilot_examples + collect_labels(train_items[len(pilot_items):])
    informative = sum(bool(good and bad) for good, bad in examples)
    write_json(ROOT / "training.json", {"questions": len(examples), "informative": informative})
    weights = fit(examples)
    if weights is None:
        print(f"Stopped with {informative}/60 informative training questions.", flush=True)
        return
    if snapshot("fresh").exists():
        if (ROOT / "fresh.json").exists():
            print("Fresh benchmark already completed.", flush=True)
        else:
            report_fresh(train_items)
        return
    if MODEL.exists():
        qualified = read_json(MODEL)["qualified"]
    else:
        qualified = report_validation(weights, train_items, informative)
    if not qualified:
        print("Current selector retained. The ranker did not qualify on validation.", flush=True)
        return
    remaining = cap - usage(ledger("pilot") + ledger("train"))["api_usd_with_reservations"]
    if estimate_per_question * COUNTS["fresh"] > remaining:
        print("Stopped at the fresh evaluation budget gate.", flush=True)
        return
    if read_json(MODEL)["runtime_sha256"] != hashlib.sha256(RUNTIME_SOURCE.read_bytes()).hexdigest():
        raise ValueError("The reranker changed after validation.")
    run_batch("fresh", remaining)
    report_fresh(train_items)
    print("Fresh benchmark completed. See fresh.json.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credit", type=float, required=True, help="Verified opening prepaid credit in USD")
    parser.add_argument("--wait-for-pilot", type=int, help="PID of an existing pilot launcher")
    args = parser.parse_args()
    if not math.isfinite(args.credit) or args.credit <= 0:
        parser.error("Provide a positive verified credit amount.")
    with exclusive_run():
        run(min(8.0, args.credit), args.wait_for_pilot)
