"""Freeze independent BIRD splits and prepare configs for the reranker runs."""

import json
import os
from pathlib import Path
import re

import tomlkit

from .routing import ORIGINAL, identity, read_json, stratify, write_json

EXPERIMENT = "sql-reranker"
ROOT = Path("workspace") / EXPERIMENT
INDEX = Path("workspace/runs/_shared/bird/train_full_index/examples.jsonl")
COUNTS = {"pilot": 40, "train": 360, "fresh": 200}


def normalized_question(row):
    return " ".join(row["question"].casefold().split())


def split(phase):
    if re.fullmatch(r"refine-0[1-9]", phase):
        rows, source = split("train")
        start = (int(phase[-2:]) - 1) * 40
        return rows[start:start + 40], source
    if phase not in COUNTS:
        raise ValueError(f"Unknown reranker phase: {phase}")
    manifest = read_json(ORIGINAL / "bird/manifest.json")
    dev = read_json(Path("data/bird/dev/dev.json"))
    pilot_dev = read_json(ORIGINAL / "pilot-benchmark/dev/dev.json")
    if phase == "fresh":
        old = {identity(dev[index]) for index in manifest["dev_indices"]}
        old.update(map(identity, pilot_dev))
        eligible = [(index, row) for index, row in enumerate(dev) if identity(row) not in old]
        return stratify(eligible, COUNTS[phase], lambda entry: (
            entry[1]["db_id"], entry[1].get("difficulty") or ""
        )), Path("data/bird/dev/dev_databases")

    train = [json.loads(line) for line in Path("data/bird/downloads/train.jsonl").read_text().splitlines()]
    validation = [train[entry["index"]] for entry in manifest["valid"]]
    excluded_questions = set(map(normalized_question, (*dev, *validation, *pilot_dev)))
    excluded_databases = {row["db_id"] for row in (*dev, *validation)}
    indexed = [json.loads(line) for line in INDEX.read_text().splitlines()]
    excluded_questions.update(map(normalized_question, indexed))
    indexed_ids = {row["example_id"] for row in indexed}
    source = Path("data/bird/train/train_databases")
    eligible = []
    for index, row in enumerate(train):
        database = row["db_id"]
        if (database in excluded_databases or f"bird:{index}" in indexed_ids
                or normalized_question(row) in excluded_questions
                or not (source / database / f"{database}.sqlite").is_file()):
            continue
        eligible.append((index, row))
        excluded_questions.add(normalized_question(row))
    selected = stratify(eligible, 400, lambda entry: entry[1]["db_id"])
    return (selected[:40] if phase == "pilot" else selected[40:]), source


def prepare(phase):
    rows, source = split(phase)
    destination = ROOT / phase / "dev"
    destination.mkdir(parents=True, exist_ok=True)
    dataset = destination / "dev.json"
    records = [dict(row, question_id=index, evidence=row.get("evidence") or "",
                    difficulty=row.get("difficulty") or "") for index, row in rows]
    if dataset.exists() and read_json(dataset) != records:
        raise ValueError("The frozen reranker split changed.")
    write_json(dataset, records)
    databases = destination / "dev_databases"
    if not databases.is_symlink():
        databases.symlink_to(source.resolve(), target_is_directory=True)
    if databases.resolve() != source.resolve() or any(
        not (databases / row["db_id"] / f'{row["db_id"]}.sqlite').is_file() for row in records
    ):
        raise ValueError("A reranker database is missing or the database path changed.")

    config = tomlkit.parse(Path(__file__).with_name("bird.toml").read_text())
    config["run"]["exp_name"] = f"{EXPERIMENT}-{phase}"
    config["run"]["checkpoint_interval"] = 5
    config["dataset"]["root_path"] = str(destination.parent)
    if phase == "train" or phase.startswith("refine-"):
        config["vector_database"]["store_root_path"] = "workspace/runs/sql-reranker-pilot/vector_database"
    config["sql_generation"]["icl_prompt_mode"] = "compact"
    config["sql_generation"]["icl_example_count"] = 0
    config["llm_profiles"]["local_icl"]["base_url"] = "http://127.0.0.1:8081/v1"
    local = Path("config/local/compact-icl")
    key = os.environ.get("TOGETHER_API_KEY") or (local / "together.key").read_text().strip()
    config["llm_profiles"]["qwen3_coder_instruct"]["api_key"] = key
    target = local / f"{EXPERIMENT}-{phase}.toml"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(tomlkit.dumps(config))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=COUNTS)
    prepare(parser.parse_args().phase)
