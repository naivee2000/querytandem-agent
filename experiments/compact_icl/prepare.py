"""Prepare database-disjoint BIRD data for local MLX fine-tuning."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sqlite3
import time

from transformers import AutoTokenizer

from app.prompt import PromptFactory
from app.services import get_schema_service


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def executable(db, sql):
    deadline = time.monotonic() + 3
    try:
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            conn.execute(sql).fetchone()
    except sqlite3.Error:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the downloaded model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--valid-count", type=int, default=100)
    parser.add_argument("--dev-count", type=int, default=200)
    args = parser.parse_args()
    rng = random.Random(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    root = Path("data/bird")
    records = [json.loads(line) for line in (root / "downloads/train.jsonl").read_text().splitlines()]
    dev = json.loads((root / "dev/dev.json").read_text())
    dev_databases = {row["db_id"] for row in dev}
    dbs = {
        path.stem: path
        for path in (root / "train/train_databases").glob("*/*.sqlite")
        if path.stem not in dev_databases
    }

    def choose(rows, target, key):
        groups = defaultdict(list)
        for row in rows:
            groups[key(row)].append(row)
        keys = sorted(groups)
        rng.shuffle(keys)
        for values in groups.values():
            rng.shuffle(values)
        count = 0
        while keys and count < target:
            for group in keys[:]:
                yield groups[group].pop()
                count += 1
                if not groups[group]:
                    keys.remove(group)
                if count == target:
                    return

    eligible = [(i, row) for i, row in enumerate(records) if row["db_id"] in dbs]
    shuffled_dbs = sorted({row["db_id"] for _, row in eligible})
    rng.shuffle(shuffled_dbs)
    validation_dbs = set(shuffled_dbs[:max(1, len(shuffled_dbs) // 5)])
    training = [(i, row) for i, row in eligible if row["db_id"] not in validation_dbs]
    validation = [(i, row) for i, row in eligible if row["db_id"] in validation_dbs]
    examples = training[:]
    rng.shuffle(examples)

    def prepare(pool, limit, validation_modes=False):
        rows, manifest, rejected = [], [], defaultdict(int)
        for index, row in choose(pool, len(pool), lambda entry: entry[1]["db_id"]):
            db_id = row["db_id"]
            gold = row["SQL"].strip()
            shot_count = len(rows) % 2
            shots = []
            if shot_count:
                _, example = next(entry for entry in examples if entry[1]["db_id"] != db_id)
                shots = [dict(question=example["question"], evidence=example["evidence"], sql=example["SQL"])]
            schema = get_schema_service().load_sqlite_schema(str(dbs[db_id]))
            profile = get_schema_service().build_schema_profile(
                schema, include_value_examples=False, include_value_statistics=False
            )
            prompt = PromptFactory.format_compact_icl_prompt(
                shots, profile, row["question"], row["evidence"], "SQLite"
            )
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"<result>{gold}</result>"},
            ]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
            if len(tokens) > args.max_tokens:
                rejected["too_long"] += 1
                continue
            if not executable(dbs[db_id], gold):
                rejected["sql_error"] += 1
                continue
            rows.append({"messages": messages})
            entry = {"index": index, "db_id": db_id, "examples": shot_count, "tokens": len(tokens)}
            if validation_modes:
                _, example = next(item for item in examples if item[1]["db_id"] != db_id)
                one = [dict(question=example["question"], evidence=example["evidence"], sql=example["SQL"])]
                entry["prompts"] = [
                    PromptFactory.format_compact_icl_prompt(
                        shots, profile, row["question"], row["evidence"], "SQLite"
                    ) for shots in ([], one)
                ]
            manifest.append(entry)
            if len(rows) % 50 == 0:
                print(f"Prepared {len(rows)}/{limit} examples", flush=True)
            if len(rows) == limit:
                break
        if len(rows) < limit:
            raise ValueError(f"Only {len(rows)} of {limit} examples available: {dict(rejected)}")
        return rows, manifest, dict(rejected)

    train, train_manifest, train_rejected = prepare(training, args.train_count)
    valid, valid_manifest, valid_rejected = prepare(validation, args.valid_count, validation_modes=True)
    chosen = sorted(
        index for index, _ in choose(
            list(enumerate(dev)), args.dev_count,
            lambda entry: (entry[1]["db_id"], entry[1].get("difficulty") or ""),
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "valid.jsonl", valid)
    benchmark = args.output / "benchmark"
    (benchmark / "train").mkdir(parents=True, exist_ok=True)
    (benchmark / "dev").mkdir(parents=True, exist_ok=True)
    (benchmark / "train/train.json").write_text(json.dumps([
        dict(records[item["index"]], question_id=item["index"]) for item in train_manifest
    ]))
    (benchmark / "dev/dev.json").write_text(json.dumps([
        dict(dev[index], question_id=index) for index in chosen
    ]))
    databases = benchmark / "dev/dev_databases"
    if not databases.is_symlink():
        databases.symlink_to((root / "dev/dev_databases").resolve(), target_is_directory=True)
    pending_manifest = args.output / "manifest.json.part"
    pending_manifest.write_text(json.dumps({
        "seed": 42,
        "train": train_manifest,
        "valid": valid_manifest,
        "dev_indices": chosen,
        "rejected": {"train": train_rejected, "valid": valid_rejected},
    }, indent=2) + "\n")
    pending_manifest.replace(args.output / "manifest.json")


if __name__ == "__main__":
    main()
