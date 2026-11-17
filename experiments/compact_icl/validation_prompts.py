"""Add paired validation prompts to an existing data split."""
import argparse
import json
from pathlib import Path

from app.prompt import PromptFactory
from app.services import get_schema_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    path = args.data / "manifest.json"
    manifest = json.loads(path.read_text())
    labels = [json.loads(line) for line in Path("data/bird/downloads/train.jsonl").read_text().splitlines()]
    examples = [labels[item["index"]] for item in manifest["train"]]
    schemas = get_schema_service()
    for item in manifest["valid"]:
        if "prompts" in item:
            continue
        row = labels[item["index"]]
        db_id = row["db_id"]
        db = Path("data/bird/train/train_databases") / db_id / f"{db_id}.sqlite"
        schema = schemas.load_sqlite_schema(str(db))
        profile = schemas.build_schema_profile(
            schema, include_value_examples=False, include_value_statistics=False
        )
        example = next(example for example in examples if example["db_id"] != db_id)
        one = [dict(question=example["question"], evidence=example["evidence"], sql=example["SQL"])]
        item["prompts"] = [
            PromptFactory.format_compact_icl_prompt(shots, profile, row["question"], row["evidence"], "SQLite")
            for shots in ([], one)
        ]
    pending = path.with_suffix(".json.part")
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.replace(path)


if __name__ == "__main__":
    main()
