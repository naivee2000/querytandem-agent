"""Compare base and saved adapters on both compact prompts on validation databases."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import time

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def result(db, sql):
    deadline = time.monotonic() + 3
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        return Counter(conn.execute(sql).fetchall())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    labels = [json.loads(line) for line in Path("data/bird/downloads/train.jsonl").read_text().splitlines()]
    for adapter in (None, *args.adapter):
        model, tokenizer = load(args.model, adapter_path=str(adapter) if adapter else None)
        for item in manifest["valid"][:args.limit]:
            row = labels[item["index"]]
            db = Path("data/bird/train/train_databases") / row["db_id"] / f'{row["db_id"]}.sqlite'
            try:
                expected = result(db, row["SQL"])
            except sqlite3.Error:
                expected = None
            for count, content in enumerate(item["prompts"]):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}], add_generation_prompt=True
                )
                start = time.monotonic()
                output = ""
                for response in stream_generate(model, tokenizer, prompt, max_tokens=512, sampler=make_sampler(temp=0)):
                    output += response.text
                match = re.search(r"<result>(.*?)</result>", output, re.DOTALL)
                sql = match.group(1).strip() if match else None
                if sql and sql.startswith("```sql") and sql.endswith("```"):
                    sql = sql[6:-3].strip()
                try:
                    correct = bool(sql and expected is not None and result(db, sql) == expected)
                except sqlite3.Error:
                    correct = False
                print(json.dumps({
                    "adapter": str(adapter) if adapter else "base",
                    "index": item["index"],
                    "examples": count,
                    "correct": correct,
                    "parsed": match is not None,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.generation_tokens,
                    "seconds": time.monotonic() - start,
                }), flush=True)
        del model, tokenizer


if __name__ == "__main__":
    main()
