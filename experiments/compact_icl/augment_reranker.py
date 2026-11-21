"""Generate training-only candidates in MLX; input contains no gold answers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time


def write_json(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def parse_sql(text):
    match = re.search(r"<result>(.*?)</result>", text, re.DOTALL)
    if not match:
        return None
    sql = match.group(1).strip()
    if sql.startswith("```sql") and sql.endswith("```"):
        sql = sql[6:-3].strip()
    return sql or None


def load_inputs(path):
    manifest = json.loads(path.read_text())
    for row in manifest["items"]:
        if set(row) != {"item_id", "database_id", "prompts"}:
            raise ValueError("Generation accepts only identifiers and inference prompts.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_inputs(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    specification = manifest["generation"]
    fingerprint = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    tasks = [(row, sample) for row in manifest["items"] for sample in range(specification["samples"])]
    pending = []
    for row, sample in tasks:
        path = args.output / f'{row["item_id"]}-{sample}.json'
        if path.exists():
            if json.loads(path.read_text())["manifest_sha256"] != fingerprint:
                raise ValueError("Cached generation belongs to another experiment.")
        else:
            pending.append((row, sample, path))
    if not pending:
        return

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(specification["model"], adapter_path=specification["adapter"])
    completed = len(tasks) - len(pending)
    for row, sample, path in pending:
        tokens = None
        for level, content in enumerate(row["prompts"]):
            candidate = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=False,
            )
            if len(candidate) <= specification["context_tokens"] - specification["max_tokens"]:
                tokens = candidate
                break
        record = dict(item_id=row["item_id"], database_id=row["database_id"], sample=sample,
                      manifest_sha256=fingerprint, sql=None, prompt_tokens=None, completion_tokens=0)
        if tokens is None:
            record.update(status="prompt_too_long", seconds=0.)
        else:
            seed = (42 + int(row["item_id"]) * specification["samples"] + sample) % (2**32)
            mx.random.seed(seed)
            started = time.monotonic()
            request_id = time.time_ns()
            with args.ledger.open("a") as ledger:
                ledger.write(json.dumps(dict(request=request_id, provider="local", reserve=0,
                                             stage="reranker_training_augmentation", item_id=row["item_id"])) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
            text = ""
            response = None
            try:
                for response in stream_generate(
                    model, tokenizer, tokens, max_tokens=specification["max_tokens"],
                    sampler=make_sampler(temp=specification["temperature"], top_p=specification["top_p"]),
                    prefill_step_size=512,
                ):
                    text += response.text
                sql = parse_sql(text)
                record.update(status="ok" if sql else "parse_error", sql=sql, response=text, seed=seed,
                              prompt_level=level, prompt_tokens=len(tokens),
                              completion_tokens=response.generation_tokens, finish_reason=response.finish_reason,
                              peak_memory_gb=response.peak_memory, seconds=time.monotonic() - started)
            finally:
                with args.ledger.open("a") as ledger:
                    ledger.write(json.dumps(dict(request=request_id, provider="local",
                                                 stage="reranker_training_augmentation", item_id=row["item_id"],
                                                 prompt_tokens=len(tokens),
                                                 completion_tokens=response.generation_tokens if response else None,
                                                 seconds=time.monotonic() - started, ok="status" in record)) + "\n")
                    ledger.flush()
                    os.fsync(ledger.fileno())
        write_json(path, record)
        completed += 1
        progress = dict(completed=completed, total=len(tasks), item_id=row["item_id"],
                        status=record["status"], updated_at=time.time())
        write_json(args.output.parent / "progress.json", progress)
        print(f'Local candidates {completed}/{len(tasks)}: question {row["item_id"]}, {record["status"]}', flush=True)
        mx.clear_cache()


if __name__ == "__main__":
    main()
