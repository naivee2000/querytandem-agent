"""Select an adapter and prompt size from completed held-out validation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    groups = defaultdict(dict)
    for line in args.validation.read_text().splitlines():
        row = json.loads(line)
        key = row["adapter"], row["examples"]
        if row["index"] in groups[key]:
            raise ValueError(f"Duplicate validation record for {key}")
        groups[key][row["index"]] = row
    if len(groups) != 10 or any(len(group) != 100 for group in groups.values()):
        parser.error("Validation must have all 1,000 results before selecting a checkpoint.")
    expected = set(groups["base", 0])
    if any(set(group) != expected for group in groups.values()):
        parser.error("Validation question IDs differ between variants.")
    summaries = []
    for (adapter, examples), rows in groups.items():
        summaries.append({
            "adapter": adapter,
            "examples": examples,
            "correct": sum(row["correct"] for row in rows.values()),
            "parsed": sum(row["parsed"] for row in rows.values()),
            "tokens": sum(row["prompt_tokens"] + row["completion_tokens"] for row in rows.values()),
        })
    selected = min(
        (row for row in summaries if row["adapter"] != "base"),
        key=lambda row: (-row["correct"], row["tokens"], row["examples"], row["adapter"]),
    )
    print(json.dumps({"selected": selected, "variants": summaries}, indent=2))


if __name__ == "__main__":
    main()
