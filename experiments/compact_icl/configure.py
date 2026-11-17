"""Write four local DeepEye configs without committing API credentials."""
import argparse
import os
from pathlib import Path

import tomlkit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, type=int, choices=(0, 1))
    args = parser.parse_args()
    destination = Path("config/local/compact-icl")
    key_path = destination / "together.key"
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key and key_path.is_file():
        api_key = key_path.read_text().strip()
    if not api_key:
        parser.error("Set TOGETHER_API_KEY or save the key in config/local/compact-icl/together.key.")
    template = Path(__file__).with_name("bird.toml").read_text()
    destination.mkdir(parents=True, exist_ok=True)
    for name, tuned, compact in (
        ("baseline", False, False),
        ("prompt-only", False, True),
        ("training-only", True, False),
        ("tuned-compact", True, True),
    ):
        config = tomlkit.parse(template)
        config["run"]["exp_name"] = f"compact-icl-{name}"
        profiles = config["llm_profiles"]
        profiles[config["run"]["default_llm_profile"]]["api_key"] = api_key
        profiles["local_icl"]["base_url"] = f"http://127.0.0.1:{8081 if tuned else 8080}/v1"
        generation = config["sql_generation"]
        generation["icl_prompt_mode"] = "compact" if compact else "upstream"
        generation["icl_example_count"] = args.examples if compact else 7
        target = destination / f"{name}.toml"
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(tomlkit.dumps(config))


if __name__ == "__main__":
    main()
