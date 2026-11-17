"""Generate ten SQL responses to check that a local adapter loads and runs."""
import argparse
import json
from pathlib import Path
import time

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    for index, line in enumerate(args.data.read_text().splitlines()[:10]):
        messages = json.loads(line)["messages"][:-1]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        start = time.monotonic()
        text = ""
        for response in stream_generate(model, tokenizer, prompt, max_tokens=256, sampler=make_sampler(temp=0)):
            text += response.text
        print(json.dumps({
            "index": index,
            "response": text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.generation_tokens,
            "seconds": time.monotonic() - start,
            "format_ok": text.strip().startswith("<result>") and text.strip().endswith("</result>"),
        }), flush=True)


if __name__ == "__main__":
    main()
