"""Optional per-request experiment ledger and conservative API spend cap."""
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import json
import os
from pathlib import Path
import time


class BudgetExceeded(RuntimeError):
    pass


_item_id = ContextVar("usage_item_id", default=None)


def run_for_item(function, item):
    token = _item_id.set(item.get_item_id())
    try:
        return function(item)
    finally:
        _item_id.reset(token)


def submit_with_usage_context(executor, function, *args, **kwargs):
    return executor.submit(copy_context().run, function, *args, **kwargs)


@contextmanager
def locked_ledger(path):
    import fcntl

    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        yield handle
        handle.flush()


def recorded_completion(client, params, config):
    ledger_path = os.environ.get("DEEPEYE_USAGE_LEDGER")
    if not ledger_path:
        return client.chat.completions.create(**params)

    path = Path(ledger_path)
    provider = "together" if "together." in config.base_url else "local"
    reserve = 0
    if provider == "together":
        # Reserve the provider's full context allowance. Values are microdollars.
        if config.model != "Qwen/Qwen3.5-9B":
            raise BudgetExceeded("The experiment budget has no price for this model.")
        reserve = 44565 + params["n"] * params["max_tokens"] * 0.25

    with locked_ledger(path) as handle:
        rows = [json.loads(line) for line in handle]
        charges = {}
        for row in rows:
            if row.get("provider") != "together":
                continue
            if "reserve" in row:
                charges[row["request"]] = row["reserve"]
            elif row.get("prompt_tokens") is not None:
                charges[row["request"]] = row["prompt_tokens"] * 0.17 + row["completion_tokens"] * 0.25
        reserved = sum(charges.values())
        cap = int(float(os.environ["DEEPEYE_API_CAP_USD"]) * 1_000_000)
        if reserved + reserve > cap:
            Path(f"{path}.cap-reached").touch()
            raise BudgetExceeded("The reserved API spend reached the experiment cap.")
        request_id = len(rows)
        handle.write(json.dumps({
            "request": request_id, "provider": provider, "reserve": reserve,
            "stage": os.environ.get("DEEPEYE_STAGE"), "item_id": _item_id.get(),
        }) + "\n")

    started = time.monotonic()
    response = None
    try:
        response = client.with_options(max_retries=0).chat.completions.create(**params)
        return response
    finally:
        usage = response.usage if response is not None else None
        record = {
            "request": request_id,
            "provider": provider,
            "item_id": _item_id.get(),
            "stage": os.environ.get("DEEPEYE_STAGE"),
            "seconds": time.monotonic() - started,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "ok": response is not None,
        }
        with locked_ledger(path) as handle:
            handle.write(json.dumps(record) + "\n")
