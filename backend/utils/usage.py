"""
Token-usage and cost tracking for OpenAI calls.

Wrap a run in ``track_usage()``; every ``record_usage()`` made in that context
(including inside Haystack tool threads, which copy the caller's context) is
collected, and ``summarize()`` turns the records into totals and an estimated cost.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from logger import get_logger

_log = get_logger("utils.usage")

# USD per 1M tokens (input, output), from developers.openai.com/api/docs/models.
# Cached-input discounts are ignored, so estimates err on the high side.
PRICES_PER_MILLION = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
}

_records: ContextVar[list | None] = ContextVar("usage_records", default=None)


@contextmanager
def track_usage():
    records: list[dict] = []
    token = _records.set(records)
    try:
        yield records
    finally:
        _records.reset(token)


def _as_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = PRICES_PER_MILLION.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def record_usage(role: str, model: str, usage: Any) -> None:
    """Record one API call. Reasoning tokens are billed as output and are included in completion_tokens."""
    data = _as_dict(usage)
    input_tokens = int(data.get("prompt_tokens") or 0)
    output_tokens = int(data.get("completion_tokens") or 0)
    reasoning_tokens = int((data.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    record = {
        "role": role,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": _cost(model, input_tokens, output_tokens),
    }
    _log.debug("usage  role=%s  model=%s  in=%d  out=%d  reasoning=%d", role, model,
               input_tokens, output_tokens, reasoning_tokens)
    records = _records.get()
    if records is not None:
        records.append(record)


def summarize(records: list[dict]) -> dict:
    by_role: dict[str, dict] = {}
    for rec in records:
        bucket = by_role.setdefault(rec["role"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        bucket["calls"] += 1
        bucket["input_tokens"] += rec["input_tokens"]
        bucket["output_tokens"] += rec["output_tokens"]
        bucket["cost_usd"] += rec["cost_usd"] or 0.0
    return {
        "calls": len(records),
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "reasoning_tokens": sum(r["reasoning_tokens"] for r in records),
        "cost_usd": round(sum(r["cost_usd"] or 0.0 for r in records), 6),
        "unpriced_calls": sum(1 for r in records if r["cost_usd"] is None),
        "by_role": {role: {**v, "cost_usd": round(v["cost_usd"], 6)} for role, v in by_role.items()},
    }
