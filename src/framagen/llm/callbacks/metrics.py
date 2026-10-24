# Provider-agnostic token and latency collection.

from __future__ import annotations

import random
import time
from copy import deepcopy
from typing import Any, Mapping

try:  # Optional dependency; a caller may supply any ``.invoke`` compatible client.
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover - exercised when the optional extra is absent
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_token_usage(usage: Any) -> dict[str, Any] | None:
    # Normalize OpenAI, Anthropic, and Ollama usage fields without guessing.
    if not isinstance(usage, Mapping):
        return None
    raw = dict(usage)
    input_tokens = _integer(raw.get("input_tokens", raw.get("prompt_tokens", raw.get("prompt_eval_count"))))
    output_tokens = _integer(raw.get("output_tokens", raw.get("completion_tokens", raw.get("eval_count"))))
    total_tokens = _integer(raw.get("total_tokens"))
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "raw": deepcopy(raw),
    }


class LLMRunMetrics(BaseCallbackHandler):
    # Collect one invocation's elapsed time and reported provider usage.

    def __init__(self) -> None:
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.token_usage: dict[str, Any] | None = None
        self.model_name: str | None = None
        self.call_count = 0
        self._captured_messages = set()

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self._started()

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        self._started()

    def _started(self) -> None:
        self.start_time = self.start_time or time.perf_counter()
        self.call_count += 1

    def capture_from_message(self, message: Any) -> None:
        if message is None:
            return
        identity = getattr(message, "id", None) or id(message)
        if identity in self._captured_messages:
            return
        self._captured_messages.add(identity)
        self._started()
        for usage in (
            getattr(message, "usage_metadata", None),
            getattr(message, "response_metadata", {}).get("token_usage")
            if isinstance(getattr(message, "response_metadata", None), Mapping)
            else None,
        ):
            normalized = normalize_token_usage(usage)
            if normalized is not None:
                self._merge(normalized)
                break
        metadata = getattr(message, "response_metadata", None)
        if isinstance(metadata, Mapping):
            self.model_name = self.model_name or metadata.get("model_name") or metadata.get("model")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            self.capture_from_message(response.generations[0][0].message)
        except (AttributeError, IndexError, TypeError):
            pass
        self.end_time = time.perf_counter()

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._started()
        self.end_time = time.perf_counter()

    def _merge(self, usage: dict[str, Any]) -> None:
        if self.token_usage is None:
            self.token_usage = deepcopy(usage)
            self.token_usage["calls"] = 1
            return
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            current, incoming = self.token_usage.get(key), usage.get(key)
            if current is None:
                self.token_usage[key] = incoming
            elif incoming is not None:
                self.token_usage[key] = current + incoming
        self.token_usage["calls"] = int(self.token_usage["calls"]) + 1

    def as_dict(self) -> dict[str, Any]:
        latency = None
        if self.start_time is not None:
            latency = (self.end_time or time.perf_counter()) - self.start_time
        return {
            "latency_seconds": latency,
            "token_usage": deepcopy(self.token_usage),
            "model_name": self.model_name,
            "call_count": self.call_count,
        }


def is_retryable_llm_error(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in (
        "overloaded", "529", "too many requests", "rate limit", "ratelimit",
        "service unavailable", "bad gateway", "gateway timeout",
    ))


def invoke_with_metrics(
    runnable: Any,
    payload: Any,
    metrics: LLMRunMetrics,
    *,
    attempts: int = 1,
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 30.0,
) -> Any:
    # Invoke a runnable with bounded exponential retries and attached metrics.
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return runnable.invoke(payload, config={"callbacks": [metrics]})
        except Exception as error:
            metrics.on_llm_error(error)
            if attempt >= max(1, attempts) or not is_retryable_llm_error(error):
                raise
            delay = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
            if delay:
                time.sleep(delay + random.uniform(0, delay * 0.2))
    raise AssertionError("unreachable")
