from __future__ import annotations

from typing import Any

# Ollama's default context window is far smaller than the deep-agent system
# prompt plus skills plus a C file; without this the prompt is truncated
# silently and the model answers a partial task.
OLLAMA_DEFAULT_NUM_CTX = 32768
# Cap each response, as ContractFix does (`num_predict`): qwen3's thinking mode
# can otherwise loop for tens of thousands of tokens on a single turn.
DEFAULT_MAX_OUTPUT_TOKENS = 8192


def make_model(
    spec: str,
    *,
    temperature: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    base_url: str | None = None,
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str | Any:
    """Return a model for ``create_deep_agent`` from a ``provider:model`` spec.

    Examples: ``ollama:qwen3:8b``, ``anthropic:claude-sonnet-5-5``, ``openai:gpt-5.4``.
    Ollama reads ``OLLAMA_HOST`` for the server address.
    """
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    if base_url is not None:
        kwargs["base_url"] = base_url
    if spec.startswith("ollama:"):
        kwargs["num_ctx"] = num_ctx or OLLAMA_DEFAULT_NUM_CTX
        if max_output_tokens:
            kwargs["num_predict"] = max_output_tokens
    elif max_output_tokens and spec.split(":", 1)[0] in {"anthropic", "openai"}:
        kwargs["max_tokens"] = max_output_tokens
    if not kwargs:
        return spec
    from deepagents.profiles.provider.provider_profiles import apply_provider_profile
    from langchain.chat_models import init_chat_model

    return init_chat_model(spec, **{**apply_provider_profile(spec), **kwargs})
