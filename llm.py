"""Thin LLM provider layer (Anthropic or any OpenAI-compatible API).

The agent talks to the model only through :class:`LLMClient.complete`, using a small
provider-neutral message format, so the control flow in ``auditor.py`` never depends on
a specific SDK. No agent framework is used: only the basic provider SDKs.

Neutral message format
----------------------
* ``{"role": "user", "content": str}``
* ``{"role": "assistant", "text": str, "tool_calls": [{"id", "name", "arguments"}]}``
* ``{"role": "tool_results", "results": [{"id", "name", "content", "is_error"}], "note": str?}``
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from tools import redact

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}
REQUEST_TIMEOUT = 90.0
MAX_OUTPUT_TOKENS = 8192


class LLMError(Exception):
    """Fatal problem talking to the LLM provider (config, auth, repeated API failures)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 force_tool: str | None = None) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# Retry handling
# --------------------------------------------------------------------------- #
_RETRYABLE_NAMES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
                    "OverloadedError"}


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status in (408, 409, 429) or status >= 500):
        return True
    return type(exc).__name__ in _RETRYABLE_NAMES


def _describe(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    msg = f"{type(exc).__name__}: {redact(str(exc))[:300]}"
    if status in (401, 403):
        msg += " (check that your API key is valid and has access to the chosen model)"
    elif status == 404:
        msg += " (check the model name; set LLM_MODEL or --model)"
    return msg


def with_retries(fn: Callable[[], Any], attempts: int = 4, sleep: Callable[[float], None] = time.sleep) -> Any:
    """Call ``fn`` with exponential backoff on transient errors; raise LLMError otherwise."""
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # SDK exception types differ per provider
            if not _retryable(exc) or attempt == attempts:
                raise LLMError(_describe(exc)) from None
            sleep(delay)
            delay *= 2
    raise LLMError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Message conversion (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.get("text"):
                blocks.append({"type": "text", "text": m["text"]})
            for c in m.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]})
            if not blocks:
                blocks.append({"type": "text", "text": "(no output)"})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool_results":
            blocks = [{"type": "tool_result", "tool_use_id": r["id"], "content": r["content"],
                       "is_error": bool(r.get("is_error"))} for r in m["results"]]
            if m.get("note"):
                blocks.append({"type": "text", "text": m["note"]})
            out.append({"role": "user", "content": blocks})
    return out


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("text") or None}
            calls = m.get("tool_calls", [])
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                     for c in calls]
            elif not msg["content"]:
                msg["content"] = "(no output)"
            out.append(msg)
        elif role == "tool_results":
            for r in m["results"]:
                out.append({"role": "tool", "tool_call_id": r["id"], "content": r["content"]})
            if m.get("note"):
                out.append({"role": "user", "content": m["note"]})
    return out


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class AnthropicClient:
    def __init__(self, api_key: str, model: str):
        try:
            import anthropic
        except ImportError:
            raise LLMError("The 'anthropic' package is not installed. Run: pip install -r requirements.txt") from None
        self._client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=0)
        self.model = model

    def complete(self, system, messages, tools, force_tool=None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "system": system,
            "messages": to_anthropic_messages(messages),
            "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                      for t in tools],
        }
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}
        resp = with_retries(lambda: self._client.messages.create(**kwargs))
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                calls.append(ToolCall(id=block.id, name=block.name, arguments=args))
        return LLMResponse(text="\n".join(text_parts).strip(), tool_calls=calls)


class OpenAIClient:
    """Works with OpenAI and any OpenAI-compatible endpoint (set OPENAI_BASE_URL)."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        try:
            import openai
        except ImportError:
            raise LLMError("The 'openai' package is not installed. Run: pip install -r requirements.txt") from None
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url or None,
                                     timeout=REQUEST_TIMEOUT, max_retries=0)
        self.model = model

    def complete(self, system, messages, tools, force_tool=None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": to_openai_messages(system, messages),
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["input_schema"]}} for t in tools],
        }
        if force_tool:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
        resp = with_retries(lambda: self._client.chat.completions.create(**kwargs))
        if not resp.choices:
            raise LLMError("The provider returned an empty response.")
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args if isinstance(args, dict) else {}))
        return LLMResponse(text=(msg.content or "").strip(), tool_calls=calls)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_llm(provider: str | None = None, model: str | None = None,
             env: Mapping[str, str] | None = None) -> LLMClient:
    """Build a client from arguments/environment. Raises LLMError with a helpful message."""
    env = os.environ if env is None else env

    def get(name: str) -> str:
        return (env.get(name) or "").strip()

    provider = (provider or get("LLM_PROVIDER")).lower()
    if not provider:
        if get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise LLMError("No LLM credentials found. Copy .env.example to .env and set ANTHROPIC_API_KEY "
                           "or OPENAI_API_KEY (see README).")
    if provider not in DEFAULT_MODELS:
        raise LLMError(f"Unknown LLM provider {provider!r}; use 'anthropic' or 'openai'.")
    model = model or get("LLM_MODEL") or DEFAULT_MODELS[provider]
    key_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = get(key_name)
    if not api_key:
        raise LLMError(f"{key_name} is not set (provider={provider}). Copy .env.example to .env and fill it in.")
    if provider == "anthropic":
        return AnthropicClient(api_key, model)
    return OpenAIClient(api_key, model, base_url=get("OPENAI_BASE_URL") or None)
