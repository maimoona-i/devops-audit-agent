import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import LLMError, make_llm, to_anthropic_messages, to_openai_messages, with_retries  # noqa: E402

CONVERSATION = [
    {"role": "user", "content": "audit it"},
    {"role": "assistant", "text": "looking", "tool_calls": [{"id": "c1", "name": "list_files", "arguments": {}}]},
    {"role": "tool_results", "results": [{"id": "c1", "name": "list_files", "content": "README.md", "is_error": False}],
     "note": "hurry up"},
]


def test_anthropic_conversion():
    out = to_anthropic_messages(CONVERSATION)
    assert out[1]["content"][1] == {"type": "tool_use", "id": "c1", "name": "list_files", "input": {}}
    last = out[2]["content"]
    assert last[0]["type"] == "tool_result" and last[0]["tool_use_id"] == "c1"
    assert last[1] == {"type": "text", "text": "hurry up"}


def test_openai_conversion():
    out = to_openai_messages("SYS", CONVERSATION)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[2]["tool_calls"][0]["function"]["name"] == "list_files"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "README.md"}
    assert out[4] == {"role": "user", "content": "hurry up"}


def test_empty_assistant_turn_still_valid():
    msgs = [{"role": "assistant", "text": "", "tool_calls": []}]
    assert to_anthropic_messages(msgs)[0]["content"][0]["type"] == "text"
    assert to_openai_messages("s", msgs)[1]["content"]


def test_make_llm_requires_credentials():
    with pytest.raises(LLMError, match="No LLM credentials"):
        make_llm(env={})
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        make_llm(provider="anthropic", env={})
    with pytest.raises(LLMError, match="Unknown LLM provider"):
        make_llm(provider="nope", env={"ANTHROPIC_API_KEY": "x"})


def test_make_llm_autodetects_provider_and_model():
    client = make_llm(env={"OPENAI_API_KEY": "k", "LLM_MODEL": "my-model"})
    assert type(client).__name__ == "OpenAIClient" and client.model == "my-model"
    client = make_llm(env={"ANTHROPIC_API_KEY": "k"})
    assert type(client).__name__ == "AnthropicClient"


class _Flaky(Exception):
    status_code = 429


class _AuthFail(Exception):
    status_code = 401


def test_retries_transient_errors_then_succeeds():
    calls = {"n": 0}
    sleeps = []

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Flaky("slow down")
        return "ok"

    assert with_retries(fn, sleep=sleeps.append) == "ok"
    assert calls["n"] == 3 and sleeps == [2.0, 4.0]


def test_gives_up_after_max_attempts():
    with pytest.raises(LLMError):
        with_retries(lambda: (_ for _ in ()).throw(_Flaky("x")), attempts=2, sleep=lambda _s: None)


def test_auth_errors_are_not_retried_and_give_a_hint():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _AuthFail("bad key")

    with pytest.raises(LLMError, match="API key"):
        with_retries(fn, sleep=lambda _s: None)
    assert calls["n"] == 1
