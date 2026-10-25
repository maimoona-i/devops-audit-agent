"""Agent-loop behaviour tests using a scripted fake LLM (no network, no API key)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditor import AgentError, AuditAgent  # noqa: E402
from llm import LLMError, LLMResponse, ToolCall  # noqa: E402
from tools import RepoWorkspace, ToolBox  # noqa: E402

FAKE_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"


class FakeLLM:
    """Calls ``policy(messages, force_tool, call_no)`` to decide each reply."""

    def __init__(self, policy):
        self.policy = policy
        self.calls = 0
        self.seen_force: list[str | None] = []
        self.last_messages: list = []

    def complete(self, system, messages, tools, force_tool=None):
        self.calls += 1
        self.seen_force.append(force_tool)
        self.last_messages = messages
        return self.policy(messages, force_tool, self.calls)


def call(name, cid="c", **args):
    return LLMResponse(tool_calls=[ToolCall(id=f"{cid}", name=name, arguments=args)])


def submit(*violations, cid="s"):
    return call("submit_report", cid, violations=list(violations), summary="done")


def viol(**over):
    base = {"rule_id": "SEC-001", "category": "security", "severity": "critical", "file": "config.py", "line": 1,
            "description": "Hardcoded API credential", "evidence": "Credential assigned in source",
            "recommendation": "Use env vars"}
    base.update(over)
    return base


@pytest.fixture()
def toolbox(tmp_path):
    (tmp_path / "README.md").write_text("# demo\n")
    (tmp_path / "config.py").write_text(f'API_KEY = "{FAKE_KEY}"\nDEBUG = True\n')
    (tmp_path / "requirements.txt").write_text("flask\n")
    return ToolBox(RepoWorkspace(tmp_path))


def last_tool_text(messages):
    for m in reversed(messages):
        if m["role"] == "tool_results":
            return "\n".join(r["content"] for r in m["results"])
    return ""


def make_agent(llm, toolbox, **kw):
    return AuditAgent(llm, toolbox, "https://github.com/o/r", log=None, **kw)


def test_loop_executes_tools_feeds_results_back_and_submits(toolbox):
    script = [call("list_files"), call("read_file", path="config.py"), submit(viol())]
    llm = FakeLLM(lambda m, f, n: script[n - 1])
    agent = make_agent(llm, toolbox)
    violations = agent.run()
    assert [v["rule_id"] for v in violations] == ["SEC-001"]
    assert [e["tool"] for e in agent.trace if e["type"] == "tool_call"] == ["list_files", "read_file"]
    # the second model turn must have seen the real tool output
    assert agent.stats()["tools_used"] == {"list_files": 1, "read_file": 1}


def test_next_tool_depends_on_previous_observation(toolbox):
    """The fake model behaves like an LLM: it picks its follow-up from what the last tool returned."""
    def policy(messages, force, n):
        if n == 1:
            return call("list_files")
        seen = last_tool_text(messages)
        if n == 2 and "requirements.txt" in seen:
            return call("read_file", path="requirements.txt")
        if n == 3 and "flask" in seen:
            return call("search_code", pattern="flask")
        return submit()

    agent = make_agent(FakeLLM(policy), toolbox)
    agent.run()
    tools = [e["tool"] for e in agent.trace if e["type"] == "tool_call"]
    assert tools == ["list_files", "read_file", "search_code"]
    assert agent.trace[1]["arguments"] == {"path": "requirements.txt"}


def test_empty_violations_is_a_valid_outcome(toolbox):
    llm = FakeLLM(lambda m, f, n: call("list_files") if n == 1 else submit())
    assert make_agent(llm, toolbox).run() == []


def test_unknown_tool_error_is_returned_to_model_and_loop_continues(toolbox):
    seen = {}

    def policy(messages, force, n):
        if n == 1:
            return call("rm_rf", path="/")
        seen["text"] = last_tool_text(messages)
        return submit()

    make_agent(FakeLLM(policy), toolbox).run()
    assert "Unknown tool" in seen["text"]


def test_secret_values_never_reach_model_trace_or_report(toolbox):
    seen = {}

    def policy(messages, force, n):
        if n == 1:
            return call("read_file", path="config.py")
        seen["text"] = last_tool_text(messages)
        # a misbehaving model tries to put the secret in its report anyway
        return submit(viol(evidence=f"API_KEY is {FAKE_KEY}"))

    agent = make_agent(FakeLLM(policy), toolbox)
    violations = agent.run()
    assert FAKE_KEY not in seen["text"]
    assert FAKE_KEY not in str(violations)
    assert FAKE_KEY not in str(agent.trace)


def test_model_that_never_calls_tools_is_nudged_then_fails(toolbox):
    llm = FakeLLM(lambda m, f, n: LLMResponse(text="I think it is fine."))
    with pytest.raises(AgentError, match="stopped without"):
        make_agent(llm, toolbox).run()
    assert llm.calls == 4  # initial turn + 3 nudges


def test_nudge_recovers_when_model_then_acts(toolbox):
    llm = FakeLLM(lambda m, f, n: LLMResponse(text="hmm") if n == 1 else submit())
    assert make_agent(llm, toolbox).run() == []


def test_step_budget_forces_submission(toolbox):
    def policy(messages, force, n):
        return submit(viol()) if force == "submit_report" else call("list_files", cid=f"c{n}")

    llm = FakeLLM(policy)
    violations = make_agent(llm, toolbox, max_steps=4).run()
    assert len(violations) == 1
    assert "submit_report" in llm.seen_force and llm.calls <= 5


def test_time_budget_forces_submission(toolbox):
    def policy(messages, force, n):
        return submit() if force == "submit_report" else call("list_files", cid=f"c{n}")

    llm = FakeLLM(policy)
    make_agent(llm, toolbox, max_seconds=-1).run()
    assert llm.seen_force[0] == "submit_report"


def test_model_ignoring_forced_submission_eventually_errors(toolbox):
    llm = FakeLLM(lambda m, f, n: call("list_files", cid=f"c{n}"))
    with pytest.raises(AgentError, match="did not submit"):
        make_agent(llm, toolbox, max_steps=3).run()


def test_invalid_submission_is_rejected_then_corrected(toolbox):
    seen = {}

    def policy(messages, force, n):
        if n == 1:
            return call("submit_report", violations="not a list")
        seen["text"] = last_tool_text(messages)
        return submit(viol())

    violations = make_agent(FakeLLM(policy), toolbox).run()
    assert "rejected" in seen["text"] and len(violations) == 1


def test_repeated_invalid_submissions_fail(toolbox):
    llm = FakeLLM(lambda m, f, n: call("submit_report", cid=f"s{n}", violations=None))
    with pytest.raises(AgentError, match="invalid report"):
        make_agent(llm, toolbox).run()


def test_llm_errors_propagate(toolbox):
    def policy(messages, force, n):
        raise LLMError("API down")

    with pytest.raises(LLMError):
        make_agent(FakeLLM(policy), toolbox).run()


def test_old_tool_results_are_trimmed_to_bound_context(toolbox):
    big = "x" * 5000
    messages = [{"role": "tool_results", "results": [{"id": str(i), "name": "t", "content": big, "is_error": False}]}
                for i in range(10)]
    AuditAgent._trim_old_results(messages)
    lengths = [len(m["results"][0]["content"]) for m in messages]
    assert lengths[0] < 1500 and lengths[-1] == 5000


def test_extra_content_survives_into_conversation_history(toolbox):
    sig = {"google": {"thought_signature": "abc"}}
    seen = {}

    def policy(messages, force, n):
        if n == 1:
            return LLMResponse(tool_calls=[ToolCall("c1", "list_files", {}, extra_content=sig)])
        seen["assistant"] = [m for m in messages if m["role"] == "assistant"][0]
        return submit()

    make_agent(FakeLLM(policy), toolbox).run()
    assert seen["assistant"]["tool_calls"][0]["extra_content"] == sig
