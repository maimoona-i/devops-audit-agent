"""End-to-end CLI tests with the network and the LLM replaced by local fakes."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as cli  # noqa: E402
from llm import LLMError, LLMResponse, ToolCall  # noqa: E402
from tools import RepoError, RepoWorkspace  # noqa: E402

URL = "https://github.com/octo/demo"


class ScriptedLLM:
    def __init__(self):
        self.n = 0

    def complete(self, system, messages, tools, force_tool=None):
        self.n += 1
        if self.n == 1:
            return LLMResponse(tool_calls=[ToolCall("1", "list_files", {})])
        return LLMResponse(tool_calls=[ToolCall("2", "submit_report", {"violations": [{
            "rule_id": "REPO-001", "category": "repository_hygiene", "severity": "medium", "file": None,
            "line": None, "description": "No LICENSE file", "evidence": "list_files shows no LICENSE",
            "recommendation": "Add a LICENSE"}], "summary": "ok"})])


@pytest.fixture()
def local_repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("# hi\n")
    monkeypatch.setattr(cli.RepoWorkspace, "from_url", staticmethod(lambda url: RepoWorkspace(root, url=url)))
    monkeypatch.setattr(cli, "make_llm", lambda provider=None, model=None: ScriptedLLM())
    monkeypatch.setattr(cli, "_load_env", lambda: None)
    return root


def test_successful_run_writes_valid_report_and_trace(local_repo, tmp_path):
    out = tmp_path / "report.json"
    assert cli.main(["--repo", URL, "--output", str(out), "--quiet"]) == 0
    report = json.loads(out.read_text())
    assert report["repository"] == URL
    assert report["violations"][0]["rule_id"] == "REPO-001"
    assert report["violations"][0]["file"] is None and report["violations"][0]["line"] is None
    trace = json.loads((tmp_path / "report.trace.json").read_text())
    assert trace["stats"]["tools_used"] == {"list_files": 1}


def test_invalid_url_exits_nonzero_without_output(local_repo, tmp_path, capsys):
    out = tmp_path / "report.json"
    assert cli.main(["--repo", "https://example.com/x/y", "--output", str(out)]) == 2
    assert "error:" in capsys.readouterr().err and not out.exists()


def test_clone_failure_exits_nonzero_with_message(local_repo, tmp_path, monkeypatch, capsys):
    def boom(url):
        raise RepoError("Could not clone: repository not found")

    monkeypatch.setattr(cli.RepoWorkspace, "from_url", staticmethod(boom))
    out = tmp_path / "report.json"
    assert cli.main(["--repo", URL, "--output", str(out), "--quiet"]) == 2
    assert "not found" in capsys.readouterr().err and not out.exists()


def test_missing_credentials_exits_nonzero(local_repo, tmp_path, monkeypatch, capsys):
    def no_key(provider=None, model=None):
        raise LLMError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(cli, "make_llm", no_key)
    out = tmp_path / "report.json"
    assert cli.main(["--repo", URL, "--output", str(out), "--quiet"]) == 2
    assert "API_KEY" in capsys.readouterr().err and not out.exists()


def test_unexpected_exception_gives_clean_error(local_repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_llm", lambda provider=None, model=None: (_ for _ in ()).throw(ValueError("boom")))
    monkeypatch.delenv("AGENT_DEBUG", raising=False)
    assert cli.main(["--repo", URL, "--output", str(tmp_path / "r.json"), "--quiet"]) == 1
    assert "unexpected ValueError" in capsys.readouterr().err


def test_missing_required_arguments_exit_nonzero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--repo", URL])
    assert exc.value.code != 0
