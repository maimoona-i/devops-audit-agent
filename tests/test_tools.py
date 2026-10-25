"""Unit tests for tools.py. They use a local throwaway git repo, so no network is needed."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import (  # noqa: E402
    REDACTED, RepoError, RepoWorkspace, ToolBox, parse_github_url, redact,
)

FAKE_AWS = "AKIA" + "ABCDEFGHIJKLMNOP"          # built at runtime so no real-looking key sits in the repo
FAKE_PASSWORD = "hunter2hunter2"

WORKFLOW = """\
name: CI
on:
  pull_request_target:
    branches: [main]
permissions: write-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - uses: some/action@main
      - name: Greet
        run: echo "${{ github.event.pull_request.title }}"
      - name: Tests
        run: pytest
        continue-on-error: true
"""


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=t@t.t", "-c", "user.name=t", *args],
                   cwd=cwd, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path: Path) -> RepoWorkspace:
    root = tmp_path / "demo"
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "README.md").write_text("# Demo\n")
    (root / ".gitignore").write_text("*.log\n")
    (root / "src" / "config.py").write_text(
        f'AWS_KEY = "{FAKE_AWS}"\nDB_PASSWORD = "{FAKE_PASSWORD}"\n'
        'SAFE = os.environ["TOKEN_VALUE"]\nPATH = "/home/alice/data"\n')
    (root / "debug.log").write_text("temporary\n")
    (root / ".github" / "workflows" / "ci.yml").write_text(WORKFLOW)
    (root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "add", "-A", "-f")                       # -f so the .log file is tracked despite .gitignore
    _git(root, "commit", "-q", "-m", "initial commit")
    return RepoWorkspace(root)


# ------------------------------------------------------------------ redaction
def test_redact_masks_known_token_formats():
    assert FAKE_AWS not in redact(f"key={FAKE_AWS}")
    assert "ghp_" not in redact("t = ghp_" + "a" * 36)


def test_redact_masks_assignments_but_keeps_references():
    assert FAKE_PASSWORD not in redact(f'password = "{FAKE_PASSWORD}"')
    assert redact('password = os.environ["PASSWORD"]') == 'password = os.environ["PASSWORD"]'
    assert "${{ secrets.TOKEN }}" in redact("token: ${{ secrets.TOKEN }}")


def test_redact_masks_connection_string_password():
    out = redact("postgres://admin:s3cretpass@db.example.com/app")
    assert "s3cretpass" not in out and "admin" in out


def test_redact_masks_private_key_block():
    out = redact("-----BEGIN RSA PRIVATE KEY-----\nMIIEabc\n-----END RSA PRIVATE KEY-----")
    assert "MIIEabc" not in out


# ------------------------------------------------------------------ url parsing
@pytest.mark.parametrize("url,expected", [
    ("https://github.com/octo/demo", ("octo", "demo")),
    ("https://github.com/octo/demo.git", ("octo", "demo")),
    ("https://github.com/octo/demo/tree/main/src", ("octo", "demo")),
    ("http://www.github.com/octo/demo/", ("octo", "demo")),
])
def test_parse_github_url_valid(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize("url", [
    "", "not a url", "ftp://github.com/a/b", "https://gitlab.com/a/b",
    "https://github.com/onlyowner", "https://github.com/a/b;rm -rf", "git@github.com:a/b.git",
])
def test_parse_github_url_invalid(url):
    with pytest.raises(RepoError):
        parse_github_url(url)


# ------------------------------------------------------------------ list_files
def test_list_files_shows_structure_and_skips_git(repo):
    out = repo.list_files()
    assert "README.md" in out and "src/config.py" in out
    assert ".git/" not in out


def test_list_files_filters(repo):
    assert "config.py" in repo.list_files(path="src")
    assert "README.md" not in repo.list_files(path="src")
    assert "ci.yml" in repo.list_files(pattern="*.yml")


# ------------------------------------------------------------------ read_file
def test_read_file_masks_secrets_and_numbers_lines(repo):
    out = repo.read_file("src/config.py")
    assert FAKE_AWS not in out and FAKE_PASSWORD not in out
    assert REDACTED in out
    assert "    1|" in out
    assert 'os.environ["TOKEN_VALUE"]' in out


def test_read_file_chunking(repo):
    out = repo.read_file("src/config.py", start_line=2, max_lines=1)
    assert "lines 2-2" in out and "more line" in out


def test_read_file_rejects_traversal(repo):
    toolbox = ToolBox(repo)
    text, is_err = toolbox.call("read_file", {"path": "../../etc/passwd"})
    assert is_err and "escapes" in text


def test_read_file_rejects_symlink_escape(repo, tmp_path):
    secret = tmp_path / "outside.txt"
    secret.write_text("top secret")
    try:
        (repo.root / "link.txt").symlink_to(secret)
    except OSError:
        pytest.skip("symlinks not permitted on this system (Windows without Developer Mode)")    
    toolbox = ToolBox(repo)
    text, is_err = toolbox.call("read_file", {"path": "link.txt"})
    assert is_err and "top secret" not in text


def test_read_file_missing_and_binary(repo):
    toolbox = ToolBox(repo)
    assert toolbox.call("read_file", {"path": "nope.txt"})[1] is True
    text, is_err = toolbox.call("read_file", {"path": "blob.bin"})
    assert not is_err and "binary" in text


# ------------------------------------------------------------------ search_code
def test_search_code_finds_machine_specific_path(repo):
    out = repo.search_code(r"/home/\w+")
    assert "src/config.py:4" in out


def test_search_code_masks_secret_in_hits(repo):
    out = repo.search_code("AWS_KEY")
    assert FAKE_AWS not in out and "src/config.py:1" in out


def test_search_code_no_match_and_bad_regex(repo):
    assert "No matches" in repo.search_code("zzz_not_present")
    text, is_err = ToolBox(repo).call("search_code", {"pattern": "([unclosed"})
    assert is_err and "invalid regular expression" in text


def test_search_code_respects_glob(repo):
    assert "No matches" in repo.search_code("Demo", file_glob="*.py")
    assert "README.md:1" in repo.search_code("Demo", file_glob="*.md")


# ------------------------------------------------------------------ git tools
def test_git_log(repo):
    out = repo.git_log()
    assert "initial commit" in out and "1 commit(s)" in out


def test_tracked_ignored_files_detects_committed_log(repo):
    out = repo.tracked_ignored_files()
    assert "debug.log" in out
    assert ".gitignore present: True" in out


# ------------------------------------------------------------------ workflows
def test_inspect_workflows_reports_risky_facts(repo):
    out = repo.inspect_workflows()
    assert "pull_request_target" in out
    assert "write-all" in out
    assert "mutable branch 'main'" in out
    assert "continue-on-error=true" in out
    assert "untrusted github.event" in out
    assert "checkout under pull_request_target" in out


def test_inspect_workflows_none(tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    assert "No GitHub Actions" in RepoWorkspace(tmp_path).inspect_workflows()


def test_inspect_workflows_bad_yaml(tmp_path):
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "bad.yml").write_text("a: [unclosed")
    assert "parse error" in RepoWorkspace(tmp_path).inspect_workflows().lower()


# ------------------------------------------------------------------ ToolBox
def test_toolbox_unknown_tool_and_bad_args(repo):
    toolbox = ToolBox(repo)
    assert toolbox.call("rm_rf", {})[1] is True
    assert toolbox.call("read_file", {"wrong": 1})[1] is True
    assert toolbox.call("list_files", None)[1] is False


def test_toolbox_never_mutates_repo(repo):
    before = subprocess.run(["git", "-C", str(repo.root), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    toolbox = ToolBox(repo)
    for spec in toolbox.specs:
        args = {"path": "README.md"} if spec["name"] == "read_file" else \
               {"pattern": "x"} if spec["name"] == "search_code" else {}
        toolbox.call(spec["name"], args)
    after = subprocess.run(["git", "-C", str(repo.root), "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    assert before == after


def test_clone_rejects_bad_url_before_touching_network():
    with pytest.raises(RepoError):
        RepoWorkspace.from_url("https://example.com/a/b")
