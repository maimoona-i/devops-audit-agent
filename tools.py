"""Inspection tools for the DevOps auditing agent.

Design principles
-----------------
* READ-ONLY: tools never modify the inspected repository and never execute code
  from it. The repository is cloned into a private temp directory, hooks are
  disabled, and only static inspection (file reads, regex search, ``git log``)
  is performed.
* SECRET-SAFE: every string returned to the LLM (and therefore to logs/report)
  passes through :func:`redact`, so credential values never leave this module.
* BOUNDED: every tool caps its output size and runtime so a hostile or huge
  repository cannot hang or flood the agent.
* FACTUAL: tools report facts (what exists, what a workflow does). Judging
  whether something is a violation is the LLM's job, not the tool's.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:  # PyYAML is only needed for inspect_workflows
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
MAX_FILE_BYTES = 200_000          # refuse to read bigger files in one go
MAX_READ_LINES = 400              # default lines returned by read_file
MAX_LIST_FILES = 600              # files returned by list_files
MAX_SEARCH_RESULTS = 60
MAX_OUTPUT_CHARS = 12_000         # hard cap on any single tool result
CLONE_TIMEOUT = 120
GIT_TIMEOUT = 30
SKIP_DIRS = {".git"}

# --------------------------------------------------------------------------- #
# Secret redaction
# --------------------------------------------------------------------------- #
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"AKIA[0-9A-Z]{16}"),                                   # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),                     # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),                         # OpenAI/Anthropic style
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),                  # Slack
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),                         # Google API key
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
    re.compile(r"(?i)\b(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://([^:/\s]+):([^@\s]+)@"),
]
# key = "value" style assignments with a sensitive-looking key name
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<key>[\w\-.]*(?:password|passwd|pwd|secret|token|api[_\-]?key|apikey|
                       access[_\-]?key|private[_\-]?key|auth|credential)[\w\-.]*)
    (?P<sep>\s*[:=]\s*)
    (?P<quote>["']?)
    (?P<val>[^\s"'#,;]{6,})
    (?P=quote)
    """
)
_PLACEHOLDER_HINTS = (
    "${{", "${", "$(", "<", "{{", "xxx", "your", "changeme", "example",
    "placeholder", "process.env", "os.environ", "os.getenv", "getenv",
    "secrets.", "env.", "none", "null", "true", "false", "***", "redacted",
)
REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    """Mask anything that looks like a credential value."""
    if not text:
        return text
    for pat in _SECRET_PATTERNS:
        if pat.groups >= 3:  # connection strings: keep scheme/user, drop password
            text = pat.sub(lambda m: f"{m.group(1)}://{m.group(2)}:{REDACTED}@", text)
        else:
            text = pat.sub(REDACTED, text)

    def _assign(m: re.Match[str]) -> str:
        val = m.group("val")
        low = val.lower()
        if any(h in low for h in _PLACEHOLDER_HINTS) or val == REDACTED:
            return m.group(0)  # references / placeholders are not secrets
        return f"{m.group('key')}{m.group('sep')}{m.group('quote')}{REDACTED}{m.group('quote')}"

    return _ASSIGNMENT.sub(_assign, text)


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [output truncated, {len(text) - limit} more chars]"


# --------------------------------------------------------------------------- #
# Errors / URL validation
# --------------------------------------------------------------------------- #
class ToolError(Exception):
    """Recoverable tool failure; the message is shown to the LLM."""


class RepoError(Exception):
    """Fatal problem obtaining the repository (bad URL, clone failure, ...)."""


_GH_SEGMENT = re.compile(r"^[A-Za-z0-9._\-]+$")


def parse_github_url(url: str) -> tuple[str, str]:
    """Validate a public GitHub repo URL and return ``(owner, repo)``."""
    if not isinstance(url, str) or not url.strip():
        raise RepoError("Repository URL is empty.")
    url = url.strip()
    if re.search(r"[\s;`$|&<>\\]", url):
        raise RepoError("URL contains whitespace or shell metacharacters.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RepoError(f"Unsupported URL scheme in {url!r}; use https://github.com/<owner>/<repo>.")
    if (parsed.hostname or "").lower() not in ("github.com", "www.github.com"):
        raise RepoError(f"Only github.com repositories are supported (got {parsed.hostname!r}).")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise RepoError("URL must look like https://github.com/<owner>/<repo>.")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not (_GH_SEGMENT.match(owner) and _GH_SEGMENT.match(repo)) or repo in (".", ".."):
        raise RepoError("Owner/repo contain invalid characters.")
    return owner, repo


# --------------------------------------------------------------------------- #
# Repository workspace
# --------------------------------------------------------------------------- #
def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "SYSTEMROOT", "HTTPS_PROXY",
                                                         "HTTP_PROXY", "NO_PROXY", "SSL_CERT_FILE",
                                                         "REQUESTS_CA_BUNDLE", "GIT_SSL_CAINFO")}
    env["GIT_TERMINAL_PROMPT"] = "0"       # never block waiting for credentials
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


class RepoWorkspace:
    """A read-only checkout of one repository plus the inspection tools.

    Use as a context manager so the temp clone is always removed::

        with RepoWorkspace.from_url(url) as repo:
            repo.list_files()
    """

    def __init__(self, root: Path, url: str = "", _tmp: Path | None = None):
        self.root = Path(root).resolve()
        self.url = url
        self._tmp = _tmp
        self._file_cache: list[str] | None = None

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_url(cls, url: str, depth: int = 200) -> "RepoWorkspace":
        owner, repo = parse_github_url(url)
        clean_url = f"https://github.com/{owner}/{repo}.git"
        tmp = Path(tempfile.mkdtemp(prefix="devops-audit-"))
        dest = tmp / repo
        cmd = [
            "git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
            "-c", "protocol.file.allow=never", "clone", "--quiet", "--depth", str(depth),
            "--no-tags", "--no-recurse-submodules", clean_url, str(dest),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CLONE_TIMEOUT,
                                  env=_git_env(), cwd=tmp)
        except subprocess.TimeoutExpired:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RepoError(f"Timed out cloning {clean_url} after {CLONE_TIMEOUT}s.") from None
        except FileNotFoundError:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RepoError("git is not installed or not on PATH.") from None
        if proc.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            err = redact(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error")
            raise RepoError(f"Could not clone {clean_url}: {err} "
                            "(is the repository public and the URL correct?)")
        return cls(dest, url=f"https://github.com/{owner}/{repo}", _tmp=tmp)

    def close(self) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def __enter__(self) -> "RepoWorkspace":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- helpers ------------------------------------------------------------ #
    def _safe_path(self, rel: str) -> Path:
        """Resolve ``rel`` inside the repo; reject traversal and symlink escapes."""
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("path must be a non-empty string.")
        if "\x00" in rel:
            raise ToolError("invalid path.")
        candidate = (self.root / rel.strip().lstrip("/")).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise ToolError("path escapes the repository root; refused.") from None
        return candidate

    def _all_files(self) -> list[str]:
        if self._file_cache is None:
            files: list[str] = []
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                for fn in sorted(filenames):
                    full = Path(dirpath) / fn
                    if full.is_symlink():
                        continue  # never follow symlinks out of the repo
                    files.append(full.relative_to(self.root).as_posix())
            self._file_cache = files
        return self._file_cache

    def _git(self, *args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", "-c", "core.hooksPath=/dev/null", "-C", str(self.root), *args],
                capture_output=True, text=True, timeout=GIT_TIMEOUT, env=_git_env(),
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"git {args[0]} timed out.") from None
        if proc.returncode != 0:
            raise ToolError(f"git {args[0]} failed: {redact(proc.stderr.strip())[:300]}")
        return proc.stdout

    @staticmethod
    def _is_binary(path: Path) -> bool:
        try:
            with open(path, "rb") as fh:
                return b"\x00" in fh.read(2048)
        except OSError:
            return True

    # -- tool 1: list_files ------------------------------------------------- #
    def list_files(self, path: str = "", pattern: str = "") -> str:
        """List repository files (optionally under ``path`` and/or matching a glob ``pattern``)."""
        files = self._all_files()
        prefix = path.strip().strip("/") if path else ""
        if prefix:
            files = [f for f in files if f == prefix or f.startswith(prefix + "/")]
        if pattern:
            files = [f for f in files if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern)]
        total = len(files)
        shown = files[:MAX_LIST_FILES]
        lines = [f"{total} file(s) matched" + (f" (showing first {MAX_LIST_FILES})" if total > len(shown) else "") + ":"]
        for f in shown:
            try:
                size = (self.root / f).stat().st_size
            except OSError:
                size = -1
            lines.append(f"{f}  [{size} B]")
        return _clip(redact("\n".join(lines)))

    # -- tool 2: read_file -------------------------------------------------- #
    def read_file(self, path: str, start_line: int = 1, max_lines: int = MAX_READ_LINES) -> str:
        """Return numbered lines of a text file with credentials masked."""
        target = self._safe_path(path)
        if not target.exists():
            raise ToolError(f"{path!r} does not exist. Use list_files to discover paths.")
        if target.is_dir():
            raise ToolError(f"{path!r} is a directory; use list_files(path=...).")
        if target.is_symlink():
            raise ToolError("symlinks are not followed.")
        size = target.stat().st_size
        if self._is_binary(target):
            return f"{path} is a binary file ({size} bytes); contents not shown."
        if size > MAX_FILE_BYTES:
            raise ToolError(f"{path} is {size} bytes (> {MAX_FILE_BYTES}); read it in chunks with "
                            "start_line/max_lines or use search_code.")
        try:
            start = max(1, int(start_line))
            count = max(1, min(int(max_lines), MAX_READ_LINES))
        except (TypeError, ValueError):
            raise ToolError("start_line and max_lines must be integers.") from None
        text = target.read_text(encoding="utf-8", errors="replace").splitlines()
        chunk = text[start - 1: start - 1 + count]
        out = [f"{path} (lines {start}-{start + len(chunk) - 1} of {len(text)}; secrets masked):"]
        out += [f"{start + i:>5}| {redact(line)}" for i, line in enumerate(chunk)]
        if start - 1 + count < len(text):
            out.append(f"... {len(text) - (start - 1 + count)} more line(s); call again with start_line={start + count}")
        return _clip("\n".join(out))

    # -- tool 3: search_code ------------------------------------------------ #
    def search_code(self, pattern: str, file_glob: str = "", ignore_case: bool = True) -> str:
        """Regex search across text files. Returns ``file:line: text`` with secrets masked."""
        if not pattern:
            raise ToolError("pattern must be non-empty.")
        if len(pattern) > 300:
            raise ToolError("pattern too long (max 300 chars).")
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from None
        hits: list[str] = []
        truncated = False
        for rel in self._all_files():
            if file_glob and not (fnmatch.fnmatch(rel, file_glob) or fnmatch.fnmatch(Path(rel).name, file_glob)):
                continue
            full = self.root / rel
            try:
                if full.stat().st_size > MAX_FILE_BYTES or self._is_binary(full):
                    continue
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if rx.search(line):
                            snippet = redact(line.strip())[:200]
                            hits.append(f"{rel}:{lineno}: {snippet}")
                            if len(hits) >= MAX_SEARCH_RESULTS:
                                truncated = True
                                break
            except OSError:
                continue
            if truncated:
                break
        if not hits:
            return f"No matches for /{pattern}/" + (f" in files matching {file_glob}" if file_glob else "") + "."
        head = f"{len(hits)} match(es)" + (f" (capped at {MAX_SEARCH_RESULTS}; narrow the pattern)" if truncated else "") + ":"
        return _clip("\n".join([head, *hits]))

    # -- tool 4: git_log ---------------------------------------------------- #
    def git_log(self, max_count: int = 30) -> str:
        """Recent commit history: hash, author, date, subject."""
        try:
            n = max(1, min(int(max_count), 100))
        except (TypeError, ValueError):
            raise ToolError("max_count must be an integer.") from None
        out = self._git("log", f"-{n}", "--date=short", "--pretty=format:%h | %ad | %an | %s")
        total = self._git("rev-list", "--count", "HEAD").strip()
        note = ""
        if (self.root / ".git" / "shallow").exists():
            note = " (clone is shallow; older history is not available)"
        return _clip(redact(f"{total} commit(s) reachable{note}. Most recent {n}:\n{out}"))

    # -- tool 5: tracked_ignored_files (git_status equivalent) ------------- #
    def tracked_ignored_files(self) -> str:
        """Files that are committed although the repo's own .gitignore says they should be ignored,
        plus tracked files that look generated/temporary."""
        ignored = self._git("ls-files", "-ci", "--exclude-standard").splitlines()
        generated_rx = re.compile(
            r"(^|/)(__pycache__|node_modules|\.venv|venv|\.idea|\.vscode|\.DS_Store|dist|build|"
            r"\.pytest_cache|\.mypy_cache|\.ruff_cache|coverage|htmlcov)(/|$)|"
            r"\.(pyc|pyo|log|tmp|swp|bak|egg-info|class|o|so)$|(^|/)\.env($|\.(local|prod|production|dev))"
        )
        tracked = self._git("ls-files").splitlines()
        suspicious = [f for f in tracked if generated_rx.search(f)]
        has_gitignore = (self.root / ".gitignore").exists()
        lines = [f".gitignore present: {has_gitignore}",
                 f"Tracked files matching the repo's own ignore rules ({len(ignored)}):"]
        lines += [f"  {f}" for f in ignored[:80]] or ["  (none)"]
        lines.append(f"Tracked files that look generated/temporary/local ({len(suspicious)}):")
        lines += [f"  {f}" for f in suspicious[:80]] or ["  (none)"]
        return _clip(redact("\n".join(lines)))

    # -- tool 6: inspect_workflows ----------------------------------------- #
    def inspect_workflows(self) -> str:
        """Parse every GitHub Actions workflow and report triggers, permissions, jobs and risky facts.

        Reports *facts* (e.g. "action X is referenced by tag, not SHA"). The agent decides whether
        any of them is a real defect.
        """
        wf_files = [f for f in self._all_files()
                    if re.match(r"^\.github/workflows/[^/]+\.ya?ml$", f)]
        if not wf_files:
            return "No GitHub Actions workflow files found under .github/workflows/."
        if yaml is None:
            raise ToolError("PyYAML is not installed; use read_file on the workflow files instead.")
        reports: list[str] = []
        for rel in wf_files:
            try:
                data = yaml.safe_load((self.root / rel).read_text(encoding="utf-8", errors="replace"))
            except yaml.YAMLError as exc:
                reports.append(f"## {rel}\nYAML parse error: {str(exc)[:200]}")
                continue
            if not isinstance(data, dict):
                reports.append(f"## {rel}\nNot a valid workflow mapping.")
                continue
            reports.append(self._describe_workflow(rel, data))
        return _clip(redact("\n\n".join(reports)), 16_000)

    @staticmethod
    def _describe_workflow(rel: str, data: dict[str, Any]) -> str:
        on = data.get("on", data.get(True))  # PyYAML parses bare `on:` as boolean True
        lines = [f"## {rel}  (name: {data.get('name')!r})"]
        if isinstance(on, str):
            triggers = {on: None}
        elif isinstance(on, list):
            triggers = {t: None for t in on}
        elif isinstance(on, dict):
            triggers = on
        else:
            triggers = {}
        lines.append(f"triggers: {json.dumps({k: v for k, v in triggers.items()}, default=str)[:500]}")
        lines.append(f"top-level permissions: {json.dumps(data.get('permissions', 'NOT SET (defaults apply)'), default=str)}")
        if "env" in data:
            lines.append(f"top-level env keys: {sorted(map(str, (data['env'] or {}).keys()))}")
        flags: list[str] = []
        if "pull_request_target" in triggers:
            flags.append("uses pull_request_target trigger (runs with base-repo secrets/token)")
        if "workflow_run" in triggers:
            flags.append("uses workflow_run trigger")
        jobs = data.get("jobs") or {}
        for jid, job in jobs.items():
            if not isinstance(job, dict):
                continue
            lines.append(f"- job '{jid}': runs-on={job.get('runs-on')!r} "
                         f"permissions={json.dumps(job.get('permissions', 'not set'), default=str)} "
                         f"needs={job.get('needs')} if={job.get('if')!r} "
                         f"continue-on-error={job.get('continue-on-error', False)}")
            for idx, step in enumerate(job.get("steps") or []):
                if not isinstance(step, dict):
                    continue
                label = step.get("name") or step.get("uses") or (str(step.get("run", ""))[:60])
                parts = [f"    step {idx + 1}: {label!r}"]
                uses = step.get("uses")
                if uses:
                    parts.append(f"uses={uses}")
                    ref = str(uses).split("@", 1)[1] if "@" in str(uses) else None
                    if ref is None and not str(uses).startswith("./"):
                        flags.append(f"job {jid}: '{uses}' has no version pin")
                    elif ref and not re.fullmatch(r"[0-9a-f]{40}", ref) and not str(uses).startswith("./"):
                        if ref in ("main", "master", "latest", "HEAD"):
                            flags.append(f"job {jid}: '{uses}' pinned to mutable branch '{ref}'")
                if step.get("run"):
                    run = str(step["run"])
                    parts.append("run=" + run.replace("\n", " ; ")[:200])
                    if re.search(r"\$\{\{\s*github\.(event|head_ref)[^}]*\}\}", run):
                        flags.append(f"job {jid} step {idx + 1}: untrusted github.event/head_ref value interpolated into run script")
                    if re.search(r"curl[^\n|]*\|\s*(sudo\s+)?(ba)?sh", run):
                        flags.append(f"job {jid} step {idx + 1}: pipes remote script to shell")
                if step.get("continue-on-error"):
                    parts.append("continue-on-error=true")
                    flags.append(f"job {jid} step {idx + 1}: continue-on-error=true (failures ignored)")
                if step.get("if") is not None:
                    parts.append(f"if={step['if']!r}")
                    if str(step["if"]).strip().lower() in ("false", "${{ false }}"):
                        flags.append(f"job {jid} step {idx + 1}: step disabled with if: false")
                lines.append(" ".join(parts))
            if str(job.get("if", "")).strip().lower() in ("false", "${{ false }}"):
                flags.append(f"job {jid}: job disabled with if: false")
            if job.get("continue-on-error") is True:
                flags.append(f"job {jid}: continue-on-error=true at job level")
            if "pull_request_target" in triggers:
                for step in job.get("steps") or []:
                    if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/checkout"):
                        ref = json.dumps((step.get("with") or {}).get("ref"), default=str)
                        flags.append(f"job {jid}: checkout under pull_request_target with ref={ref}")
        lines.append("notable facts: " + ("; ".join(dict.fromkeys(flags)) if flags else "none"))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tool registry exposed to the LLM
# --------------------------------------------------------------------------- #
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List files in the repository (relative paths with sizes). Optionally restrict to a "
                       "sub-directory and/or a glob such as '*.yml'. Start here to learn the project layout.",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Sub-directory to list, e.g. '.github/workflows'. Default: whole repo."},
            "pattern": {"type": "string", "description": "Glob filter, e.g. '*.py' or 'requirements*.txt'."}}},
    },
    {
        "name": "read_file",
        "description": "Read a text file with line numbers. Secret values are masked as [REDACTED]. "
                       "Long files can be read in chunks with start_line/max_lines.",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string"}, "start_line": {"type": "integer"}, "max_lines": {"type": "integer"}},
            "required": ["path"]},
    },
    {
        "name": "search_code",
        "description": "Regex search over all text files; returns file:line: snippet (secrets masked). Use it to "
                       "hunt for credentials, absolute/machine-specific paths, TODO/FIXME, debug flags, etc.",
        "input_schema": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "file_glob": {"type": "string", "description": "Optional glob, e.g. '*.py'."},
            "ignore_case": {"type": "boolean"}}, "required": ["pattern"]},
    },
    {
        "name": "git_log",
        "description": "Recent commit history (hash, date, author, subject) and total commit count.",
        "input_schema": {"type": "object", "properties": {"max_count": {"type": "integer"}}},
    },
    {
        "name": "tracked_ignored_files",
        "description": "Show committed files that the repo's own .gitignore says should be ignored, and tracked "
                       "files that look generated/temporary (pycache, .env, logs, build output, IDE settings).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "inspect_workflows",
        "description": "Parse all GitHub Actions workflows and summarise triggers, permissions, jobs, steps, action "
                       "version pinning and risky constructs. Use it before reading workflow YAML by hand.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


class ToolBox:
    """Dispatches tool calls by name and guarantees a string result (never raises)."""

    def __init__(self, repo: RepoWorkspace):
        self.repo = repo
        self._dispatch = {
            "list_files": repo.list_files,
            "read_file": repo.read_file,
            "search_code": repo.search_code,
            "git_log": repo.git_log,
            "tracked_ignored_files": repo.tracked_ignored_files,
            "inspect_workflows": repo.inspect_workflows,
        }

    @property
    def specs(self) -> list[dict[str, Any]]:
        return TOOL_SPECS

    def call(self, name: str, arguments: dict[str, Any] | None) -> tuple[str, bool]:
        """Execute a tool. Returns ``(result_text, is_error)``."""
        fn = self._dispatch.get(name)
        if fn is None:
            return f"Unknown tool {name!r}. Available: {', '.join(self._dispatch)}.", True
        args = arguments if isinstance(arguments, dict) else {}
        try:
            return fn(**args), False
        except ToolError as exc:
            return f"Tool error: {exc}", True
        except TypeError as exc:
            return f"Tool error: bad arguments for {name}: {exc}", True
        except Exception as exc:  # defensive: a tool bug must not crash the audit
            return f"Tool error: unexpected failure in {name}: {type(exc).__name__}: {exc}", True
