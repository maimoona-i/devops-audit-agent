"""Validation, normalisation and writing of the JSON report.

The LLM proposes findings; this module is the safety net that guarantees the output
contract (all fields present, allowed severities, null for inapplicable file/line),
removes duplicates and unsupported claims, and scrubs secrets one last time.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from tools import redact

SEVERITIES = ("critical", "high", "medium", "low")
_SEV_ORDER = {s: i for i, s in enumerate(SEVERITIES)}
FIELDS = ("rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation")

CATEGORIES = ("repository_hygiene", "dependency_management", "code_quality", "testing", "ci_cd",
              "security", "configuration", "collaboration", "build_release")
_CATEGORY_ALIASES = {
    "ci": "ci_cd", "cicd": "ci_cd", "ci_cd": "ci_cd", "github_actions": "ci_cd", "actions": "ci_cd",
    "ci_cd_pipeline": "ci_cd", "workflow": "ci_cd", "workflows": "ci_cd",
    "hygiene": "repository_hygiene", "repository": "repository_hygiene", "repo_hygiene": "repository_hygiene",
    "dependencies": "dependency_management", "dependency": "dependency_management",
    "quality": "code_quality", "lint": "code_quality", "linting": "code_quality",
    "tests": "testing", "test": "testing",
    "build": "build_release", "release": "build_release", "build_and_release": "build_release",
    "build_and_deploy": "build_release", "deployment": "build_release",
    "config": "configuration", "collab": "collaboration",
    "secrets": "security",
}
_PREFIX = {"security": "SEC", "repository_hygiene": "REPO", "dependency_management": "DEP",
           "code_quality": "QUAL", "testing": "TEST", "ci_cd": "CI", "configuration": "CFG",
           "collaboration": "COLL", "build_release": "BUILD"}
_SEV_ALIASES = {"moderate": "medium", "major": "high", "minor": "low", "info": "low", "informational": "low",
                "warning": "medium", "error": "high", "severe": "critical", "blocker": "critical"}
_NULLISH = {"", "none", "null", "n/a", "na", "-", "unknown"}
MAX_LINECOUNT_BYTES = 2_000_000


class ReportError(Exception):
    """The proposed report is unusable; the message is returned to the LLM so it can retry."""


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _norm_category(value: Any) -> str:
    slug = _slug(str(value or ""))
    if slug in CATEGORIES:
        return slug
    return _CATEGORY_ALIASES.get(slug, slug or "configuration")


def _norm_severity(value: Any) -> str:
    sev = str(value or "").strip().lower()
    sev = _SEV_ALIASES.get(sev, sev)
    return sev if sev in SEVERITIES else "medium"


def _norm_file(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("\\", "/")
    if text.lower() in _NULLISH:
        return None
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/") or None


def _norm_line(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        n = int(str(value).strip().split("-")[0].split(":")[0])
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _line_count(root: Path | None, rel: str) -> int | None:
    """Number of lines in ``rel`` if it is a regular text file inside ``root``."""
    if root is None:
        return None
    try:
        path = (root / rel).resolve()
        path.relative_to(root.resolve())
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LINECOUNT_BYTES:
            return None
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except (OSError, ValueError):
        return None


def _text(value: Any) -> str:
    return redact(" ".join(str(value or "").split()))


def normalize_violations(raw: Any, root: Path | None = None) -> list[dict[str, Any]]:
    """Clean the model's findings. Raises :class:`ReportError` if ``raw`` is not a list."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise ReportError("'violations' must be an array of objects, not a string.") from None
    if not isinstance(raw, list):
        raise ReportError("'violations' must be an array (use [] when nothing is wrong).")

    cleaned: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    counters: dict[str, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        description = _text(item.get("description"))
        evidence = _text(item.get("evidence"))
        if not description or not evidence:
            continue  # unsupported claim: no evidence, no finding
        category = _norm_category(item.get("category"))
        file = _norm_file(item.get("file"))
        line = _norm_line(item.get("line")) if file else None
        if file and line is not None:
            total = _line_count(root, file)
            if total is not None and line > total:
                line = None  # line number does not exist in that file
        rule_id = re.sub(r"[^A-Z0-9\-]", "", str(item.get("rule_id") or "").strip().upper())
        if not rule_id:
            prefix = _PREFIX.get(category, "GEN")
            counters[prefix] = counters.get(prefix, 0) + 1
            rule_id = f"{prefix}-9{counters[prefix]:02d}"
        key1 = (rule_id, file, line)
        key2 = (file, line, category, description.lower())
        if key1 in seen or key2 in seen:
            continue
        seen.update((key1, key2))
        cleaned.append({
            "rule_id": rule_id,
            "category": category,
            "severity": _norm_severity(item.get("severity")),
            "file": redact(file) if file else None,
            "line": line,
            "description": description,
            "evidence": evidence,
            "recommendation": _text(item.get("recommendation")) or "Review and remediate this issue.",
        })
    cleaned.sort(key=lambda v: (_SEV_ORDER[v["severity"]], v["rule_id"], v["file"] or ""))
    return cleaned


def build_report(repository: str, violations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"repository": repository, "violations": violations}


def validate_report(report: Any) -> list[str]:
    """Return a list of contract violations (empty list means the report is valid)."""
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]
    if not isinstance(report.get("repository"), str) or not report["repository"]:
        errors.append("'repository' must be a non-empty string")
    violations = report.get("violations")
    if not isinstance(violations, list):
        return errors + ["'violations' must be an array"]
    for i, v in enumerate(violations):
        if not isinstance(v, dict):
            errors.append(f"violations[{i}] must be an object")
            continue
        for name in FIELDS:
            if name not in v:
                errors.append(f"violations[{i}] is missing '{name}'")
        if v.get("severity") not in SEVERITIES:
            errors.append(f"violations[{i}].severity must be one of {SEVERITIES}")
        if v.get("file") is not None and not isinstance(v.get("file"), str):
            errors.append(f"violations[{i}].file must be a string or null")
        line = v.get("line")
        if line is not None and (isinstance(line, bool) or not isinstance(line, int) or line < 1):
            errors.append(f"violations[{i}].line must be a positive integer or null")
        for name in ("rule_id", "category", "description", "evidence", "recommendation"):
            if name in v and (not isinstance(v[name], str) or not v[name].strip()):
                errors.append(f"violations[{i}].{name} must be a non-empty string")
    return errors


def write_report(path: str | os.PathLike[str], report: dict[str, Any]) -> None:
    """Validate and write the report atomically (never leaves a half-written file)."""
    errors = validate_report(report)
    if errors:
        raise ReportError("Refusing to write an invalid report: " + "; ".join(errors[:5]))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".report-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
