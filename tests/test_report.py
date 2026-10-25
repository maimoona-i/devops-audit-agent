import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report import (  # noqa: E402
    ReportError,
    build_report,
    normalize_violations,
    validate_report,
    write_report,
)

FAKE_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"


def finding(**over):
    base = {"rule_id": "SEC-001", "category": "security", "severity": "critical", "file": "config.py",
            "line": 2, "description": "Hardcoded API credential", "evidence": "Credential assigned in source",
            "recommendation": "Use environment variables"}
    base.update(over)
    return base


def test_valid_finding_is_kept_with_all_fields():
    out = normalize_violations([finding()])
    assert len(out) == 1
    assert set(out[0]) == {"rule_id", "category", "severity", "file", "line", "description", "evidence",
                           "recommendation"}


def test_severity_and_category_aliases_are_normalised():
    out = normalize_violations([finding(rule_id="CI-001", severity="Moderate", category="GitHub Actions")])
    assert out[0]["severity"] == "medium" and out[0]["category"] == "ci_cd"
    assert normalize_violations([finding(severity="???")])[0]["severity"] == "medium"


def test_findings_without_evidence_are_dropped():
    assert normalize_violations([finding(evidence=""), finding(description="   ")]) == []


def test_duplicates_are_removed():
    out = normalize_violations([finding(), finding(description="Same issue worded differently"),
                                finding(rule_id="SEC-002", file="other.py")])
    assert [v["file"] for v in out] == ["config.py", "other.py"]


def test_secrets_are_scrubbed_from_every_field():
    out = normalize_violations([finding(evidence=f"key is {FAKE_KEY}", description=f"leak {FAKE_KEY}",
                                        recommendation=f"rotate {FAKE_KEY}")])
    assert FAKE_KEY not in json.dumps(out)


def test_null_file_and_line_stay_null():
    out = normalize_violations([finding(file=None, line=None), finding(rule_id="REPO-001", file="null", line="n/a")])
    assert all(v["file"] is None and v["line"] is None for v in out)


def test_line_without_file_is_cleared_and_bad_lines_become_null():
    out = normalize_violations([finding(file=None, line=7), finding(rule_id="X-1", line="abc"),
                                finding(rule_id="X-2", line=-3)])
    assert all(v["line"] is None for v in out)


def test_line_beyond_end_of_file_is_cleared(tmp_path):
    (tmp_path / "config.py").write_text("a\nb\n")
    out = normalize_violations([finding(line=2), finding(rule_id="SEC-009", file="config.py", line=99)], root=tmp_path)
    assert {v["rule_id"]: v["line"] for v in out} == {"SEC-001": 2, "SEC-009": None}


def test_missing_rule_id_gets_generated_id():
    out = normalize_violations([finding(rule_id="")])
    assert out[0]["rule_id"] == "SEC-901"


def test_path_is_normalised():
    assert normalize_violations([finding(file=".\\src\\config.py")])[0]["file"] == "src/config.py"


def test_sorted_by_severity():
    out = normalize_violations([finding(rule_id="A-1", severity="low", file="a"),
                                finding(rule_id="B-1", severity="critical", file="b")])
    assert [v["severity"] for v in out] == ["critical", "low"]


@pytest.mark.parametrize("bad", [None, "nonsense", {"a": 1}, 5])
def test_non_list_violations_rejected(bad):
    with pytest.raises(ReportError):
        normalize_violations(bad)


def test_json_string_of_violations_is_accepted():
    assert len(normalize_violations(json.dumps([finding()]))) == 1


def test_validate_report_detects_problems():
    assert validate_report(build_report("https://github.com/a/b", [])) == []
    assert validate_report({"repository": "x"})
    bad = {"repository": "x", "violations": [{"rule_id": "A", "severity": "urgent"}]}
    errors = validate_report(bad)
    assert any("severity" in e for e in errors) and any("missing" in e for e in errors)


def test_write_report_empty_violations_is_valid_json(tmp_path):
    target = tmp_path / "out" / "report.json"
    write_report(target, build_report("https://github.com/a/b", []))
    assert json.loads(target.read_text()) == {"repository": "https://github.com/a/b", "violations": []}
    assert not list(target.parent.glob(".report-*"))


def test_write_report_refuses_invalid_and_leaves_no_file(tmp_path):
    target = tmp_path / "report.json"
    with pytest.raises(ReportError):
        write_report(target, {"repository": "x", "violations": [{"oops": 1}]})
    assert not target.exists()


@pytest.mark.parametrize("raw,expected", [
    ("SEC-001SEVERITY", "SEC-001"), ("sec-1", "SEC-001"), ("CI-002 (high)", "CI-002"),
    ("SEC001", "SEC-001"), ("REPO-003\nSEVERITY", "REPO-003"),
])
def test_rule_id_is_extracted_from_noisy_text(raw, expected):
    assert normalize_violations([finding(rule_id=raw)])[0]["rule_id"] == expected


def test_rule_prefix_decides_category():
    out = normalize_violations([finding(rule_id="SEC-002", category="ci_cd", file="a"),
                                finding(rule_id="CI-002", category="security", file="b"),
                                finding(rule_id="ZZZ-100", category="testing", file="c")])
    assert {v["rule_id"]: v["category"] for v in out} == {"SEC-002": "security", "CI-002": "ci_cd",
                                                          "ZZZ-100": "testing"}
