#!/usr/bin/env python3
"""DevOps Repository Auditing Agent - command line entry point.

    python agent.py --repo https://github.com/<owner>/<repo> --output report.json

Runs unattended. Exit codes: 0 = report written, 1 = unexpected error, 2 = fatal error
(bad input, clone failure, missing credentials, LLM failure) with a message on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from auditor import AgentError, AuditAgent, stderr_log
from llm import LLMError, make_llm
from report import ReportError, build_report, write_report
from tools import RepoError, RepoWorkspace, ToolBox, parse_github_url, redact

EXIT_OK, EXIT_UNEXPECTED, EXIT_FATAL = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent.py", description="Audit a public GitHub repository for DevOps "
                                                             "best-practice violations using an LLM-driven agent.")
    p.add_argument("--repo", required=True, help="public GitHub repository URL, e.g. https://github.com/o/r")
    p.add_argument("--output", required=True, help="path of the JSON report to write")
    p.add_argument("--provider", choices=["anthropic", "openai"], help="LLM provider (default: from env)")
    p.add_argument("--model", help="model name (default: LLM_MODEL env var or a provider default)")
    p.add_argument("--max-steps", type=int, default=30, help="maximum agent steps (default 30)")
    p.add_argument("--max-seconds", type=float, default=600.0, help="wall-clock budget for the investigation")
    p.add_argument("--trace", help="where to write the tool-call trace (default: <output>.trace.json)")
    p.add_argument("--quiet", action="store_true", help="do not print progress to stderr")
    return p


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _fail(message: str, code: int = EXIT_FATAL) -> int:
    print(f"error: {redact(message)}", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    if args.max_steps < 2:
        return _fail("--max-steps must be at least 2")

    try:
        owner, repo_name = parse_github_url(args.repo)
        repo_url = f"https://github.com/{owner}/{repo_name}"
        llm = make_llm(provider=args.provider, model=args.model)
        if not args.quiet:
            print(f"Auditing {repo_url} ...", file=sys.stderr)
        with RepoWorkspace.from_url(repo_url) as workspace:
            agent = AuditAgent(llm, ToolBox(workspace), repo_url, max_steps=args.max_steps,
                               max_seconds=args.max_seconds, log=None if args.quiet else stderr_log)
            violations = agent.run()

        write_report(args.output, build_report(repo_url, violations))
        trace_path = Path(args.trace) if args.trace else Path(args.output).with_suffix(".trace.json")
        trace_path.write_text(json.dumps({"repository": repo_url, "stats": agent.stats(),
                                          "summary": agent.summary, "steps": agent.trace},
                                         indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"Done: {len(violations)} violation(s) written to {args.output}", file=sys.stderr)
        return EXIT_OK
    except (RepoError, LLMError, AgentError, ReportError) as exc:
        return _fail(str(exc))
    except OSError as exc:
        return _fail(f"file error: {exc}")
    except KeyboardInterrupt:
        return _fail("interrupted", 130)
    except Exception as exc:  # last resort: still a clean message and nonzero exit
        if os.environ.get("AGENT_DEBUG"):
            raise
        return _fail(f"unexpected {type(exc).__name__}: {exc}", EXIT_UNEXPECTED)


if __name__ == "__main__":
    sys.exit(main())
