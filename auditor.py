"""The agent loop: Observe -> Reason -> Select tool -> Execute tool -> Interpret -> Report.

All control flow is plain Python. The LLM decides *which* tool to call next from what it has
observed so far; this module executes the call, feeds the (secret-redacted) result back, enforces
step/time budgets, and validates the final submission.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from typing import Any, Callable

from llm import LLMClient
from prompts import SUBMIT_TOOL, SYSTEM_PROMPT, initial_user_message
from report import ReportError, normalize_violations
from tools import ToolBox, redact

SUBMIT = SUBMIT_TOOL["name"]
KEEP_FULL_RESULTS = 6          # most recent tool results kept in full; older ones are trimmed
TRIMMED_CHARS = 1200
MAX_NUDGES = 3
MAX_BAD_SUBMITS = 3


class AgentError(Exception):
    """The agent could not complete the inspection."""


def stderr_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _short(value: Any, limit: int = 160) -> str:
    text = redact(json.dumps(value, default=str, ensure_ascii=False))
    return text if len(text) <= limit else text[: limit - 3] + "..."


class AuditAgent:
    def __init__(self, llm: LLMClient, toolbox: ToolBox, repo_url: str, max_steps: int = 30,
                 max_seconds: float = 600.0, log: Callable[[str], None] | None = stderr_log):
        self.llm = llm
        self.toolbox = toolbox
        self.repo_url = repo_url
        self.max_steps = max(2, max_steps)
        self.max_seconds = max_seconds
        self._log = log or (lambda _msg: None)
        self.trace: list[dict[str, Any]] = []
        self.summary: str = ""

    # ------------------------------------------------------------------ public
    def run(self) -> list[dict[str, Any]]:
        """Run the investigation and return validated violations."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": initial_user_message(self.repo_url)}]
        specs = self.toolbox.specs + [SUBMIT_TOOL]
        started = time.monotonic()
        nudges = bad_submits = 0
        forced = False
        step = 0

        while True:
            step += 1
            over_budget = step >= self.max_steps or (time.monotonic() - started) > self.max_seconds
            force_tool = None
            if over_budget:
                if step > self.max_steps + MAX_BAD_SUBMITS + 1:
                    raise AgentError("The model did not submit a report within the step budget.")
                if not forced:
                    forced = True
                    self._log(f"[budget] step/time budget reached at step {step}; requesting final report")
                    self._add_note(messages, "Budget exhausted. Do not call any more inspection tools. Call "
                                             "submit_report now with the verified findings you already have.")
                force_tool = SUBMIT

            self._trim_old_results(messages)
            response = self.llm.complete(SYSTEM_PROMPT, messages, specs, force_tool=force_tool)

            messages.append({
                "role": "assistant",
                "text": response.text,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            })
            if response.text:
                thought = redact(response.text)
                self._log(f"[step {step}] thought: {thought[:300]}")
                self.trace.append({"step": step, "type": "thought", "text": thought[:1000]})

            if not response.tool_calls:
                nudges += 1
                if nudges > MAX_NUDGES:
                    raise AgentError("The model stopped without calling a tool or submitting a report.")
                messages.append({"role": "user", "content": "Continue: call an inspection tool to gather evidence, "
                                                            "or call submit_report if you are done."})
                continue

            results: list[dict[str, Any]] = []
            for call in response.tool_calls:
                if call.name == SUBMIT:
                    try:
                        violations = self._finalize(call.arguments, step)
                    except ReportError as exc:
                        bad_submits += 1
                        if bad_submits >= MAX_BAD_SUBMITS:
                            raise AgentError(f"The model repeatedly submitted an invalid report: {exc}") from None
                        self._log(f"[step {step}] submit_report rejected: {exc}")
                        results.append({"id": call.id, "name": call.name, "is_error": True,
                                        "content": f"submit_report rejected: {exc} Fix it and call submit_report again."})
                        continue
                    return violations
                results.append(self._execute(call, step))
            messages.append({"role": "tool_results", "results": results})

    def stats(self) -> dict[str, Any]:
        calls = [e["tool"] for e in self.trace if e["type"] == "tool_call"]
        return {"tool_calls": len(calls), "tools_used": dict(Counter(calls))}

    # ----------------------------------------------------------------- helpers
    def _execute(self, call: Any, step: int) -> dict[str, Any]:
        self._log(f"[step {step}] tool: {call.name}({_short(call.arguments)})")
        text, is_error = self.toolbox.call(call.name, call.arguments)
        self.trace.append({"step": step, "type": "tool_call", "tool": call.name,
                           "arguments": json.loads(_short_json(call.arguments)), "is_error": is_error,
                           "result_chars": len(text), "result_preview": text[:300]})
        return {"id": call.id, "name": call.name, "content": text, "is_error": is_error}

    def _finalize(self, arguments: dict[str, Any], step: int) -> list[dict[str, Any]]:
        violations = normalize_violations(arguments.get("violations"), root=self.toolbox.repo.root)
        self.summary = redact(str(arguments.get("summary") or ""))[:500]
        self.trace.append({"step": step, "type": "submit_report", "violations": len(violations),
                           "summary": self.summary})
        self._log(f"[step {step}] submit_report: {len(violations)} violation(s)")
        return violations

    @staticmethod
    def _add_note(messages: list[dict[str, Any]], note: str) -> None:
        """Attach a message to the next model turn without breaking role alternation."""
        if messages and messages[-1]["role"] == "tool_results":
            messages[-1]["note"] = note
        elif messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += "\n\n" + note
        else:
            messages.append({"role": "user", "content": note})

    @staticmethod
    def _trim_old_results(messages: list[dict[str, Any]]) -> None:
        """Keep the context window bounded: shorten tool output older than the last few results."""
        indexes = [i for i, m in enumerate(messages) if m["role"] == "tool_results"]
        for i in indexes[:-KEEP_FULL_RESULTS]:
            for r in messages[i]["results"]:
                if len(r["content"]) > TRIMMED_CHARS and not r.get("trimmed"):
                    r["content"] = r["content"][:TRIMMED_CHARS] + "\n...[older output trimmed to save context]"
                    r["trimmed"] = True


def _short_json(value: Any) -> str:
    """JSON text of tool arguments with secrets masked, for the trace file."""
    return redact(json.dumps(value if isinstance(value, dict) else {}, default=str, ensure_ascii=False))
