#!/usr/bin/env python3
"""Batch code review baseline generation across SWE-CARE instances.

Runs PR-Agent, CodeRabbit, and/or Devin on each instance and saves raw outputs
plus parsed findings. Follows the two-phase pattern from run_batch_testgen.py:
Phase 1 clones repos sequentially, Phase 2 processes instances in parallel.

Usage:
  python run_batch_baselines.py --split dev                       # All dev instances
  python run_batch_baselines.py --split dev --repo fastapi/fastapi --limit 5
  python run_batch_baselines.py --split dev --tools pr-agent      # PR-Agent only
  python run_batch_baselines.py --split dev --tools devin         # Devin only
  python run_batch_baselines.py --split dev --workers 4           # Parallel
  python run_batch_baselines.py --no-resume                       # Re-process all
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from pipeline import dataset_utils, repo_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOOL_CHOICES = ["pr-agent", "coderabbit", "devin", "claude-code", "codex"]

# ANSI escape code pattern for stripping color codes from tool output
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_pr_agent_output(raw: str) -> list[dict]:
    """Extract findings from PR-Agent review output.

    Looks for the YAML block after the "AI response:" marker and parses
    the key_issues_to_review list.
    """
    text = strip_ansi(raw)

    # Find the YAML block after "AI response:"
    marker = "AI response:\n"
    idx = text.find(marker)
    if idx < 0:
        return []

    yaml_text = text[idx + len(marker):]

    # The YAML block ends when we hit a line starting with a timestamp or EOF
    lines = yaml_text.split("\n")
    yaml_lines = []
    for line in lines:
        # PR-Agent log lines start with timestamps like "2026-02-25 12:30:..."
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:", line):
            break
        yaml_lines.append(line)

    yaml_str = "\n".join(yaml_lines).strip()
    if not yaml_str:
        return []

    try:
        data = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        logger.warning("Failed to parse PR-Agent YAML output")
        return []

    if not isinstance(data, dict):
        return []

    review = data.get("review", data)
    issues = review.get("key_issues_to_review", [])
    if not isinstance(issues, list):
        return []

    findings = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        findings.append({
            "file": str(issue.get("relevant_file", "")).strip(),
            "issue_header": str(issue.get("issue_header", "")).strip(),
            "issue_content": str(issue.get("issue_content", "")).strip(),
            "start_line": issue.get("start_line"),
            "end_line": issue.get("end_line"),
        })

    return findings


def parse_coderabbit_output(raw: str) -> list[dict]:
    """Extract findings from CodeRabbit plain-text review output.

    Parses blocks delimited by ============ lines containing File/Line/Type/Comment.
    """
    text = strip_ansi(raw)
    findings = []

    # Split on separator lines
    blocks = re.split(r"={10,}", text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Look for structured fields
        file_m = re.search(r"^File:\s*(.+)$", block, re.MULTILINE)
        line_m = re.search(r"^Line:\s*(\d+)", block, re.MULTILINE)
        type_m = re.search(r"^Type:\s*(.+)$", block, re.MULTILINE)
        comment_m = re.search(
            r"^Comment:\s*\n(.*?)(?=\n✏️|\n🐛|\nPrompt for AI Agent:|\n🧹|\Z)",
            block, re.MULTILINE | re.DOTALL,
        )

        if not file_m:
            continue

        # Extract proposed fix if present (various emoji prefixes)
        fix_m = re.search(
            r"(?:✏️|🐛|🧹) Proposed fix[^\n]*\n(.*?)(?=\nPrompt for AI Agent:|\Z)",
            block, re.DOTALL,
        )

        findings.append({
            "file": file_m.group(1).strip(),
            "line": int(line_m.group(1)) if line_m else None,
            "type": type_m.group(1).strip() if type_m else None,
            "comment": comment_m.group(1).strip() if comment_m else "",
            "proposed_fix": fix_m.group(1).strip() if fix_m else None,
        })

    return findings


def parse_devin_output(raw: str) -> list[dict]:
    """Extract findings from Devin code review output.

    Parses the structured_output JSON (if present) for review findings,
    falling back to extracting issues from session messages.
    """
    # Try to extract structured output block first
    # The raw output file contains a JSON section with structured_output
    structured_marker = "=== STRUCTURED OUTPUT ===\n"
    idx = raw.find(structured_marker)
    if idx >= 0:
        json_text = raw[idx + len(structured_marker):].strip()
        # Find the end of the JSON block
        end_marker = "\n=== "
        end_idx = json_text.find(end_marker)
        if end_idx >= 0:
            json_text = json_text[:end_idx].strip()
        try:
            data = json.loads(json_text)
            if isinstance(data, dict):
                issues = data.get("issues", [])
                if isinstance(issues, list):
                    findings = []
                    for issue in issues:
                        if not isinstance(issue, dict):
                            continue
                        findings.append({
                            "file": str(issue.get("file", "")).strip(),
                            "line": issue.get("line"),
                            "type": str(issue.get("type", "")).strip(),
                            "description": str(issue.get("description", "")).strip(),
                        })
                    return findings
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: extract findings from messages section
    messages_marker = "=== MESSAGES ===\n"
    idx = raw.find(messages_marker)
    if idx < 0:
        return []

    messages_text = raw[idx + len(messages_marker):]
    findings = []

    # Look for file-referencing review comments in messages
    # Pattern: file paths with line references and descriptions
    for m in re.finditer(
        r"(?:^|\n)\s*[-•*]\s*\*?\*?`?([^\n`*]+\.\w{1,5})`?\*?\*?"
        r"(?:\s*(?:line|L)\s*(\d+))?\s*[:—–-]\s*([^\n]+)",
        messages_text,
    ):
        findings.append({
            "file": m.group(1).strip(),
            "line": int(m.group(2)) if m.group(2) else None,
            "type": None,
            "description": m.group(3).strip(),
        })

    return findings


def parse_claude_code_output(raw: str) -> list[dict]:
    """Extract findings from Claude Code JSON output.

    Expects the ``--output-format json`` wrapper with a ``result`` field
    containing either raw JSON or a markdown-fenced JSON block with an
    ``issues`` array.
    """
    # The --output-format json wrapper looks like:
    # {"type":"result","subtype":"success","cost_usd":...,"result":"..."}
    text = raw
    try:
        wrapper = json.loads(raw)
        if isinstance(wrapper, dict) and "result" in wrapper:
            text = wrapper["result"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to extract a JSON object from the text (may be fenced in ```json)
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find a raw JSON object with "issues"
        json_match = re.search(r'(\{[^{}]*"issues"\s*:\s*\[.*?\]\s*\})', text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse Claude Code JSON output")
        return []

    if not isinstance(data, dict):
        return []

    issues = data.get("issues", [])
    if not isinstance(issues, list):
        return []

    findings = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        findings.append({
            "file": str(issue.get("file", "")).strip(),
            "line": issue.get("line"),
            "type": str(issue.get("type", "")).strip(),
            "description": str(issue.get("description", "")).strip(),
        })

    return findings


# ---------------------------------------------------------------------------
# Tool runners
# ---------------------------------------------------------------------------

def _setup_pr_agent_branches(workdir: Path, instance: dict) -> str:
    """Set up local branches for PR-Agent's LocalGitProvider.

    Creates a 'target' branch at base_commit and a 'pr-branch' checked out
    at head_commit.  LocalGitProvider diffs HEAD (pr-branch) against the
    merge-base of HEAD and target, which yields the correct patch_to_review.

    Returns the target branch name.
    """
    base = instance["base_commit"]
    head = instance["commit_to_review"]["head_commit"]

    def _git(args: list[str]) -> None:
        subprocess.run(
            ["git"] + args, cwd=workdir,
            check=True, capture_output=True, text=True,
        )

    # Create target branch at base_commit
    _git(["branch", "-f", "target", base])
    # Create and checkout pr-branch at head_commit
    _git(["checkout", "-B", "pr-branch", head])

    return "target"


def run_pr_agent(
    instance: dict,
    workdir: Path,
    output_dir: Path,
    timeout: int,
) -> dict:
    """Run PR-Agent review on a single instance using LocalGitProvider.

    Uses local branches so PR-Agent reviews the correct diff (patch_to_review)
    rather than the full merged PR from GitHub.
    """
    target_branch = _setup_pr_agent_branches(workdir, instance)
    problem_statement = instance.get("problem_statement", "")

    env = os.environ.copy()
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    env["OPENAI__KEY"] = openai_key
    env["OPENAI_API_KEY"] = openai_key
    env["CONFIG__GIT_PROVIDER"] = "local"

    cmd = [
        "pr-agent",
        f"--pr_url={target_branch}",
        "review",
        "--config.publish_output=false",
        "--config.verbosity_level=2",
    ]
    if problem_statement:
        cmd.append(f"--pr_reviewer.extra_instructions=Problem statement: {problem_statement}")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=workdir,
            capture_output=True, text=True, timeout=timeout,
        )
        raw = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        raw = f"TIMEOUT after {timeout}s"
    elapsed = time.time() - t0

    # Save raw output
    out_file = output_dir / "pr-agent.txt"
    out_file.write_text(raw)

    # Parse findings
    findings = parse_pr_agent_output(raw)

    return {
        "success": "AI response:" in strip_ansi(raw),
        "num_findings": len(findings),
        "findings": findings,
        "raw_output_file": "pr-agent.txt",
        "elapsed_seconds": round(elapsed, 1),
        "error": None if "AI response:" in strip_ansi(raw) else raw[-500:],
    }


def run_coderabbit(
    instance: dict,
    workdir: Path,
    output_dir: Path,
    timeout: int,
) -> dict:
    """Run CodeRabbit review on a single instance.

    Requires a local checkout at the head commit.  Uses the merge-base of
    base_commit and head_commit as --base so CodeRabbit reviews exactly
    the patch_to_review (PR's own changes only, excluding upstream noise).
    """
    base_commit = instance["base_commit"]
    head_commit = instance["commit_to_review"]["head_commit"]
    problem_statement = instance.get("problem_statement", "")

    # Checkout head commit in workdir
    repo_manager.checkout_commit(workdir, head_commit)

    # Compute merge-base for correct diff
    mb_result = subprocess.run(
        ["git", "merge-base", base_commit, head_commit],
        cwd=workdir, capture_output=True, text=True,
    )
    merge_base = mb_result.stdout.strip() if mb_result.returncode == 0 else base_commit

    cmd = [
        "coderabbit", "review",
        "--base", merge_base,
        "--plain",
    ]

    # Pass problem statement as additional instructions via a temp file
    # Write it outside the workdir so CodeRabbit doesn't review it as part of the diff
    instructions_file = None
    if problem_statement:
        instructions_file = output_dir / ".review_instructions.md"
        instructions_file.write_text(
            f"# Problem Statement / PR Description\n\n{problem_statement}\n"
        )
        cmd.extend(["--config", str(instructions_file)])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout,
        )
        raw = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        raw = f"TIMEOUT after {timeout}s"
    elapsed = time.time() - t0

    # Save raw output
    out_file = output_dir / "coderabbit.txt"
    out_file.write_text(raw)

    # Parse findings
    findings = parse_coderabbit_output(raw)
    success = "Review completed" in raw or len(findings) > 0

    return {
        "success": success,
        "num_findings": len(findings),
        "findings": findings,
        "raw_output_file": "coderabbit.txt",
        "elapsed_seconds": round(elapsed, 1),
        "error": None if success else raw[-500:],
    }


def run_claude_code(
    instance: dict,
    workdir: Path,
    output_dir: Path,
    timeout: int,
) -> dict:
    """Run Claude Code review on a single instance.

    Uses the ``claude`` CLI in print mode.  The workdir should be a git
    clone containing both *base_commit* and *head_commit* so that Claude
    can compute the merge-base and diff locally.
    """
    base_commit = instance["base_commit"]
    head_commit = instance["commit_to_review"]["head_commit"]
    repo = instance["repo"]
    problem_statement = instance.get("problem_statement", "")

    prompt_parts = [
        f"Review the following code changes from the repository {repo}.",
        "",
        f"The pre-change code is at the merge-base of commits {base_commit} and {head_commit}.",
        f"The post-change code is at commit {head_commit}.",
        f"Run `git diff $(git merge-base {base_commit} {head_commit}) {head_commit}` to see the diff you should review.",
        "IMPORTANT: Only review the diff between the merge-base and the head commit above. Do NOT compare against the current main/develop branch — this is a historical PR and the repo has changed since. Do NOT look at any pull request pages or existing review comments.",
        "Do not make any commits or pushes.",
        "",
        'Return your findings as a JSON object with this exact format: {"issues": [{"file": "<path>", "line": <int_or_null>, "type": "<bug|warning|suggestion|info>", "description": "<details>"}]}',
    ]
    if problem_statement:
        prompt_parts.append(
            f"\n<problem_statement>\n{problem_statement}\n</problem_statement>"
        )
    prompt = "\n".join(prompt_parts)

    system_prompt = (
        "You are a code reviewer. You have access to a local git checkout. "
        "Use the tools available to inspect the diff and source files, then "
        "return your findings as JSON."
    )

    cmd = [
        "claude",
        "-p",
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--model", "sonnet",
        "--system-prompt", system_prompt,
        prompt,
    ]

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=workdir,
            capture_output=True, text=True, timeout=timeout,
        )
        raw = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        raw = f"TIMEOUT after {timeout}s"
    elapsed = time.time() - t0

    # Save raw output
    out_file = output_dir / "claude-code.txt"
    out_file.write_text(raw)

    # Parse findings
    findings = parse_claude_code_output(raw)
    success = len(findings) > 0

    return {
        "success": success,
        "num_findings": len(findings),
        "findings": findings,
        "raw_output_file": "claude-code.txt",
        "elapsed_seconds": round(elapsed, 1),
        "error": None if success else raw[-500:],
    }


def parse_codex_output(raw: str) -> list[dict]:
    """Extract findings from Codex CLI output.

    When run with ``--json``, Codex emits newline-delimited JSON events.
    Each event has a ``type`` field.  The final review lives in an
    ``item.completed`` event whose ``item.type`` is ``agent_message``
    with the text in ``item.text``.
    """
    # Try to find the last agent_message from item.completed events
    text = raw

    for line in reversed(raw.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            # Codex NDJSON format: {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
            if event.get("type") == "item.completed":
                item = event.get("item", {})
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    item_text = item.get("text", "")
                    if item_text and "issues" in item_text:
                        text = item_text
                        break
        except (json.JSONDecodeError, TypeError):
            continue

    # Extract JSON object with issues array
    # First try direct parse (Codex often outputs pure JSON)
    data = None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    if data is None:
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'(\{"issues"\s*:\s*\[.*\]\s*\})', text, re.DOTALL)
            json_str = json_match.group(1) if json_match else text
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse Codex JSON output")
            return []

    if not isinstance(data, dict):
        return []

    issues = data.get("issues", [])
    if not isinstance(issues, list):
        return []

    findings = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        findings.append({
            "file": str(issue.get("file", "")).strip(),
            "line": issue.get("line"),
            "type": str(issue.get("type", "")).strip(),
            "description": str(issue.get("description", "")).strip(),
        })

    return findings


def run_codex(
    instance: dict,
    workdir: Path,
    output_dir: Path,
    timeout: int,
) -> dict:
    """Run OpenAI Codex review on a single instance.

    Uses the ``codex`` CLI in exec mode with ``--json`` output.
    The workdir should be a git clone containing both commits so Codex
    can compute the merge-base and diff locally.
    """
    base_commit = instance["base_commit"]
    head_commit = instance["commit_to_review"]["head_commit"]
    repo = instance["repo"]
    problem_statement = instance.get("problem_statement", "")

    prompt_parts = [
        f"Review the following code changes from the repository {repo}.",
        "",
        f"The pre-change code is at the merge-base of commits {base_commit} and {head_commit}.",
        f"The post-change code is at commit {head_commit}.",
        f"Run `git diff $(git merge-base {base_commit} {head_commit}) {head_commit}` to see the diff you should review.",
        "IMPORTANT: Only review the diff between the merge-base and the head commit above. Do NOT compare against the current main/develop branch — this is a historical PR and the repo has changed since. Do NOT look at any pull request pages or existing review comments.",
        "Do not make any commits or pushes.",
        "",
        'Return your findings as a JSON object with this exact format: {"issues": [{"file": "<path>", "line": <int_or_null>, "type": "<bug|warning|suggestion|info>", "description": "<details>"}]}',
    ]
    if problem_statement:
        prompt_parts.append(
            f"\n<problem_statement>\n{problem_statement}\n</problem_statement>"
        )
    prompt = "\n".join(prompt_parts)

    cmd = [
        "codex", "exec",
        "--json",
        "--full-auto",
        prompt,
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=workdir,
            capture_output=True, text=True, timeout=timeout,
        )
        raw = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        raw = f"TIMEOUT after {timeout}s"
    elapsed = time.time() - t0

    # Save raw output
    out_file = output_dir / "codex.txt"
    out_file.write_text(raw)

    # Parse findings
    findings = parse_codex_output(raw)
    success = len(findings) > 0

    return {
        "success": success,
        "num_findings": len(findings),
        "findings": findings,
        "raw_output_file": "codex.txt",
        "elapsed_seconds": round(elapsed, 1),
        "error": None if success else raw[-500:],
    }


# -- Devin API helpers -----------------------------------------------------

DEVIN_API_BASE = "https://api.devin.ai/v1"

# Structured output schema requesting Devin return review findings as JSON
_DEVIN_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "File path relative to repo root"},
                    "line": {"type": ["integer", "null"], "description": "Line number"},
                    "type": {"type": "string", "description": "Issue type: bug, warning, suggestion, or info"},
                    "description": {"type": "string", "description": "Detailed description of the issue"},
                },
                "required": ["file", "type", "description"],
            },
        },
        "approved": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["issues"],
}


def _devin_api_request(
    method: str,
    path: str,
    api_key: str,
    data: dict | None = None,
    max_retries: int = 6,
) -> dict:
    """Make an authenticated request to the Devin API and return parsed JSON.

    Retries with exponential backoff on HTTP 429 (rate limit / concurrent
    session limit).
    """
    url = f"{DEVIN_API_BASE}{path}"
    body = json.dumps(data).encode() if data else None
    backoff = 30
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                logger.warning(
                    "Devin API 429 (attempt %d/%d), retrying in %ds...",
                    attempt + 1, max_retries, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            raise


def _poll_devin_session(
    session_id: str,
    api_key: str,
    timeout: int,
    poll_interval: int = 10,
) -> dict:
    """Poll a Devin session until it reaches a terminal state or timeout.

    Returns the final session details dict.  Terminal states are
    ``finished``, ``blocked``, and ``expired``.
    """
    terminal_states = {"finished", "blocked", "expired", "stopped"}
    deadline = time.time() + timeout
    backoff = poll_interval

    while time.time() < deadline:
        try:
            session = _devin_api_request("GET", f"/sessions/{session_id}", api_key)
        except Exception as exc:
            logger.warning("Devin poll error for %s: %s", session_id, exc)
            time.sleep(min(backoff, 30))
            backoff = min(backoff * 1.5, 30)
            continue

        status = session.get("status_enum", "")
        if status in terminal_states:
            return session

        time.sleep(min(backoff, 30))
        backoff = min(backoff * 1.5, 30)

    return {"status_enum": "timeout", "error": f"Polling timed out after {timeout}s"}


def run_devin(
    instance: dict,
    output_dir: Path,
    timeout: int,
) -> dict:
    """Run Devin code review on a single instance via the Devin REST API.

    Devin works against GitHub so no local checkout is needed.

    FIXME: Devin produces low-quality output — findings are mostly change
    summaries rather than actual review comments, and line numbers reference
    the old file (from diff hunk headers) instead of the post-change code.
    """
    api_key = os.environ.get("DEVIN_API_KEY", "")
    if not api_key:
        return {
            "success": False,
            "num_findings": 0,
            "findings": [],
            "raw_output_file": "devin.txt",
            "elapsed_seconds": 0,
            "error": "DEVIN_API_KEY not set",
        }

    repo = instance["repo"]
    base_commit = instance["base_commit"]
    head_commit = instance["commit_to_review"]["head_commit"]
    problem_statement = instance.get("problem_statement", "")

    prompt_parts = [
        f"Review the following code changes from the repository {repo}.",
        "",
        f"The pre-change code is at the merge-base of commits {base_commit} and {head_commit}.",
        f"The post-change code is at commit {head_commit}.",
        f"Clone https://github.com/{repo} and run `git diff $(git merge-base {base_commit} {head_commit}) {head_commit}` to see the diff you should review.",
        "IMPORTANT: Only review the diff between the merge-base and the head commit above. Do NOT compare against the current main/develop branch — this is a historical PR and the repo has changed since. Do NOT look at any pull request pages or existing review comments.",
        "Do not make any commits or pushes.",
    ]
    if problem_statement:
        prompt_parts.append(f"\n<problem_statement>\n{problem_statement}\n</problem_statement>")
    prompt = "\n".join(prompt_parts)

    t0 = time.time()
    raw_parts: list[str] = []

    try:
        # Create a Devin session
        create_resp = _devin_api_request("POST", "/sessions", api_key, {
            "prompt": prompt,
            "idempotent": False,
            "structured_output_schema": _DEVIN_REVIEW_SCHEMA,
        })
        session_id = create_resp.get("session_id", "")
        session_url = create_resp.get("url", "")
        raw_parts.append(f"=== SESSION CREATED ===\n")
        raw_parts.append(f"session_id: {session_id}\n")
        raw_parts.append(f"url: {session_url}\n\n")

        if not session_id:
            raise RuntimeError(f"No session_id in response: {create_resp}")

        # Poll until completion
        session = _poll_devin_session(session_id, api_key, timeout)
        status = session.get("status_enum", "unknown")
        raw_parts.append(f"=== SESSION STATUS: {status} ===\n\n")

        # Capture structured output
        structured = session.get("structured_output")
        if structured:
            raw_parts.append("=== STRUCTURED OUTPUT ===\n")
            raw_parts.append(json.dumps(structured, indent=2))
            raw_parts.append("\n\n")

        # Capture messages
        messages = session.get("messages", [])
        if messages:
            raw_parts.append("=== MESSAGES ===\n")
            for msg in messages:
                origin = msg.get("origin", "")
                text = msg.get("message", "")
                ts = msg.get("timestamp", "")
                raw_parts.append(f"[{ts}] ({origin}): {text}\n")
            raw_parts.append("\n")

        # Terminate session to free the concurrent slot
        try:
            _devin_api_request("DELETE", f"/sessions/{session_id}", api_key)
            logger.info("Terminated Devin session %s", session_id)
        except Exception as exc:
            logger.debug("Could not terminate Devin session %s: %s", session_id, exc)

    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode()
        except Exception:
            pass
        raw_parts.append(f"HTTP ERROR {exc.code}: {exc.reason}\n{error_body}\n")
    except Exception as exc:
        raw_parts.append(f"ERROR: {exc}\n")

    elapsed = time.time() - t0
    raw = "".join(raw_parts)

    # Save raw output
    out_file = output_dir / "devin.txt"
    out_file.write_text(raw)

    # Parse findings
    findings = parse_devin_output(raw)
    # Devin may end as "finished" or "blocked" (when it needs input but already produced output)
    has_terminal = any(f"SESSION STATUS: {s}" in raw for s in ("finished", "blocked"))
    success = has_terminal and len(findings) > 0

    return {
        "success": success,
        "num_findings": len(findings),
        "findings": findings,
        "raw_output_file": "devin.txt",
        "elapsed_seconds": round(elapsed, 1),
        "error": None if success else raw[-500:],
    }


# ---------------------------------------------------------------------------
# Resume / result management
# ---------------------------------------------------------------------------

def load_existing_result(output_dir: Path, instance_id: str) -> dict | None:
    """Load existing result.json for an instance, or None if not found."""
    safe_id = instance_id.replace("/", "__")
    result_file = output_dir / safe_id / "result.json"
    if result_file.exists():
        try:
            return json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load result for %s: %s", instance_id, e)
    return None


def instance_needs_processing(
    existing: dict | None, requested_tools: list[str]
) -> bool:
    """Check if an instance needs processing for the requested tools."""
    if existing is None:
        return True
    tools_done = existing.get("tools", {})
    for tool in requested_tools:
        if tool not in tools_done or not tools_done[tool].get("success"):
            return True
    return False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(
    output_dir: Path,
    all_results: list[dict],
    elapsed: float,
    tools: list[str],
) -> dict:
    """Write summary.json with aggregated statistics."""
    # Per-repo aggregation
    repo_data: dict[str, dict] = {}
    for r in all_results:
        repo = r["repo"]
        if repo not in repo_data:
            repo_data[repo] = {"repo": repo, "instances": 0}
            for tool in TOOL_CHOICES:
                repo_data[repo][f"{tool}_findings"] = 0
                repo_data[repo][f"{tool}_successes"] = 0
                repo_data[repo][f"{tool}_errors"] = 0

        rd = repo_data[repo]
        rd["instances"] += 1
        for tool in TOOL_CHOICES:
            tool_result = r.get("tools", {}).get(tool)
            if tool_result:
                rd[f"{tool}_findings"] += tool_result.get("num_findings", 0)
                if tool_result.get("success"):
                    rd[f"{tool}_successes"] += 1
                else:
                    rd[f"{tool}_errors"] += 1

    # Totals
    totals: dict[str, dict] = {}
    for tool in tools:
        tool_results = [
            r.get("tools", {}).get(tool, {}) for r in all_results
            if tool in r.get("tools", {})
        ]
        totals[tool] = {
            "instances_run": len(tool_results),
            "successes": sum(1 for t in tool_results if t.get("success")),
            "errors": sum(1 for t in tool_results if not t.get("success")),
            "total_findings": sum(t.get("num_findings", 0) for t in tool_results),
            "avg_elapsed": (
                round(
                    sum(t.get("elapsed_seconds", 0) for t in tool_results)
                    / len(tool_results),
                    1,
                )
                if tool_results else 0
            ),
        }

    summary = {
        "total_instances": len(all_results),
        "tools_run": tools,
        "elapsed_seconds": round(elapsed, 1),
        "tool_totals": totals,
        "repo_summary": list(repo_data.values()),
        "instance_results": [
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "tools": {
                    tool: {
                        "success": r.get("tools", {}).get(tool, {}).get("success"),
                        "num_findings": r.get("tools", {}).get(tool, {}).get("num_findings", 0),
                    }
                    for tool in tools
                    if tool in r.get("tools", {})
                },
            }
            for r in all_results
        ],
    }

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch code review baseline generation across SWE-CARE instances"
    )
    parser.add_argument(
        "--split", nargs="+", default=["dev"],
        choices=["dev", "test"],
        help="Dataset split(s) to process (default: dev)",
    )
    parser.add_argument(
        "--repo", type=str, default=None,
        help="Only process instances for this repo (e.g. 'fastapi/fastapi')",
    )
    parser.add_argument(
        "--instances-file", type=str, default=None,
        help="File with instance IDs to process (one per line, # comments ignored)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of instances to process",
    )
    parser.add_argument(
        "--output-dir", type=str, default="baselines_output",
        help="Where to save results (default: baselines_output/)",
    )
    parser.add_argument(
        "--repos-dir", type=str, default="repos",
        help="Where to cache repo clones (default: repos/)",
    )
    parser.add_argument(
        "--tools", nargs="+", default=TOOL_CHOICES,
        choices=TOOL_CHOICES,
        help=f"Which tools to run (default: {' '.join(TOOL_CHOICES)})",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Per-tool timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-process all instances even if results already exist",
    )
    parser.add_argument(
        "--keep-repos", action="store_true",
        help="Keep cloned repos instead of cleaning up after processing",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    repos_dir = Path(args.repos_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    tools = args.tools
    workers = args.workers

    # Load .env file if present
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    # Load instances from all requested splits
    all_instances = []
    for split in args.split:
        instances = dataset_utils.load_instances(split=split, repo=args.repo)
        logger.info("Loaded %d instances from split '%s'", len(instances), split)
        all_instances.extend(instances)

    # Filter to specific instance IDs if a file is provided
    if args.instances_file:
        id_file = Path(args.instances_file)
        if not id_file.exists():
            logger.error("Instances file not found: %s", id_file)
            return
        wanted_ids = set()
        for line in id_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                wanted_ids.add(line)
        all_instances = [i for i in all_instances if i["instance_id"] in wanted_ids]
        logger.info(
            "Filtered to %d instances from %s",
            len(all_instances), id_file,
        )

    if args.limit and len(all_instances) > args.limit:
        all_instances = all_instances[: args.limit]
        logger.info("Limited to %d instances", len(all_instances))

    if not all_instances:
        logger.error("No instances found.")
        return

    # Group by repo
    by_repo: dict[str, list[dict]] = {}
    for inst in all_instances:
        by_repo.setdefault(inst["repo"], []).append(inst)

    logger.info(
        "Total: %d instances across %d repos, tools: %s",
        len(all_instances), len(by_repo), ", ".join(tools),
    )

    # --- Phase 1: Sequential setup ---
    source_repos: dict[str, Path] = {}
    all_results: list[dict] = []
    to_process: list[dict] = []

    for repo, repo_instances in by_repo.items():
        logger.info("=== Setup: %s (%d instances) ===", repo, len(repo_instances))

        repo_to_process = []
        for inst in repo_instances:
            if resume:
                existing = load_existing_result(output_dir, inst["instance_id"])
                if existing is not None and not instance_needs_processing(existing, tools):
                    logger.info("Skipping (already done): %s", inst["instance_id"])
                    all_results.append(existing)
                    continue
            repo_to_process.append(inst)

        if not repo_to_process:
            logger.info("All instances for %s already processed, skipping.", repo)
            continue

        logger.info("%d instance(s) to process for %s", len(repo_to_process), repo)

        # Clone source repo + fetch PR commits
        repo_path = repo_manager.clone_repo(repo, cache_dir=repos_dir)

        fetched_prs: set[int] = set()
        for inst in repo_to_process:
            pn = inst.get("pull_number")
            if pn and pn not in fetched_prs:
                repo_manager.fetch_pr_commits(repo_path, pn)
                fetched_prs.add(pn)

        source_repos[repo] = repo_path
        to_process.extend(repo_to_process)

    if not to_process:
        logger.info("All instances already processed.")
        write_summary(output_dir, all_results, 0.0, tools)
        return

    # --- Phase 2: Parallel processing ---
    workdir_root = repos_dir / "workdirs"
    start_time = time.time()
    processed = 0
    lock = threading.Lock()

    logger.info(
        "Processing %d instance(s) with %d worker(s)",
        len(to_process), workers,
    )

    def _process_one(inst: dict) -> dict:
        """Process a single instance: run requested tools and save results."""
        instance_id = inst["instance_id"]
        repo = inst["repo"]
        source_path = source_repos[repo]
        safe_id = instance_id.replace("/", "__")
        inst_output_dir = output_dir / safe_id
        inst_output_dir.mkdir(parents=True, exist_ok=True)
        workdir = None

        tool_results: dict[str, dict] = {}

        try:
            # Both PR-Agent and CodeRabbit need a local workdir
            needs_workdir = ("pr-agent" in tools or "coderabbit" in tools or "claude-code" in tools or "codex" in tools)
            if needs_workdir:
                workdir = repo_manager.create_instance_workdir(
                    source_path, instance_id, workdir_root,
                )

            if "pr-agent" in tools:
                logger.info("[%s] Running PR-Agent...", instance_id)
                tool_results["pr-agent"] = run_pr_agent(
                    inst, workdir, inst_output_dir, args.timeout,
                )
                status = "OK" if tool_results["pr-agent"]["success"] else "FAIL"
                logger.info(
                    "[%s] PR-Agent: %s (%d findings, %.1fs)",
                    instance_id, status,
                    tool_results["pr-agent"]["num_findings"],
                    tool_results["pr-agent"]["elapsed_seconds"],
                )

            if "coderabbit" in tools:
                logger.info("[%s] Running CodeRabbit...", instance_id)
                tool_results["coderabbit"] = run_coderabbit(
                    inst, workdir, inst_output_dir, args.timeout,
                )
                status = "OK" if tool_results["coderabbit"]["success"] else "FAIL"
                logger.info(
                    "[%s] CodeRabbit: %s (%d findings, %.1fs)",
                    instance_id, status,
                    tool_results["coderabbit"]["num_findings"],
                    tool_results["coderabbit"]["elapsed_seconds"],
                )

            if "claude-code" in tools:
                logger.info("[%s] Running Claude Code...", instance_id)
                tool_results["claude-code"] = run_claude_code(
                    inst, workdir, inst_output_dir, args.timeout,
                )
                status = "OK" if tool_results["claude-code"]["success"] else "FAIL"
                logger.info(
                    "[%s] Claude Code: %s (%d findings, %.1fs)",
                    instance_id, status,
                    tool_results["claude-code"]["num_findings"],
                    tool_results["claude-code"]["elapsed_seconds"],
                )

            if "codex" in tools:
                logger.info("[%s] Running Codex...", instance_id)
                tool_results["codex"] = run_codex(
                    inst, workdir, inst_output_dir, args.timeout,
                )
                status = "OK" if tool_results["codex"]["success"] else "FAIL"
                logger.info(
                    "[%s] Codex: %s (%d findings, %.1fs)",
                    instance_id, status,
                    tool_results["codex"]["num_findings"],
                    tool_results["codex"]["elapsed_seconds"],
                )

            # Devin doesn't need a local checkout (uses GitHub API)
            if "devin" in tools:
                logger.info("[%s] Running Devin...", instance_id)
                tool_results["devin"] = run_devin(
                    inst, inst_output_dir, args.timeout,
                )
                status = "OK" if tool_results["devin"]["success"] else "FAIL"
                logger.info(
                    "[%s] Devin: %s (%d findings, %.1fs)",
                    instance_id, status,
                    tool_results["devin"]["num_findings"],
                    tool_results["devin"]["elapsed_seconds"],
                )

        except Exception:
            logger.exception("[%s] Error processing instance", instance_id)

        result = {
            "instance_id": instance_id,
            "repo": repo,
            "pull_number": inst.get("pull_number"),
            "base_commit": inst.get("base_commit"),
            "head_commit": inst.get("commit_to_review", {}).get("head_commit"),
            "tools": tool_results,
        }

        # Merge with existing result if doing partial re-run
        if resume:
            existing = load_existing_result(output_dir, instance_id)
            if existing is not None:
                merged_tools = existing.get("tools", {})
                merged_tools.update(tool_results)
                result["tools"] = merged_tools

        # Save per-instance result
        result_file = inst_output_dir / "result.json"
        result_file.write_text(json.dumps(result, indent=2, default=str))

        # Clean up workdir
        if workdir and workdir.exists():
            try:
                shutil.rmtree(workdir)
            except OSError as e:
                logger.warning("[%s] Failed to clean up workdir: %s", instance_id, e)

        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_inst = {
            executor.submit(_process_one, inst): inst for inst in to_process
        }

        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            try:
                result = future.result()
            except Exception:
                logger.exception("Future failed for %s", inst["instance_id"])
                result = {
                    "instance_id": inst["instance_id"],
                    "repo": inst["repo"],
                    "pull_number": inst.get("pull_number"),
                    "tools": {},
                    "error": "Processing failed",
                }

            with lock:
                all_results.append(result)
                processed += 1
                count = processed

            logger.info(
                "[%d/%d] Completed %s",
                len(all_results), len(all_instances), inst["instance_id"],
            )

            # Progressive summary every 10 completions
            if count % 10 == 0:
                with lock:
                    elapsed_so_far = time.time() - start_time
                    summary = write_summary(output_dir, list(all_results), elapsed_so_far, tools)
                for tool in tools:
                    ts = summary.get("tool_totals", {}).get(tool, {})
                    logger.info(
                        "Progress [%s]: %d/%d OK, %d findings",
                        tool, ts.get("successes", 0),
                        ts.get("instances_run", 0),
                        ts.get("total_findings", 0),
                    )

    # Cleanup source repos
    if not args.keep_repos:
        for repo in source_repos:
            repo_dir = repos_dir / repo.replace("/", "__")
            if repo_dir.exists():
                logger.info("Cleaning up repo: %s", repo_dir)
                shutil.rmtree(repo_dir)
        if workdir_root.exists():
            try:
                workdir_root.rmdir()
            except OSError:
                pass

    # Final summary
    total_elapsed = time.time() - start_time
    summary = write_summary(output_dir, all_results, total_elapsed, tools)
    logger.info("=== DONE === %d instances in %.0fs", len(all_results), total_elapsed)
    for tool in tools:
        ts = summary.get("tool_totals", {}).get(tool, {})
        logger.info(
            "  %s: %d/%d succeeded, %d total findings, avg %.1fs/instance",
            tool, ts.get("successes", 0), ts.get("instances_run", 0),
            ts.get("total_findings", 0), ts.get("avg_elapsed", 0),
        )
    logger.info("Summary: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
