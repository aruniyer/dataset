"""Agent resolution of code review comments using Claude Code inside Docker.

Builds prompts from review comments, invokes Claude Code CLI inside a Docker
container, and verifies the agent's changes against Stage 3 tests.

All matched comments for an instance are batched into a single Claude Code
invocation so the agent sees the full review context and makes one coherent
set of changes.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from execution.container_runtime import DockerContainerSession

logger = logging.getLogger(__name__)

# Claude Code invocation timeout (seconds).  Must be generous — the agent
# may read many files and make multiple edits in a large repo.
CLAUDE_TIMEOUT = 600

# Copilot CLI invocation timeout (seconds).
COPILOT_TIMEOUT = 600


@dataclass
class AgentResolution:
    """Result of attempting to resolve a single review comment with Claude Code."""

    comment_index: int
    comment_text: str
    file_path: str
    resolved: bool  # Stage 3 test passed on agent's change
    test_passed: bool
    test_output: str
    agent_diff: str  # git diff of agent's changes (shared across batch)
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def build_tool_prompt(
    findings: list[dict],
    patch_to_review: str,
    repo: str,
) -> str:
    """Build prompt from tool-generated findings (file, issue_header, issue_content, start_line, end_line).

    Each finding becomes a section with file path, line range, issue header, and content.
    Includes the full PR diff as context.
    """
    sections = []
    for i, finding in enumerate(findings, 1):
        # Support both pr-agent (start_line/end_line) and devin (line) formats
        start_line = finding.get("start_line", "")
        end_line = finding.get("end_line", "")
        single_line = finding.get("line")
        if start_line and end_line:
            line_range = f" (lines {start_line}–{end_line})"
        elif single_line:
            line_range = f" (line {single_line})"
        else:
            line_range = ""

        # Support both pr-agent (issue_header/issue_content) and devin (type/description)
        header = finding.get("issue_header") or finding.get("type") or "Code Issue"
        content = (
            finding.get("issue_content")
            or finding.get("description")
            or finding.get("comment")
            or ""
        )

        sections.append(
            f"### Finding {i}\n"
            f"**File:** `{finding['file']}`{line_range}\n\n"
            f"**Issue:** {header}\n\n"
            f"{content}"
        )

    findings_block = "\n\n---\n\n".join(sections)

    return f"""You are resolving automated code review findings on the repository `{repo}`.

## Review Findings

{findings_block}

## Full PR Diff
```
{patch_to_review}
```

## Instructions
1. Read each finding and understand what code problem it identifies.
2. Modify the code to address ALL of the findings.
3. Make the minimal changes necessary — do not make unrelated modifications.
"""


def build_batch_prompt(
    comments: list[dict],
    patch_to_review: str,
    repo: str,
) -> str:
    """Build a single prompt covering all review comments for an instance.

    Each comment includes its text, file path, and diff hunk.
    Excludes: test code, merged_patch, after-file content.
    """
    sections = []
    for i, comment in enumerate(comments, 1):
        sections.append(
            f"### Comment {i}\n"
            f"**File:** `{comment['path']}`\n\n"
            f"**Diff hunk being reviewed:**\n"
            f"```\n{comment.get('diff_hunk', '')}\n```\n\n"
            f"**Reviewer says:**\n{comment['text']}"
        )

    comments_block = "\n\n---\n\n".join(sections)

    return f"""You are resolving code review comments on the repository `{repo}`.

## Review Comments

{comments_block}

## Full PR Diff
```
{patch_to_review}
```

## Instructions
1. Read each review comment and understand what change the reviewer is requesting.
2. Modify the code to address ALL of the reviewer's feedback.
3. Make the minimal changes necessary — do not make unrelated modifications.
"""


def setup_copilot_in_container(session: DockerContainerSession) -> None:
    """Install GitHub Copilot CLI and trust the workspace directory."""
    # Install Copilot CLI via npm (Node.js should be available in most images)
    result = session.run_command(
        "npm install -g @github/copilot 2>&1 || "
        "apt-get update -qq && apt-get install -y -qq nodejs npm && npm install -g @github/copilot 2>&1",
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install Copilot CLI: {result.stderr[-2000:]}")
    logger.info("Copilot CLI installed successfully")

    # Trust the workspace so Copilot doesn't prompt
    session.run_command(
        "mkdir -p ~/.copilot && "
        'echo \'{"trust":["/workspace","/testbed"]}\' > ~/.copilot/trust.json',
        timeout=10,
    )

    # Mark workspace as safe for git
    session.run_command(
        "git config --global --add safe.directory /workspace",
        timeout=10,
    )


def invoke_copilot_in_container(
    session: DockerContainerSession,
    prompt: str,
    model: str | None = None,
) -> tuple[str, int]:
    """Invoke Copilot CLI in non-interactive --yolo mode.

    Returns (stdout, returncode).
    """
    # Write prompt to a temp file locally, then copy into container.
    # This avoids shell escaping issues with complex prompts.
    prompt_path = "/tmp/copilot_prompt.txt"
    with tempfile.NamedTemporaryFile(mode="w", suffix="_prompt.txt", delete=False) as f:
        f.write(prompt)
        local_prompt = Path(f.name)
    try:
        session.copy_to(local_prompt, prompt_path)
    finally:
        local_prompt.unlink(missing_ok=True)

    # Build the copilot command
    cmd = 'cd /workspace && copilot -p "$(cat /tmp/copilot_prompt.txt)" --yolo -s'

    logger.info("  Copilot command: %s", cmd[:200])
    result = session.run_command(cmd, timeout=COPILOT_TIMEOUT)
    if result.returncode != 0:
        logger.warning("  Copilot stderr: %s", (result.stderr or "")[:500])
    return result.stdout, result.returncode


def setup_claude_in_container(session: DockerContainerSession) -> None:
    """One-time container setup:
    1. Create non-root 'agent' user
    2. Install Claude Code via curl | bash as agent user
    3. Copy credentials from neutral mount path into agent's .claude dir
    4. chown /workspace to agent
    """
    # Create agent user (home dir must not pre-exist for skel copy to work)
    result = session.run_command(
        "useradd -m agent",
        timeout=30,
    )
    if result.returncode != 0:
        # User may already exist
        logger.warning(
            "useradd returned %d: %s", result.returncode, result.stderr.strip()
        )

    # Install Claude Code as agent user.
    # Use a generous timeout — parallel workers downloading simultaneously
    # can cause network/disk contention.
    result = session.run_command(
        "su - agent -c 'curl -fsSL https://claude.ai/install.sh | bash'",
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install Claude Code: {result.stderr[-2000:]}")
    logger.info("Claude Code installed successfully")

    # Copy credentials from neutral mount path into agent's .claude dir.
    # Credentials are mounted at /etc/claude-credentials.json to avoid
    # Docker pre-creating /home/agent/ before the user is created.
    session.run_command(
        "mkdir -p /home/agent/.claude && "
        "cp /etc/claude-credentials.json /home/agent/.claude/.credentials.json 2>/dev/null; "
        "chown -R agent:agent /home/agent/.claude",
        timeout=10,
    )

    # Transfer workspace ownership and mark as safe for git
    result = session.run_command(
        "chown -R agent:agent /workspace && "
        "git config --global --add safe.directory /workspace",
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to chown /workspace: {result.stderr[-1000:]}")


def invoke_claude_in_container(
    session: DockerContainerSession,
    prompt: str,
    model: str,
) -> tuple[str, int]:
    """Run claude -p --dangerously-skip-permissions as the agent user.

    Writes the prompt to a temp file inside the container to avoid
    shell quoting issues, then pipes it to claude via stdin.

    Returns (stdout, returncode).
    """
    # Write prompt to a file inside the container to avoid quoting issues.
    # Place it in agent's home dir so it's readable by the agent user.
    prompt_path = "/home/agent/prompt.txt"
    with tempfile.NamedTemporaryFile(mode="w", suffix="_prompt.txt", delete=False) as f:
        f.write(prompt)
        local_prompt = Path(f.name)
    try:
        session.copy_to(local_prompt, prompt_path)
    finally:
        local_prompt.unlink(missing_ok=True)

    # Ensure prompt file is readable by agent
    session.run_command(f"chown agent:agent {prompt_path}", timeout=10)

    # Use the full path to the claude binary to avoid PATH/shell issues.
    claude_bin = "/home/agent/.local/bin/claude"
    cmd = [
        "su",
        "-",
        "agent",
        "-s",
        "/bin/bash",
        "-c",
        f"cd /workspace && "
        f"cat {prompt_path} | {claude_bin} -p --dangerously-skip-permissions "
        f"--model {model}",
    ]

    logger.info("  Claude command: %s", cmd[:200])
    result = session.run_command(cmd, timeout=CLAUDE_TIMEOUT)
    if result.returncode != 0:
        logger.warning("  Claude stderr: %s", (result.stderr or "")[:500])
    return result.stdout, result.returncode


def verify_with_test(
    session: DockerContainerSession,
    test_code: str,
    test_filename: str,
    language: str,
) -> tuple[bool, str]:
    """Write test into container and run it. Returns (passed, output).

    Uses -c /dev/null -p no:cacheprovider for isolation (Python).
    """
    # Write test code to a temp file and copy into container
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{test_filename}", delete=False
    ) as f:
        f.write(test_code)
        local_test_path = Path(f.name)

    try:
        container_test_path = f"/workspace/{test_filename}"
        session.copy_to(local_test_path, container_test_path)
    finally:
        local_test_path.unlink(missing_ok=True)

    # Run the test
    if language == "python":
        run_cmd = (
            f"python -m pytest {container_test_path} -x -v --tb=short --no-header "
            f"-c /dev/null -p no:cacheprovider"
        )
    elif language in ("javascript", "typescript"):
        run_cmd = f"npx jest --no-coverage {container_test_path}"
    elif language == "go":
        test_dir = str(Path(container_test_path).parent)
        run_cmd = f"go test -v -run . {test_dir}"
    else:
        return False, f"Execution not supported for language: {language}"

    result = session.run_command(run_cmd, timeout=120)
    combined = (result.stdout + "\n" + result.stderr).strip()
    passed = result.returncode == 0
    return passed, combined


def resolve_instance(
    instance: dict,
    matched_comments: dict[int, tuple[dict, str, str]],
    session: DockerContainerSession,
    model: str,
    language: str,
) -> list[AgentResolution]:
    """Resolve all matched comments for an instance in a single Claude invocation.

    Args:
        instance: The dataset instance dict.
        matched_comments: Mapping of comment_index -> (comment_dict, test_code, test_filename).
        session: Active Docker container session.
        model: Claude model to use.
        language: Programming language for test execution.

    Flow:
        1. Reset to head_commit, reinstall
        2. Build a single prompt with all comments
        3. Invoke Claude Code once
        4. Capture git diff
        5. Reinstall (agent may have edited source)
        6. Verify each Stage 3 test individually
        7. Return list of AgentResolution
    """
    head_commit = instance["commit_to_review"]["head_commit"]
    repo = instance["repo"]
    patch_to_review = instance["commit_to_review"]["patch_to_review"]

    # Collect comments in index order for prompt building
    ordered_indices = sorted(matched_comments.keys())
    comments_for_prompt = [matched_comments[i][0] for i in ordered_indices]

    try:
        # 1. Reset to head commit
        reset_result = session.run_command(
            f"git checkout --force {head_commit} && git clean -fd --quiet",
            timeout=120,
        )
        if reset_result.returncode != 0:
            error = f"git checkout failed: {reset_result.stderr[:500]}"
            return [
                AgentResolution(
                    comment_index=i,
                    comment_text=matched_comments[i][0]["text"],
                    file_path=matched_comments[i][0]["path"],
                    resolved=False,
                    test_passed=False,
                    test_output="",
                    agent_diff="",
                    error=error,
                )
                for i in ordered_indices
            ]

        # 2. Reinstall
        if language == "python":
            session.run_command("pip install -e . --no-deps --quiet", timeout=120)

        # 3. Build batch prompt and invoke Claude Code once
        prompt = build_batch_prompt(
            comments=comments_for_prompt,
            patch_to_review=patch_to_review,
            repo=repo,
        )
        logger.info("  Invoking Claude Code for %d comment(s)...", len(ordered_indices))
        agent_stdout, agent_rc = invoke_claude_in_container(session, prompt, model)
        logger.info(
            "  Claude Code returned (rc=%d, output=%d chars)",
            agent_rc,
            len(agent_stdout),
        )

        # 4. Capture git diff
        diff_result = session.run_command("git diff", timeout=30)
        agent_diff = diff_result.stdout

        no_changes = False
        if not agent_diff.strip():
            status_result = session.run_command("git status --porcelain", timeout=15)
            if not status_result.stdout.strip():
                no_changes = True

        if no_changes:
            return [
                AgentResolution(
                    comment_index=i,
                    comment_text=matched_comments[i][0]["text"],
                    file_path=matched_comments[i][0]["path"],
                    resolved=False,
                    test_passed=False,
                    test_output="",
                    agent_diff="",
                    error="Agent made no changes",
                )
                for i in ordered_indices
            ]

        # 5. Reinstall (agent may have edited source)
        if language == "python":
            session.run_command("pip install -e . --no-deps --quiet", timeout=120)

        # 6. Verify each Stage 3 test individually
        resolutions = []
        for i in ordered_indices:
            comment, test_code, test_filename = matched_comments[i]
            test_passed, test_output = verify_with_test(
                session, test_code, test_filename, language
            )
            logger.info(
                "  Comment %d test result: %s", i, "PASS" if test_passed else "FAIL"
            )
            resolutions.append(
                AgentResolution(
                    comment_index=i,
                    comment_text=comment["text"],
                    file_path=comment["path"],
                    resolved=test_passed,
                    test_passed=test_passed,
                    test_output=test_output,
                    agent_diff=agent_diff,
                    error=None,
                )
            )
        return resolutions

    except Exception as e:
        logger.exception("  Error resolving instance")
        return [
            AgentResolution(
                comment_index=i,
                comment_text=matched_comments[i][0]["text"],
                file_path=matched_comments[i][0]["path"],
                resolved=False,
                test_passed=False,
                test_output="",
                agent_diff="",
                error=str(e),
            )
            for i in ordered_indices
        ]
