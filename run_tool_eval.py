#!/usr/bin/env python3
"""Per-instance evaluation of a review tool using Claude Code in Docker.

Loads tool-generated findings (e.g. from pr-agent), invokes Claude Code inside
a Docker container to apply the suggested fixes, then verifies the agent's
changes against all successful Stage 3 tests for the instance.

Key difference from run_agent_resolution.py: tool findings don't correspond to
specific reference comments, so ALL successful stage 3 tests are run (not just
matched ones).

Usage:
  python run_tool_eval.py --instance-id ansible__ansible-20646@f695114
  python run_tool_eval.py --instance-id <id> --tool pr-agent --model claude-sonnet-4-6
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from execution.container_runtime import DockerContainerSession, get_docker_image_name
from pipeline.agent_resolver import (
    build_tool_prompt,
    invoke_claude_in_container,
    setup_claude_in_container,
    verify_with_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TOOL = "pr-agent"
DEFAULT_TOOL_RESULTS_DIR = "pr-agent-result/pr-agent"
DEFAULT_TESTGEN_DIR = "results_testgen_docker_full"
DEFAULT_STAGE3_FILE = "results_pipeline_funnel/stage3_testgen_verified.jsonl"
DEFAULT_OUTPUT_DIR = "results_tool_eval"
DEFAULT_CREDENTIALS = Path.home() / ".claude" / ".credentials.json"


def load_stage3_instance(stage3_file: Path, instance_id: str) -> dict | None:
    """Load a single instance from the stage3 JSONL file."""
    with stage3_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            if inst["instance_id"] == instance_id:
                return inst
    return None


def load_testgen_results(testgen_dir: Path, instance_id: str) -> dict | None:
    """Load testgen result.json for an instance."""
    slug = instance_id.replace("/", "__")
    result_file = testgen_dir / slug / "result.json"
    if not result_file.exists():
        return None
    return json.loads(result_file.read_text())


def load_tool_findings(tool_results_dir: Path, instance_id: str, tool: str) -> dict | None:
    """Load tool result.json for an instance.

    Returns the tool's data dict (with 'findings', 'success', etc.), or None if not found.
    """
    result_file = tool_results_dir / instance_id / "result.json"
    if not result_file.exists():
        return None
    data = json.loads(result_file.read_text())
    tools = data.get("tools", {})
    return tools.get(tool)


def get_stage3_tests(testgen_results: dict) -> list[tuple[int, dict]]:
    """Extract all successful stage 3 tests from testgen results.

    Returns list of (comment_index, entry) for entries where success=True.
    """
    tests = []
    for i, entry in enumerate(testgen_results.get("results", [])):
        if entry.get("success"):
            tests.append((i, entry))
    return tests


def process_tool_instance(
    instance: dict,
    tool_data: dict,
    testgen_results: dict,
    output_dir: Path,
    model: str,
    tool: str,
    docker_image: str,
    credentials_path: Path,
) -> dict:
    """Process a single instance: apply tool findings via Claude Code and verify.

    Flow:
    1. Start Docker container with credentials mounted read-only
    2. Setup: create agent user, install Claude Code, chown workspace
    3. Reset to head_commit
    4. Build prompt from tool findings, invoke Claude Code once
    5. Capture git diff
    6. If no changes: mark all tests failed
    7. Reinstall, verify each Stage 3 test individually
    8. Save result.json
    9. Remove container
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    head_commit = instance["commit_to_review"]["head_commit"]
    patch_to_review = instance["commit_to_review"]["patch_to_review"]

    findings = tool_data.get("findings", [])
    logger.info(
        "Processing instance: %s (%d findings)", instance_id, len(findings)
    )

    instance_dir = output_dir / instance_id.replace("/", "__")
    instance_dir.mkdir(parents=True, exist_ok=True)

    # Collect all successful stage 3 tests
    stage3_tests = get_stage3_tests(testgen_results)
    if not stage3_tests:
        logger.warning("No successful stage 3 tests for %s", instance_id)
        result = {
            "instance_id": instance_id,
            "repo": repo,
            "tool": tool,
            "model": model,
            "num_findings": len(findings),
            "agent_diff": "",
            "num_tests": 0,
            "num_tests_passed": 0,
            "test_pass_rate": 0.0,
            "results": [],
            "error": "No successful Stage 3 tests found",
        }
        (instance_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
        return result

    # Determine language from first successful test
    language = "python"
    for _, entry in stage3_tests:
        language = entry.get("language", "python")
        break

    # Build test filename mapping
    ext_map = {
        "python": ".py",
        "javascript": ".test.js",
        "typescript": ".test.ts",
        "go": "_test.go",
    }
    ext = ext_map.get(language, ".py")

    # Start container
    safe_name = instance_id.replace("/", "--").replace("@", "-")
    container_name = f"rb-tooleval-{safe_name}"

    # Remove any stale container
    rm_result = subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True, text=True,
    )
    if rm_result.returncode == 0 and rm_result.stdout.strip():
        time.sleep(2)

    volumes = []
    if credentials_path.exists():
        volumes.append(f"{credentials_path}:/etc/claude-credentials.json:ro")

    session = DockerContainerSession(
        docker_image,
        name=container_name,
        volumes=volumes,
    )

    test_results: list[dict] = []
    agent_diff = ""

    try:
        session.start()
        logger.info("Started container %s (image: %s)", container_name, docker_image)

        # Setup Claude Code in container
        setup_claude_in_container(session)

        # Reset to head commit
        reset_result = session.run_command(
            f"git checkout --force {head_commit} && git clean -fd --quiet",
            timeout=120,
        )
        if reset_result.returncode != 0:
            error = f"git checkout failed: {reset_result.stderr[:500]}"
            logger.error("  %s", error)
            result = {
                "instance_id": instance_id,
                "repo": repo,
                "tool": tool,
                "model": model,
                "num_findings": len(findings),
                "agent_diff": "",
                "num_tests": len(stage3_tests),
                "num_tests_passed": 0,
                "test_pass_rate": 0.0,
                "results": [],
                "error": error,
            }
            (instance_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
            return result

        # Reinstall before agent runs
        if language == "python":
            session.run_command("pip install -e . --no-deps --quiet", timeout=120)

        # Build prompt and invoke Claude Code
        prompt = build_tool_prompt(
            findings=findings,
            patch_to_review=patch_to_review,
            repo=repo,
        )
        logger.info("  Invoking Claude Code for %d finding(s)...", len(findings))
        agent_stdout, agent_rc = invoke_claude_in_container(session, prompt, model)
        logger.info(
            "  Claude Code returned (rc=%d, output=%d chars)", agent_rc, len(agent_stdout)
        )

        # Capture git diff
        diff_result = session.run_command("git diff", timeout=30)
        agent_diff = diff_result.stdout

        no_changes = False
        if not agent_diff.strip():
            status_result = session.run_command("git status --porcelain", timeout=15)
            if not status_result.stdout.strip():
                no_changes = True

        if no_changes:
            logger.warning("  Agent made no changes")
            test_results = [
                {
                    "comment_index": i,
                    "comment_text": entry.get("comment_text", ""),
                    "test_passed": False,
                    "test_output": "",
                    "error": "Agent made no changes",
                }
                for i, entry in stage3_tests
            ]
        else:
            # Reinstall (agent may have edited source)
            if language == "python":
                session.run_command("pip install -e . --no-deps --quiet", timeout=120)

            # Verify each stage 3 test
            for i, entry in stage3_tests:
                test_code = entry["test_code"]
                test_filename = f"test_review_comment_{i}{ext}"
                logger.info("  Running test %d/%d...", i + 1, len(stage3_tests))
                test_passed, test_output = verify_with_test(
                    session, test_code, test_filename, language
                )
                logger.info(
                    "  Test %d: %s", i, "PASS" if test_passed else "FAIL"
                )
                test_results.append({
                    "comment_index": i,
                    "comment_text": entry.get("comment_text", ""),
                    "test_passed": test_passed,
                    "test_output": test_output,
                    "error": None,
                })

    except Exception as e:
        logger.exception("  Error processing instance")
        test_results = [
            {
                "comment_index": i,
                "comment_text": entry.get("comment_text", ""),
                "test_passed": False,
                "test_output": "",
                "error": str(e),
            }
            for i, entry in stage3_tests
        ]
    finally:
        session.remove(force=True)
        logger.info("Removed container %s", container_name)

    # Compute stats
    num_tests = len(test_results)
    num_tests_passed = sum(1 for r in test_results if r["test_passed"])
    test_pass_rate = num_tests_passed / num_tests if num_tests > 0 else 0.0

    result = {
        "instance_id": instance_id,
        "repo": repo,
        "tool": tool,
        "model": model,
        "num_findings": len(findings),
        "agent_diff": agent_diff,
        "num_tests": num_tests,
        "num_tests_passed": num_tests_passed,
        "test_pass_rate": test_pass_rate,
        "results": test_results,
    }

    result_file = instance_dir / "result.json"
    result_file.write_text(json.dumps(result, indent=2, default=str))
    logger.info(
        "Result saved to %s (tests: %d/%d passed = %.1f%%)",
        result_file, num_tests_passed, num_tests, test_pass_rate * 100,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a review tool using Claude Code in Docker"
    )
    parser.add_argument(
        "--instance-id", type=str, required=True,
        help="Instance ID to process",
    )
    parser.add_argument(
        "--tool", type=str, default=DEFAULT_TOOL,
        help=f"Review tool name (default: {DEFAULT_TOOL})",
    )
    parser.add_argument(
        "--tool-results-dir", type=str, default=DEFAULT_TOOL_RESULTS_DIR,
        help=f"Directory with tool result.json files (default: {DEFAULT_TOOL_RESULTS_DIR})",
    )
    parser.add_argument(
        "--testgen-dir", type=str, default=DEFAULT_TESTGEN_DIR,
        help=f"Testgen results directory (default: {DEFAULT_TESTGEN_DIR})",
    )
    parser.add_argument(
        "--stage3-file", type=str, default=DEFAULT_STAGE3_FILE,
        help=f"Stage 3 JSONL file (default: {DEFAULT_STAGE3_FILE})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--credentials", type=str, default=str(DEFAULT_CREDENTIALS),
        help=f"Path to Claude credentials file (default: {DEFAULT_CREDENTIALS})",
    )

    args = parser.parse_args()

    stage3_file = Path(args.stage3_file)
    testgen_dir = Path(args.testgen_dir)
    tool_results_dir = Path(args.tool_results_dir)
    output_dir = Path(args.output_dir)
    credentials_path = Path(args.credentials)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load instance from stage3
    instance = load_stage3_instance(stage3_file, args.instance_id)
    if instance is None:
        logger.error("Instance %s not found in %s", args.instance_id, stage3_file)
        sys.exit(1)

    # Load testgen results
    testgen_results = load_testgen_results(testgen_dir, args.instance_id)
    if testgen_results is None:
        logger.error(
            "Testgen results not found for %s in %s",
            args.instance_id, testgen_dir,
        )
        sys.exit(1)

    # Load tool findings
    tool_data = load_tool_findings(tool_results_dir, args.instance_id, args.tool)
    if tool_data is None:
        logger.error(
            "Tool findings not found for %s in %s",
            args.instance_id, tool_results_dir,
        )
        sys.exit(1)

    if not tool_data.get("success"):
        logger.error("Tool reported failure for %s, skipping", args.instance_id)
        sys.exit(1)

    if not tool_data.get("findings"):
        logger.error("Tool has no findings for %s, skipping", args.instance_id)
        sys.exit(1)

    # Get Docker image
    docker_image = get_docker_image_name(args.instance_id)

    # Check credentials
    if not credentials_path.exists():
        logger.warning(
            "Credentials file not found: %s (container will have no API access)",
            credentials_path,
        )

    t0 = time.time()
    result = process_tool_instance(
        instance=instance,
        tool_data=tool_data,
        testgen_results=testgen_results,
        output_dir=output_dir,
        model=args.model,
        tool=args.tool,
        docker_image=docker_image,
        credentials_path=credentials_path,
    )
    elapsed = time.time() - t0

    logger.info(
        "=== DONE === %s: %d/%d tests passed (%.1f%%) in %.1fs",
        args.instance_id,
        result["num_tests_passed"],
        result["num_tests"],
        result["test_pass_rate"] * 100,
        elapsed,
    )


if __name__ == "__main__":
    main()
