#!/usr/bin/env python3
"""Per-instance agent resolution of code review comments.

Uses Claude Code inside a Docker container to resolve review comments,
then verifies the agent's changes against Stage 3 tests.

All matched comments for an instance are batched into a single Claude Code
invocation so the agent sees the full review context and makes one coherent
set of changes.

Usage:
  python run_agent_resolution.py --instance-id <id>
  python run_agent_resolution.py --instance-id <id> --model claude-sonnet-4-6
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
    resolve_instance,
    setup_claude_in_container,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TESTGEN_DIR = "results_testgen_docker_full"
DEFAULT_STAGE3_FILE = "results_pipeline_funnel/stage3_testgen_verified.jsonl"
DEFAULT_OUTPUT_DIR = "results_agent_resolution"
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


def match_tests_to_comments(
    comments: list[dict], testgen_result: dict
) -> dict[int, tuple[dict, str, str]]:
    """Match Stage 3 comments to their verified test code.

    Returns dict mapping comment index -> (comment_dict, test_code, test_filename).
    Only includes comments that had successful tests in Stage 3.
    """
    matched = {}
    for i, comment in enumerate(comments):
        for entry in testgen_result.get("results", []):
            if entry.get("comment_text") == comment["text"] and entry.get("success"):
                test_code = entry["test_code"]
                language = entry.get("language", "python")
                ext_map = {
                    "python": ".py",
                    "javascript": ".test.js",
                    "typescript": ".test.ts",
                    "go": "_test.go",
                }
                ext = ext_map.get(language, ".py")
                test_filename = f"test_review_comment_{i}{ext}"
                matched[i] = (comment, test_code, test_filename)
                break
    return matched


def process_instance(
    instance: dict,
    testgen_results: dict,
    output_dir: Path,
    model: str,
    docker_image: str,
    credentials_path: Path,
) -> dict:
    """Process a single instance: resolve all comments together and verify.

    Flow:
    1. Start Docker container with credentials mounted read-only
    2. Setup: create agent user, install Claude Code, chown workspace
    3. Match comments to Stage 3 tests
    4. Build a single prompt with all comments, invoke Claude Code once
    5. Verify each Stage 3 test individually
    6. Save result.json
    7. Remove container
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    comments = instance["reference_review_comments"]

    logger.info("Processing instance: %s (%d comments)", instance_id, len(comments))

    instance_dir = output_dir / instance_id.replace("/", "__")
    instance_dir.mkdir(parents=True, exist_ok=True)

    # Match comments to Stage 3 tests
    matched = match_tests_to_comments(comments, testgen_results)
    if not matched:
        logger.warning("No verified tests found for %s", instance_id)
        result = {
            "instance_id": instance_id,
            "repo": repo,
            "agent": "claude-code",
            "model": model,
            "num_comments": 0,
            "num_resolved": 0,
            "resolution_rate": 0.0,
            "results": [],
            "error": "No verified Stage 3 tests to match",
        }
        result_file = instance_dir / "result.json"
        result_file.write_text(json.dumps(result, indent=2, default=str))
        return result

    logger.info("  Matched %d comment(s) with Stage 3 tests", len(matched))

    # Determine language from first matched test
    language = "python"
    for entry in testgen_results.get("results", []):
        if entry.get("success"):
            language = entry.get("language", "python")
            break

    # Start container
    safe_name = instance_id.replace("/", "--").replace("@", "-")
    container_name = f"rb-agent-{safe_name}"

    # Remove any stale container
    rm_result = subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True, text=True,
    )
    if rm_result.returncode == 0 and rm_result.stdout.strip():
        time.sleep(2)

    # Mount credentials to a neutral path to avoid Docker pre-creating
    # /home/agent/ before the agent user is created (which breaks useradd).
    # setup_claude_in_container will copy them into the right location.
    volumes = []
    if credentials_path.exists():
        volumes.append(f"{credentials_path}:/etc/claude-credentials.json:ro")

    session = DockerContainerSession(
        docker_image,
        name=container_name,
        volumes=volumes,
    )

    resolutions: list[dict] = []

    try:
        session.start()
        logger.info("Started container %s (image: %s)", container_name, docker_image)

        # Setup Claude Code in container
        setup_claude_in_container(session)

        # Log comments being resolved
        for i in sorted(matched.keys()):
            comment = matched[i][0]
            logger.info("  Comment %d: [%s] %s", i, comment["path"], comment["text"][:80])

        # Resolve all comments in a single Claude invocation
        results = resolve_instance(
            instance=instance,
            matched_comments=matched,
            session=session,
            model=model,
            language=language,
        )

        for r in results:
            resolutions.append(r.to_dict())
            logger.info(
                "  Comment %d: %s (test=%s, error=%s)",
                r.comment_index,
                "RESOLVED" if r.resolved else "NOT RESOLVED",
                "PASS" if r.test_passed else "FAIL",
                r.error or "none",
            )

    finally:
        session.remove(force=True)
        logger.info("Removed container %s", container_name)

    # Compute stats
    num_comments = len(resolutions)
    num_resolved = sum(1 for r in resolutions if r["resolved"])
    resolution_rate = num_resolved / num_comments if num_comments > 0 else 0.0

    result = {
        "instance_id": instance_id,
        "repo": repo,
        "agent": "claude-code",
        "model": model,
        "num_comments": num_comments,
        "num_resolved": num_resolved,
        "resolution_rate": resolution_rate,
        "results": resolutions,
    }

    # Save result
    result_file = instance_dir / "result.json"
    result_file.write_text(json.dumps(result, indent=2, default=str))
    logger.info(
        "Result saved to %s (resolved: %d/%d = %.1f%%)",
        result_file, num_resolved, num_comments, resolution_rate * 100,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Resolve review comments using Claude Code in Docker"
    )
    parser.add_argument(
        "--instance-id", type=str, required=True,
        help="Instance ID to process",
    )
    parser.add_argument(
        "--stage3-file", type=str, default=DEFAULT_STAGE3_FILE,
        help=f"Stage 3 JSONL file (default: {DEFAULT_STAGE3_FILE})",
    )
    parser.add_argument(
        "--testgen-dir", type=str, default=DEFAULT_TESTGEN_DIR,
        help=f"Testgen results directory (default: {DEFAULT_TESTGEN_DIR})",
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

    # Get Docker image
    docker_image = get_docker_image_name(args.instance_id)

    # Check credentials
    if not credentials_path.exists():
        logger.warning(
            "Credentials file not found: %s (container will have no API access)",
            credentials_path,
        )

    t0 = time.time()
    result = process_instance(
        instance=instance,
        testgen_results=testgen_results,
        output_dir=output_dir,
        model=args.model,
        docker_image=docker_image,
        credentials_path=credentials_path,
    )
    elapsed = time.time() - t0

    logger.info(
        "=== DONE === %s: %d/%d resolved (%.1f%%) in %.1fs",
        args.instance_id,
        result["num_resolved"],
        result["num_comments"],
        result["resolution_rate"] * 100,
        elapsed,
    )


if __name__ == "__main__":
    main()
