#!/usr/bin/env python3
"""Batch evaluation of a review tool using Claude Code in Docker.

Processes Stage 3 verified instances in parallel, invoking Claude Code inside
Docker containers to apply tool-generated findings and verifying against all
successful Stage 3 tests.

Usage:
  python run_batch_tool_eval.py --limit 5 --workers 2
  python run_batch_tool_eval.py --repo ansible/ansible
  python run_batch_tool_eval.py --tool pr-agent --no-resume
  python run_batch_tool_eval.py --model claude-sonnet-4-6
"""

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from execution.container_runtime import docker_image_exists, get_docker_image_name
from run_tool_eval import (
    DEFAULT_CREDENTIALS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STAGE3_FILE,
    DEFAULT_TESTGEN_DIR,
    DEFAULT_TOOL,
    DEFAULT_TOOL_RESULTS_DIR,
    load_stage3_instance,
    load_testgen_results,
    load_tool_findings,
    process_tool_instance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_stage3_instances(stage3_file: Path) -> list[dict]:
    """Load all instances from the stage3 JSONL file."""
    instances = []
    with stage3_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
    return instances


def load_existing_result(output_dir: Path, instance_id: str) -> dict | None:
    """Load an existing result.json for an instance, or None if not found."""
    slug = instance_id.replace("/", "__")
    result_file = output_dir / slug / "result.json"
    if result_file.exists():
        try:
            return json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load existing result for %s: %s", instance_id, e)
    return None


def write_summary(
    output_dir: Path,
    all_results: list[dict],
    elapsed: float,
    tool: str,
    agent: str = "claude-code",
) -> dict:
    """Write summary.json with aggregated stats."""
    total_tests = sum(r.get("num_tests", 0) for r in all_results)
    total_tests_passed = sum(r.get("num_tests_passed", 0) for r in all_results)
    total_errors = sum(1 for r in all_results if r.get("error"))

    # Per-repo aggregation
    repo_data: dict[str, dict] = {}
    for r in all_results:
        repo = r["repo"]
        if repo not in repo_data:
            repo_data[repo] = {
                "repo": repo,
                "instances": 0,
                "total_tests": 0,
                "total_tests_passed": 0,
            }
        rd = repo_data[repo]
        rd["instances"] += 1
        rd["total_tests"] += r.get("num_tests", 0)
        rd["total_tests_passed"] += r.get("num_tests_passed", 0)

    for rd in repo_data.values():
        rd["test_pass_rate"] = (
            rd["total_tests_passed"] / rd["total_tests"]
            if rd["total_tests"] > 0
            else 0.0
        )

    # Get model from first result
    model = ""
    for r in all_results:
        if r.get("model"):
            model = r["model"]
            break

    summary = {
        "total_instances": len(all_results),
        "total_tests": total_tests,
        "total_tests_passed": total_tests_passed,
        "test_pass_rate": (
            total_tests_passed / total_tests if total_tests > 0 else 0.0
        ),
        "total_errors": total_errors,
        "elapsed_seconds": elapsed,
        "model": model,
        "tool": tool,
        "agent": agent,
        "repo_summary": list(repo_data.values()),
        "instance_results": [
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "num_findings": r.get("num_findings", 0),
                "num_tests": r.get("num_tests", 0),
                "num_tests_passed": r.get("num_tests_passed", 0),
                "test_pass_rate": r.get("test_pass_rate", 0.0),
                "error": r.get("error"),
            }
            for r in all_results
        ],
    }

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of a review tool using Claude Code in Docker"
    )

    parser.add_argument(
        "--tool",
        type=str,
        default=DEFAULT_TOOL,
        help=f"Review tool name (default: {DEFAULT_TOOL})",
    )
    parser.add_argument(
        "--tool-results-dir",
        type=str,
        default=DEFAULT_TOOL_RESULTS_DIR,
        help=f"Directory with tool result.json files (default: {DEFAULT_TOOL_RESULTS_DIR})",
    )
    parser.add_argument(
        "--stage3-file",
        type=str,
        default=DEFAULT_STAGE3_FILE,
        help=f"Stage 3 JSONL file (default: {DEFAULT_STAGE3_FILE})",
    )
    parser.add_argument(
        "--testgen-dir",
        type=str,
        default=DEFAULT_TESTGEN_DIR,
        help=f"Testgen results directory (default: {DEFAULT_TESTGEN_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default=str(DEFAULT_CREDENTIALS),
        help=f"Path to Claude credentials file (default: {DEFAULT_CREDENTIALS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Only process instances for this repo",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of instances to process",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process all instances even if results already exist",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include instances where the tool had no findings (marked as 0 tests passed)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="claude-code",
        choices=["claude-code", "copilot"],
        help="Resolution agent to use (default: claude-code)",
    )

    args = parser.parse_args()

    stage3_file = Path(args.stage3_file)
    testgen_dir = Path(args.testgen_dir)
    tool_results_dir = Path(args.tool_results_dir)
    output_dir = Path(args.output_dir)
    credentials_path = Path(args.credentials)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    if not stage3_file.exists():
        logger.error("Stage 3 file not found: %s", stage3_file)
        sys.exit(1)

    # Load all stage3 instances
    all_stage3 = load_stage3_instances(stage3_file)
    logger.info("Loaded %d instances from %s", len(all_stage3), stage3_file)

    # Filter by repo
    if args.repo:
        all_stage3 = [i for i in all_stage3 if i["repo"] == args.repo]
        logger.info("Filtered to %d instances for repo %s", len(all_stage3), args.repo)

    # Apply limit
    if args.limit and len(all_stage3) > args.limit:
        all_stage3 = all_stage3[: args.limit]
        logger.info("Limited to %d instances", len(all_stage3))

    if not all_stage3:
        logger.error("No instances to process.")
        sys.exit(1)

    # Check resume, Docker images, and tool findings
    all_results: list[dict] = []
    to_process: list[dict] = []

    for instance in all_stage3:
        iid = instance["instance_id"]

        # Check resume
        if resume:
            existing = load_existing_result(output_dir, iid)
            if existing is not None:
                logger.info("Skipping (already done): %s", iid)
                all_results.append(existing)
                continue

        # Check Docker image
        docker_image = get_docker_image_name(iid)
        if not docker_image_exists(docker_image):
            logger.warning("No Docker image for %s (%s), skipping", iid, docker_image)
            all_results.append(
                {
                    "instance_id": iid,
                    "repo": instance["repo"],
                    "tool": args.tool,
                    "model": args.model,
                    "num_findings": 0,
                    "agent_diff": "",
                    "num_tests": 0,
                    "num_tests_passed": 0,
                    "test_pass_rate": 0.0,
                    "results": [],
                    "error": f"Docker image not found: {docker_image}",
                }
            )
            continue

        # Check testgen results
        testgen = load_testgen_results(testgen_dir, iid)
        if testgen is None:
            logger.warning("No testgen results for %s, skipping", iid)
            all_results.append(
                {
                    "instance_id": iid,
                    "repo": instance["repo"],
                    "tool": args.tool,
                    "model": args.model,
                    "num_findings": 0,
                    "agent_diff": "",
                    "num_tests": 0,
                    "num_tests_passed": 0,
                    "test_pass_rate": 0.0,
                    "results": [],
                    "error": "No testgen results found",
                }
            )
            continue

        # Check tool findings
        tool_data = load_tool_findings(tool_results_dir, iid, args.tool)
        if tool_data is None:
            logger.warning("No tool findings for %s, skipping", iid)
            all_results.append(
                {
                    "instance_id": iid,
                    "repo": instance["repo"],
                    "tool": args.tool,
                    "model": args.model,
                    "num_findings": 0,
                    "agent_diff": "",
                    "num_tests": 0,
                    "num_tests_passed": 0,
                    "test_pass_rate": 0.0,
                    "results": [],
                    "error": "No tool findings found",
                }
            )
            continue

        if not tool_data.get("success") or not tool_data.get("findings"):
            if args.include_empty:
                logger.info(
                    "Tool had no findings for %s (include-empty: recording as 0 tests)",
                    iid,
                )
                all_results.append(
                    {
                        "instance_id": iid,
                        "repo": instance["repo"],
                        "tool": args.tool,
                        "model": args.model,
                        "num_findings": 0,
                        "agent_diff": "",
                        "num_tests": 0,
                        "num_tests_passed": 0,
                        "test_pass_rate": 0.0,
                        "results": [],
                        "error": "Tool had no findings",
                    }
                )
            else:
                logger.warning(
                    "Tool had no findings for %s (success=%s), skipping",
                    iid,
                    tool_data.get("success"),
                )
            continue

        to_process.append(instance)

    if not to_process:
        logger.info("All instances already processed or skipped.")
        summary = write_summary(output_dir, all_results, 0.0, args.tool, args.agent)
        logger.info(
            "Summary: %d instances, %d/%d tests passed (%.1f%%)",
            summary["total_instances"],
            summary["total_tests_passed"],
            summary["total_tests"],
            summary["test_pass_rate"] * 100,
        )
        return

    # Phase 2: Parallel processing
    start_time = time.time()
    processed = 0
    lock = threading.Lock()

    logger.info(
        "Processing %d instance(s) with %d worker(s)",
        len(to_process),
        args.workers,
    )

    # Check credentials
    if not credentials_path.exists():
        logger.warning(
            "Credentials file not found: %s (containers will have no API access)",
            credentials_path,
        )

    def _process_one(instance: dict) -> dict:
        """Process a single instance."""
        iid = instance["instance_id"]
        try:
            testgen = load_testgen_results(testgen_dir, iid)
            tool_data = load_tool_findings(tool_results_dir, iid, args.tool)
            docker_image = get_docker_image_name(iid)

            result = process_tool_instance(
                instance=instance,
                tool_data=tool_data,
                testgen_results=testgen,
                output_dir=output_dir,
                model=args.model,
                tool=args.tool,
                docker_image=docker_image,
                credentials_path=credentials_path,
                agent=args.agent,
            )
            return result
        except Exception:
            logger.exception("[%s] Error processing instance", iid)
            return {
                "instance_id": iid,
                "repo": instance["repo"],
                "tool": args.tool,
                "model": args.model,
                "num_findings": 0,
                "agent_diff": "",
                "num_tests": 0,
                "num_tests_passed": 0,
                "test_pass_rate": 0.0,
                "results": [],
                "error": "Processing failed",
            }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_inst = {
            executor.submit(_process_one, inst): inst for inst in to_process
        }

        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            result = future.result()

            with lock:
                all_results.append(result)
                processed += 1
                count = processed

            logger.info(
                "[%d/%d] Completed %s: %d/%d tests passed",
                count,
                len(to_process),
                inst["instance_id"],
                result.get("num_tests_passed", 0),
                result.get("num_tests", 0),
            )

            # Write progressive summary every 10 completions
            if count % 10 == 0:
                with lock:
                    elapsed_so_far = time.time() - start_time
                    summary = write_summary(
                        output_dir,
                        list(all_results),
                        elapsed_so_far,
                        args.tool,
                        args.agent,
                    )
                logger.info(
                    "Progress: %d/%d instances, %d/%d tests passed (%.1f%%)",
                    len(all_results),
                    len(all_stage3),
                    summary["total_tests_passed"],
                    summary["total_tests"],
                    summary["test_pass_rate"] * 100,
                )

    # Final summary
    total_elapsed = time.time() - start_time
    summary = write_summary(
        output_dir, all_results, total_elapsed, args.tool, args.agent
    )

    logger.info(
        "=== DONE === %d instances, %d/%d tests passed (%.1f%%), "
        "%d errors, %.0fs elapsed",
        summary["total_instances"],
        summary["total_tests_passed"],
        summary["total_tests"],
        summary["test_pass_rate"] * 100,
        summary["total_errors"],
        total_elapsed,
    )
    logger.info("Summary: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
