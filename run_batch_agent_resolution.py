#!/usr/bin/env python3
"""Batch agent resolution of code review comments using Claude Code in Docker.

Processes Stage 3 verified instances in parallel, invoking Claude Code inside
Docker containers to resolve each review comment and verifying against Stage 3
tests.

Usage:
  python run_batch_agent_resolution.py --limit 5 --workers 2
  python run_batch_agent_resolution.py --repo tobymao/sqlglot
  python run_batch_agent_resolution.py --no-resume
  python run_batch_agent_resolution.py --model claude-sonnet-4-6
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
from run_agent_resolution import (
    DEFAULT_CREDENTIALS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STAGE3_FILE,
    DEFAULT_TESTGEN_DIR,
    load_testgen_results,
    process_instance,
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


def write_summary(output_dir: Path, all_results: list[dict], elapsed: float) -> dict:
    """Write summary.json with aggregated stats."""
    total_comments = sum(r.get("num_comments", 0) for r in all_results)
    total_resolved = sum(r.get("num_resolved", 0) for r in all_results)
    total_errors = sum(1 for r in all_results if r.get("error"))

    # Per-repo aggregation
    repo_data: dict[str, dict] = {}
    for r in all_results:
        repo = r["repo"]
        if repo not in repo_data:
            repo_data[repo] = {
                "repo": repo,
                "instances": 0,
                "comments": 0,
                "resolved": 0,
            }
        rd = repo_data[repo]
        rd["instances"] += 1
        rd["comments"] += r.get("num_comments", 0)
        rd["resolved"] += r.get("num_resolved", 0)

    for rd in repo_data.values():
        rd["resolution_rate"] = (
            rd["resolved"] / rd["comments"] if rd["comments"] > 0 else 0.0
        )

    # Get model from first result
    model = ""
    for r in all_results:
        if r.get("model"):
            model = r["model"]
            break

    summary = {
        "total_instances": len(all_results),
        "total_comments": total_comments,
        "total_resolved": total_resolved,
        "total_errors": total_errors,
        "overall_resolution_rate": (
            total_resolved / total_comments if total_comments > 0 else 0.0
        ),
        "elapsed_seconds": elapsed,
        "model": model,
        "agent": "claude-code",
        "repo_summary": list(repo_data.values()),
        "instance_results": [
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "num_comments": r.get("num_comments", 0),
                "num_resolved": r.get("num_resolved", 0),
                "resolution_rate": r.get("resolution_rate", 0.0),
                "error": r.get("error"),
            }
            for r in all_results
        ],
    }

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def write_stage4_jsonl(
    output_dir: Path,
    stage3_instances: list[dict],
    all_results: list[dict],
    funnel_dir: Path,
) -> int:
    """Write stage4_agent_resolved.jsonl — instances where at least one comment was resolved.

    Returns the number of instances written.
    """
    # Build lookup: instance_id -> result
    result_map = {r["instance_id"]: r for r in all_results}

    funnel_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = funnel_dir / "stage4_agent_resolved.jsonl"
    count = 0

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for instance in stage3_instances:
            iid = instance["instance_id"]
            result = result_map.get(iid)
            if result and result.get("num_resolved", 0) > 0:
                handle.write(json.dumps(instance, ensure_ascii=False, default=str))
                handle.write("\n")
                count += 1

    logger.info("Wrote %d instances to %s", count, jsonl_path)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Batch agent resolution of review comments using Claude Code in Docker"
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
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--repo", type=str, default=None,
        help="Only process instances for this repo",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of instances to process",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-process all instances even if results already exist",
    )
    parser.add_argument(
        "--funnel-dir", type=str, default="results_pipeline_funnel",
        help="Directory for pipeline funnel JSONL files (default: results_pipeline_funnel/)",
    )

    args = parser.parse_args()

    stage3_file = Path(args.stage3_file)
    testgen_dir = Path(args.testgen_dir)
    output_dir = Path(args.output_dir)
    credentials_path = Path(args.credentials)
    funnel_dir = Path(args.funnel_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    if not stage3_file.exists():
        logger.error("Stage 3 file not found: %s", stage3_file)
        sys.exit(1)

    # Phase 1: Load stage3 instances, check Docker images, check resume
    all_stage3 = load_stage3_instances(stage3_file)
    logger.info("Loaded %d instances from %s", len(all_stage3), stage3_file)

    # Filter by repo
    if args.repo:
        all_stage3 = [i for i in all_stage3 if i["repo"] == args.repo]
        logger.info("Filtered to %d instances for repo %s", len(all_stage3), args.repo)

    # Apply limit
    if args.limit and len(all_stage3) > args.limit:
        all_stage3 = all_stage3[:args.limit]
        logger.info("Limited to %d instances", len(all_stage3))

    if not all_stage3:
        logger.error("No instances to process.")
        sys.exit(1)

    # Check resume and Docker images
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
            all_results.append({
                "instance_id": iid,
                "repo": instance["repo"],
                "agent": "claude-code",
                "model": args.model,
                "num_comments": 0,
                "num_resolved": 0,
                "resolution_rate": 0.0,
                "results": [],
                "error": f"Docker image not found: {docker_image}",
            })
            continue

        # Check testgen results exist
        testgen = load_testgen_results(testgen_dir, iid)
        if testgen is None:
            logger.warning("No testgen results for %s, skipping", iid)
            all_results.append({
                "instance_id": iid,
                "repo": instance["repo"],
                "agent": "claude-code",
                "model": args.model,
                "num_comments": 0,
                "num_resolved": 0,
                "resolution_rate": 0.0,
                "results": [],
                "error": "No testgen results found",
            })
            continue

        to_process.append(instance)

    if not to_process:
        logger.info("All instances already processed or skipped.")
        total_elapsed = 0.0
        summary = write_summary(output_dir, all_results, total_elapsed)
        write_stage4_jsonl(output_dir, all_stage3, all_results, funnel_dir)
        logger.info(
            "Summary: %d instances, %d/%d resolved (%.1f%%)",
            summary["total_instances"],
            summary["total_resolved"],
            summary["total_comments"],
            summary["overall_resolution_rate"] * 100,
        )
        return

    # Phase 2: Parallel processing
    start_time = time.time()
    processed = 0
    lock = threading.Lock()

    logger.info(
        "Processing %d instance(s) with %d worker(s)",
        len(to_process), args.workers,
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
            docker_image = get_docker_image_name(iid)

            result = process_instance(
                instance=instance,
                testgen_results=testgen,
                output_dir=output_dir,
                model=args.model,
                docker_image=docker_image,
                credentials_path=credentials_path,
            )
            return result
        except Exception:
            logger.exception("[%s] Error processing instance", iid)
            return {
                "instance_id": iid,
                "repo": instance["repo"],
                "agent": "claude-code",
                "model": args.model,
                "num_comments": 0,
                "num_resolved": 0,
                "resolution_rate": 0.0,
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
                "[%d/%d] Completed %s: %d/%d resolved",
                count, len(to_process),
                inst["instance_id"],
                result.get("num_resolved", 0),
                result.get("num_comments", 0),
            )

            # Write progressive summary every 10 completions
            if count % 10 == 0:
                with lock:
                    elapsed_so_far = time.time() - start_time
                    summary = write_summary(output_dir, list(all_results), elapsed_so_far)
                logger.info(
                    "Progress: %d/%d instances, %d/%d resolved (%.1f%%)",
                    len(all_results), len(all_stage3),
                    summary["total_resolved"],
                    summary["total_comments"],
                    summary["overall_resolution_rate"] * 100,
                )

    # Final summary and stage4 JSONL
    total_elapsed = time.time() - start_time
    summary = write_summary(output_dir, all_results, total_elapsed)
    stage4_count = write_stage4_jsonl(output_dir, all_stage3, all_results, funnel_dir)

    logger.info(
        "=== DONE === %d instances, %d/%d resolved (%.1f%%), "
        "%d errors, %.0fs elapsed",
        summary["total_instances"],
        summary["total_resolved"],
        summary["total_comments"],
        summary["overall_resolution_rate"] * 100,
        summary["total_errors"],
        total_elapsed,
    )
    logger.info(
        "Stage 4 funnel: %d instances (from %d in Stage 3)",
        stage4_count, len(all_stage3),
    )
    logger.info("Summary: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
