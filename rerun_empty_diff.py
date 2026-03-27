#!/usr/bin/env python3
"""Rerun agent resolution for instances with empty agent_diff.

Processes only the affected instances into a separate output directory,
then merges successful reruns back into the original results directory.

Usage:
  python rerun_empty_diff.py                    # dry-run: list affected instances
  python rerun_empty_diff.py --run              # rerun affected instances
  python rerun_empty_diff.py --run --workers 2  # parallel
  python rerun_empty_diff.py --merge            # merge rerun results into original
"""

import argparse
import json
import logging
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from execution.container_runtime import get_docker_image_name
from run_agent_resolution import (
    DEFAULT_CREDENTIALS,
    DEFAULT_MODEL,
    DEFAULT_STAGE3_FILE,
    DEFAULT_TESTGEN_DIR,
    load_stage3_instance,
    load_testgen_results,
    process_instance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ORIGINAL_DIR = Path("results_agent_resolution")
RERUN_DIR = Path("results_agent_resolution_rerun")


def find_empty_diff_instances(results_dir: Path) -> list[str]:
    """Find instance IDs where all results have empty agent_diff."""
    empty = []
    for d in sorted(results_dir.iterdir()):
        result_file = d / "result.json"
        if not result_file.exists():
            continue
        data = json.loads(result_file.read_text())
        results = data.get("results", [])
        if not results:
            continue
        has_nonempty = any(r.get("agent_diff", "").strip() for r in results)
        if not has_nonempty:
            empty.append(data["instance_id"])
    return empty


def run_reruns(
    instance_ids: list[str],
    output_dir: Path,
    model: str,
    workers: int,
    stage3_file: Path,
    testgen_dir: Path,
    credentials_path: Path,
):
    """Rerun agent resolution for the given instance IDs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    lock = threading.Lock()
    all_results = []
    start_time = time.time()

    def _process_one(iid: str) -> dict:
        instance = load_stage3_instance(stage3_file, iid)
        if instance is None:
            logger.error("Instance %s not found in %s", iid, stage3_file)
            return {
                "instance_id": iid, "repo": "", "agent": "claude-code",
                "model": model, "num_comments": 0, "num_resolved": 0,
                "resolution_rate": 0.0, "results": [],
                "error": "Instance not found in stage3 file",
            }

        testgen = load_testgen_results(testgen_dir, iid)
        if testgen is None:
            logger.error("Testgen results not found for %s", iid)
            return {
                "instance_id": iid, "repo": instance["repo"],
                "agent": "claude-code", "model": model,
                "num_comments": 0, "num_resolved": 0,
                "resolution_rate": 0.0, "results": [],
                "error": "No testgen results found",
            }

        docker_image = get_docker_image_name(iid)
        return process_instance(
            instance=instance,
            testgen_results=testgen,
            output_dir=output_dir,
            model=model,
            docker_image=docker_image,
            credentials_path=credentials_path,
        )

    logger.info("Rerunning %d instance(s) with %d worker(s)", len(instance_ids), workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_iid = {
            executor.submit(_process_one, iid): iid for iid in instance_ids
        }

        for future in as_completed(future_to_iid):
            iid = future_to_iid[future]
            try:
                result = future.result()
            except Exception:
                logger.exception("Error processing %s", iid)
                result = {
                    "instance_id": iid, "repo": "", "agent": "claude-code",
                    "model": model, "num_comments": 0, "num_resolved": 0,
                    "resolution_rate": 0.0, "results": [],
                    "error": "Processing failed",
                }

            with lock:
                all_results.append(result)
                processed += 1

            has_diff = any(
                r.get("agent_diff", "").strip()
                for r in result.get("results", [])
            )
            logger.info(
                "[%d/%d] %s: %d/%d resolved, has_diff=%s",
                processed, len(instance_ids), iid,
                result.get("num_resolved", 0),
                result.get("num_comments", 0),
                has_diff,
            )

    elapsed = time.time() - start_time
    logger.info("Rerun complete in %.0fs. %d instances processed.", elapsed, processed)

    # Summary
    still_empty = 0
    for r in all_results:
        has_diff = any(
            x.get("agent_diff", "").strip() for x in r.get("results", [])
        )
        if not has_diff:
            still_empty += 1
    logger.info(
        "Results: %d now have agent_diff, %d still empty",
        len(all_results) - still_empty, still_empty,
    )


def merge_results(original_dir: Path, rerun_dir: Path):
    """Merge rerun results into original dir, replacing only instances
    where the rerun produced a non-empty agent_diff."""
    if not rerun_dir.exists():
        logger.error("Rerun directory does not exist: %s", rerun_dir)
        sys.exit(1)

    replaced = 0
    skipped = 0

    for d in sorted(rerun_dir.iterdir()):
        if not d.is_dir():
            continue
        result_file = d / "result.json"
        if not result_file.exists():
            continue

        data = json.loads(result_file.read_text())
        has_diff = any(
            r.get("agent_diff", "").strip() for r in data.get("results", [])
        )

        if not has_diff:
            logger.info("Skipping %s (still empty agent_diff)", d.name)
            skipped += 1
            continue

        target = original_dir / d.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(d, target)
        replaced += 1
        logger.info("Replaced %s", d.name)

    logger.info(
        "Merge complete: %d replaced, %d skipped (still empty)", replaced, skipped
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rerun agent resolution for instances with empty agent_diff"
    )
    parser.add_argument("--run", action="store_true", help="Run the reprocessing")
    parser.add_argument("--merge", action="store_true", help="Merge rerun results into original dir")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--stage3-file", type=str, default=DEFAULT_STAGE3_FILE)
    parser.add_argument("--testgen-dir", type=str, default=DEFAULT_TESTGEN_DIR)
    parser.add_argument("--credentials", type=str, default=str(DEFAULT_CREDENTIALS))
    parser.add_argument("--original-dir", type=str, default=str(ORIGINAL_DIR))
    parser.add_argument("--rerun-dir", type=str, default=str(RERUN_DIR))

    args = parser.parse_args()
    original_dir = Path(args.original_dir)
    rerun_dir = Path(args.rerun_dir)

    if args.merge:
        merge_results(original_dir, rerun_dir)
        return

    # Find affected instances
    affected = find_empty_diff_instances(original_dir)
    logger.info("Found %d instances with empty agent_diff", len(affected))
    for iid in affected:
        logger.info("  %s", iid)

    if not args.run:
        logger.info("Dry run — pass --run to reprocess these instances")
        return

    run_reruns(
        instance_ids=affected,
        output_dir=rerun_dir,
        model=args.model,
        workers=args.workers,
        stage3_file=Path(args.stage3_file),
        testgen_dir=Path(args.testgen_dir),
        credentials_path=Path(args.credentials),
    )


if __name__ == "__main__":
    main()
