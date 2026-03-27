#!/usr/bin/env python3
"""Batch comment quality filtering across all SWE-CARE instances.

Classifies every review comment in the dataset as HIGH or LOW quality.
Supports resuming interrupted runs and writes progressive summaries.

Usage:
  python run_batch_filter.py                         # All dev instances
  python run_batch_filter.py --split dev test        # Both splits
  python run_batch_filter.py --instance-ids-file ids.txt  # Only listed IDs
  python run_batch_filter.py --no-resume             # Re-process everything
  python run_batch_filter.py --json-only             # No terminal tables
"""

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline import comment_filter, dataset_utils
from pipeline.llm_client import DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_instance_id_subset(ids_file: str) -> list[str]:
    """Load instance IDs from a text file (one ID per line), preserving order."""
    path = Path(ids_file)
    if not path.exists():
        raise FileNotFoundError(f"Instance ID file not found: {ids_file}")

    ids: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        ids.append(line)
    return ids


def load_existing_result(output_dir: Path, instance_id: str) -> dict | None:
    """Load an existing filter result for an instance, or None if not found."""
    safe_id = instance_id.replace("/", "__")
    result_file = output_dir / f"{safe_id}.json"
    if result_file.exists():
        try:
            return json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load existing result for %s: %s", instance_id, e)
    return None


def write_summary(output_dir: Path, all_results: list[dict], elapsed: float) -> dict:
    """Write summary.json with aggregate statistics."""
    summary = comment_filter.compute_summary(all_results)
    summary["elapsed_seconds"] = elapsed
    summary["total_errors"] = sum(1 for r in all_results if r.get("error"))

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch comment quality filtering across all SWE-CARE instances"
    )

    parser.add_argument(
        "--split", nargs="+", default=["dev"],
        choices=["dev", "test"],
        help="Dataset split(s) to process (default: dev)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results_filter",
        help="Where to save results (default: results_filter/)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-process all instances even if results already exist",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Suppress terminal table display, only write JSON",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show reasoning in terminal output",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--instance-ids-file", type=str, default=None,
        help="Path to .txt file with one instance_id per line to process only that subset",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    # Load instances from all requested splits
    all_instances = []
    for split in args.split:
        instances = dataset_utils.load_instances(split=split)
        logger.info("Loaded %d instances from split '%s'", len(instances), split)
        all_instances.extend(instances)

    if args.instance_ids_file:
        requested_ids = load_instance_id_subset(args.instance_ids_file)
        if not requested_ids:
            logger.error("No instance IDs found in file: %s", args.instance_ids_file)
            sys.exit(1)

        all_by_id = {inst["instance_id"]: inst for inst in all_instances}
        missing_ids = [iid for iid in requested_ids if iid not in all_by_id]
        all_instances = [all_by_id[iid] for iid in requested_ids if iid in all_by_id]

        logger.info(
            "Subset mode: %d requested IDs, %d found in loaded splits",
            len(requested_ids), len(all_instances),
        )
        if missing_ids:
            logger.warning(
                "Instance IDs not found in splits %s (%d): %s",
                ",".join(args.split), len(missing_ids), ", ".join(missing_ids),
            )

    if not all_instances:
        logger.error("No instances found.")
        sys.exit(1)

    # Check resumability
    to_process = []
    all_results: list[dict] = []

    for inst in all_instances:
        if resume:
            existing = load_existing_result(output_dir, inst["instance_id"])
            if existing is not None:
                all_results.append(existing)
                continue
        to_process.append(inst)

    skipped = len(all_instances) - len(to_process)
    if skipped > 0:
        logger.info("Resuming: %d already done, %d to process", skipped, len(to_process))

    if not to_process:
        logger.info("All instances already processed.")
        total_elapsed = 0.0
        summary = write_summary(output_dir, all_results, total_elapsed)
        logger.info(
            "Summary: %d instances, %d comments, %d high (%.1f%%), %d low",
            summary["total_instances"], summary["total_comments"],
            summary["total_high_quality"], summary["high_quality_rate"] * 100,
            summary["total_low_quality"],
        )
        return

    workers = args.workers
    logger.info(
        "Processing %d instance(s) with %d worker(s) (splits: %s)",
        len(to_process), workers, ", ".join(args.split),
    )

    start_time = time.time()
    processed = 0
    lock = threading.Lock()

    def _process_one(inst: dict) -> dict:
        """Classify one instance and save its result JSON."""
        instance_id = inst["instance_id"]
        try:
            result = comment_filter.classify_comments(
                instance=inst,
                model=args.model,
            )
            safe_id = instance_id.replace("/", "__")
            result_file = output_dir / f"{safe_id}.json"
            result_file.write_text(json.dumps(result, indent=2, default=str))
            return result
        except Exception:
            logger.exception("Error processing %s", instance_id)
            return {
                "instance_id": instance_id,
                "repo": inst["repo"],
                "num_comments": len(inst.get("reference_review_comments", [])),
                "num_high_quality": 0,
                "num_low_quality": 0,
                "comments": [],
                "error": "Processing failed",
            }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_inst = {
            executor.submit(_process_one, inst): inst for inst in to_process
        }

        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            t0 = time.time()
            result = future.result()

            with lock:
                all_results.append(result)
                processed += 1
                count = processed

                if not args.json_only and not result.get("error"):
                    print(comment_filter.format_instance_results(result, verbose=args.verbose))

            logger.info(
                "[%d/%d] Completed %s",
                skipped + count, len(all_instances), inst["instance_id"],
            )

            # Write progressive summary every 50 instances
            if count % 50 == 0:
                with lock:
                    elapsed_so_far = time.time() - start_time
                    summary = write_summary(output_dir, list(all_results), elapsed_so_far)
                logger.info(
                    "Progress: %d/%d instances, %d high (%.1f%%), %d errors, cost $%.4f",
                    summary["total_instances"], len(all_instances),
                    summary["total_high_quality"], summary["high_quality_rate"] * 100,
                    summary["total_errors"],
                    summary.get("usage", {}).get("cost_usd", 0.0),
                )

    # Final summary
    total_elapsed = time.time() - start_time
    summary = write_summary(output_dir, all_results, total_elapsed)

    logger.info(
        "=== DONE === %d instances, %d comments, %d high quality (%.1f%%), "
        "%d low quality, %d errors, %.0fs elapsed, cost $%.4f",
        summary["total_instances"],
        summary["total_comments"],
        summary["total_high_quality"],
        summary["high_quality_rate"] * 100,
        summary["total_low_quality"],
        summary["total_errors"],
        total_elapsed,
        summary.get("usage", {}).get("cost_usd", 0.0),
    )
    logger.info("Summary: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
