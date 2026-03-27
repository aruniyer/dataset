#!/usr/bin/env python3
"""CLI entry point for the comment quality filter pipeline.

Classifies review comments as HIGH or LOW quality using an LLM,
helping identify which comments are worth generating tests for.

Usage:
  python filter_comments.py --instance-id <id>          # Single instance
  python filter_comments.py --repo tobymao/sqlglot       # All instances for a repo
  python filter_comments.py --split dev --limit 10       # Batch mode
  python filter_comments.py --instance-id <id> --print-prompt-only  # Print prompt and exit

Options:
  --output-dir filter_results/    Where to save results
  --model gpt-5.1              LLM model to use
  --print-prompt-only           Print prompt(s) only, skip inference
  --verbose                       Show reasoning in terminal output
  --json-only                     Suppress terminal display, only write JSON
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pipeline import comment_filter, dataset_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Classify review comment quality using LLM"
    )

    # Instance selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--instance-id", type=str, help="Single instance ID to process")
    group.add_argument("--repo", type=str, help="Process all instances for a repo")

    # Filtering
    parser.add_argument("--split", default="dev", choices=["dev", "test"],
                        help="Dataset split (default: dev)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of instances to process")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Filter by difficulty level")
    parser.add_argument("--max-comments", type=int, default=None,
                        help="Only process instances with at most N comments")

    # Output
    parser.add_argument("--output-dir", type=str, default="filter_results",
                        help="Where to save results (default: filter_results/)")
    parser.add_argument("--model", type=str, default=comment_filter.DEFAULT_MODEL,
                        help=f"LLM model to use (default: {comment_filter.DEFAULT_MODEL})")
    parser.add_argument("--print-prompt-only", action="store_true",
                        help="Print constructed prompt(s) and exit without LLM inference")
    parser.add_argument("--verbose", action="store_true",
                        help="Show reasoning in terminal output")
    parser.add_argument("--json-only", action="store_true",
                        help="Suppress terminal display, only write JSON")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load instances
    if args.instance_id:
        instance = dataset_utils.load_instance(args.instance_id, split=args.split)
        if instance is None:
            logger.error("Instance not found: %s", args.instance_id)
            sys.exit(1)
        instances = [instance]
    else:
        instances = dataset_utils.load_instances(
            split=args.split,
            repo=args.repo,
            difficulty=args.difficulty,
            max_comments=args.max_comments,
            limit=args.limit,
        )

    if not instances:
        logger.error("No instances matched the given filters.")
        sys.exit(1)

    if args.print_prompt_only:
        logger.info("Printing prompt(s) only for %d instance(s)", len(instances))
        for idx, inst in enumerate(instances):
            prompt = comment_filter._build_filter_prompt(  # noqa: SLF001
                inst["reference_review_comments"],
                inst["instance_id"],
            )
            if len(instances) > 1:
                print("=" * 80)
                print(f"PROMPT {idx + 1}/{len(instances)}: {inst['instance_id']}")
                print("=" * 80)
            print(prompt)
            if idx < len(instances) - 1:
                print()
        return

    logger.info("Processing %d instance(s)", len(instances))

    all_results = []

    for inst in instances:
        t0 = time.time()
        try:
            result = comment_filter.classify_comments(
                instance=inst,
                model=args.model,
            )
            all_results.append(result)

            # Display terminal output
            if not args.json_only:
                print(comment_filter.format_instance_results(result, verbose=args.verbose))

            # Save per-instance JSON
            safe_id = result["instance_id"].replace("/", "__")
            result_file = output_dir / f"{safe_id}.json"
            result_file.write_text(json.dumps(result, indent=2, default=str))
            logger.info("Saved %s", result_file)

        except Exception:
            logger.exception("Error processing %s", inst["instance_id"])
            all_results.append({
                "instance_id": inst["instance_id"],
                "repo": inst["repo"],
                "num_comments": len(inst["reference_review_comments"]),
                "num_high_quality": 0,
                "num_low_quality": 0,
                "comments": [],
                "error": "Processing failed",
            })

        elapsed = time.time() - t0
        logger.info("Completed %s in %.1fs", inst["instance_id"], elapsed)

    # Write summary
    summary = comment_filter.compute_summary(all_results)
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    # Print summary to terminal
    if not args.json_only:
        print("=" * 80)
        print(" SUMMARY")
        print("=" * 80)
        print(f"  Instances:    {summary['total_instances']}")
        print(f"  Comments:     {summary['total_comments']}")
        print(f"  High quality: {summary['total_high_quality']} ({summary['high_quality_rate']:.1%})")
        print(f"  Low quality:  {summary['total_low_quality']}")
        print()
        if summary["category_breakdown"]:
            print("  Category breakdown:")
            for cat, count in summary["category_breakdown"].items():
                print(f"    {cat:<40} {count}")
        print()

    logger.info("Summary saved to %s", summary_file)


if __name__ == "__main__":
    main()
