#!/usr/bin/env python3
"""Preprocess SWE-CARE by removing LOW comments and keeping HIGH comments."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import preprocess

DEFAULT_DATASET = "inclusionAI/SWE-CARE"
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FILTER_DIR = _SCRIPT_DIR / "results_filter"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "results_preprocessed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess SWE-CARE instances using filter results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Dataset split (default: dev)",
    )
    parser.add_argument(
        "--filter-dir",
        type=Path,
        default=DEFAULT_FILTER_DIR,
        help=f"Directory containing filter JSON files (default: {DEFAULT_FILTER_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="select_high,validate_comments,drop_empty",
        help="Comma-separated preprocessing steps (default: select_high,validate_comments,drop_empty)",
    )
    parser.add_argument(
        "--mismatch-policy",
        type=str,
        default="warn_skip",
        choices=["warn_skip", "fail_fast"],
        help="Mismatch handling policy (default: warn_skip)",
    )
    parser.add_argument(
        "--jsonl-name",
        type=str,
        default="swe_care_high_instances.jsonl",
        help="Filename for cleaned instance JSONL output",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="preprocess_report.json",
        help="Filename for JSON report output",
    )
    parser.add_argument(
        "--instance-ids-name",
        type=str,
        default="instance_ids_high.txt",
        help="Filename for kept instance-id list",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    step_names = preprocess.parse_step_names(args.steps)

    instances, kept_ids, report = preprocess.run_preprocessing(
        dataset=args.dataset,
        split=args.split,
        filter_dir=args.filter_dir,
        output_dir=args.output_dir,
        steps=step_names,
        mismatch_policy=args.mismatch_policy,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / args.jsonl_name
    report_path = args.output_dir / args.report_name
    ids_path = args.output_dir / args.instance_ids_name

    preprocess.write_instances_jsonl(jsonl_path, instances)
    preprocess.write_report(report_path, report)
    preprocess.write_instance_ids(ids_path, kept_ids, prefix="reviewbench")

    print("Preprocessing complete")
    print(f"Output JSONL: {jsonl_path}")
    print(f"Report JSON: {report_path}")
    print(f"Instance IDs: {ids_path}")
    print(
        "Stats: "
        f"processed={report.instances_processed}, "
        f"kept_instances={report.instances_kept}, "
        f"kept_comments={report.comments_kept}, "
        f"dropped_mismatched_comments={report.comments_dropped_mismatch}"
    )


if __name__ == "__main__":
    main()
