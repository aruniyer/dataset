"""Build stage2_docker_image.jsonl from stage1_comment_filter.jsonl.

The stage2 filter keeps only stage1 rows whose instances have image-test
status `ok` or `skipped` in execution/assets/swe_care_images_test_report.json.
If the report is missing, this script runs the image test module to generate it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = REPO_ROOT / "execution" / "assets" / "swe_care_images_test_report.json"
DEFAULT_STAGE1_PATH = REPO_ROOT / "results_pipeline_funnel" / "stage1_comment_filter.jsonl"
DEFAULT_STAGE2_PATH = REPO_ROOT / "results_pipeline_funnel" / "stage2_docker_image.jsonl"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "results_pipeline_funnel" / "summary.json"
DEFAULT_TEST_PATH = REPO_ROOT / "execution" / "tests" / "test_swe_care_images.py"


def _tail(text: str | None, *, limit: int = 2000) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]


def _canonical_image_instance_id(raw: str) -> str | None:
    value = raw.strip()
    if not value:
        return None

    # Stage1 uses `Org__Repo-PR@commit`; image report uses `reviewbench/org__repo-pr`.
    base = value.split("@", 1)[0].strip().lower()
    if not base:
        return None
    if base.startswith("reviewbench/"):
        return base
    return f"reviewbench/{base}"


def _canonical_image_instance_from_repo_pull(repo: object, pull_number: object) -> str | None:
    if not isinstance(repo, str) or "/" not in repo:
        return None

    pull_value: int | None = None
    if isinstance(pull_number, int):
        pull_value = pull_number
    elif isinstance(pull_number, str) and pull_number.strip().isdigit():
        pull_value = int(pull_number.strip())
    if pull_value is None:
        return None

    return f"reviewbench/{repo.replace('/', '__').lower()}-{pull_value}"


def _ensure_report_exists(report_path: Path, test_path: Path) -> None:
    if report_path.exists():
        return

    cmd = [sys.executable, "-m", "pytest", str(test_path)]
    print(f"[make_dataset] Report not found at {report_path}, running: {' '.join(cmd)}")
    run = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if not report_path.exists():
        detail = _tail(run.stderr) or _tail(run.stdout) or f"pytest exit code {run.returncode}"
        raise RuntimeError(f"Could not generate report file: {report_path}\n{detail}")

    if run.returncode != 0:
        detail = _tail(run.stderr) or _tail(run.stdout)
        print(
            "[make_dataset] pytest exited non-zero but report file was generated; "
            "continuing.\n"
            f"{detail}",
            file=sys.stderr,
        )


def _load_allowed_instances(report_path: Path) -> set[str]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("report payload is missing list field 'results'")

    allowed: set[str] = set()
    for entry in results:
        if not isinstance(entry, dict):
            continue
        status_raw = entry.get("status")
        if not isinstance(status_raw, str):
            continue
        status = status_raw.strip().lower()
        if status != "ok" and not status.startswith("skipped"):
            continue

        instance_id_raw = entry.get("instance_id")
        if not isinstance(instance_id_raw, str):
            continue
        canonical = _canonical_image_instance_id(instance_id_raw)
        if canonical is not None:
            allowed.add(canonical)
    return allowed


def _filter_stage1(
    stage1_path: Path,
    stage2_path: Path,
    *,
    allowed_instances: set[str],
) -> tuple[int, int]:
    if not stage1_path.exists():
        raise FileNotFoundError(f"Missing input file: {stage1_path}")

    total_rows = 0
    kept_rows = 0
    stage2_path.parent.mkdir(parents=True, exist_ok=True)

    with stage1_path.open("r", encoding="utf-8") as src, stage2_path.open(
        "w",
        encoding="utf-8",
    ) as dst:
        for line_no, raw_line in enumerate(src, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            total_rows += 1

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {stage1_path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object JSON at {stage1_path}:{line_no}")

            candidates: set[str] = set()
            instance_id = row.get("instance_id")
            if isinstance(instance_id, str):
                canonical = _canonical_image_instance_id(instance_id)
                if canonical is not None:
                    candidates.add(canonical)

            repo_pull_canonical = _canonical_image_instance_from_repo_pull(
                row.get("repo"),
                row.get("pull_number"),
            )
            if repo_pull_canonical is not None:
                candidates.add(repo_pull_canonical)

            if not candidates.intersection(allowed_instances):
                continue

            dst.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            kept_rows += 1

    return total_rows, kept_rows


def _compute_jsonl_stage_stats(path: Path) -> tuple[int, int]:
    instances: set[str] = set()
    comments = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object JSON at {path}:{line_no}")

            instance_id = row.get("instance_id")
            if isinstance(instance_id, str) and instance_id:
                instances.add(instance_id)

            reference_comments = row.get("reference_review_comments")
            if isinstance(reference_comments, list):
                comments += len(reference_comments)

    return len(instances), comments


def _update_stage2_summary(summary_path: Path, *, instances: int, comments: int) -> None:
    if summary_path.exists():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid summary payload (expected object): {summary_path}")

    stages = raw.get("stages")
    if stages is None:
        stages = []
        raw["stages"] = stages
    if not isinstance(stages, list):
        raise ValueError(f"Invalid summary payload (expected list 'stages'): {summary_path}")

    updated = False
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        if stage.get("name") != "stage2_docker_image":
            continue
        stage["instances"] = instances
        stage["comments"] = comments
        updated = True
        break

    if not updated:
        stages.append(
            {
                "name": "stage2_docker_image",
                "description": "Docker image available",
                "instances": instances,
                "comments": comments,
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(raw, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter stage1_comment_filter.jsonl to stage2_docker_image.jsonl using "
            "SWE-CARE docker image test report status."
        )
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Image test report JSON. Default: {DEFAULT_REPORT_PATH}",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help=f"Pytest file used to generate report if missing. Default: {DEFAULT_TEST_PATH}",
    )
    parser.add_argument(
        "--stage1-path",
        type=Path,
        default=DEFAULT_STAGE1_PATH,
        help=f"Input stage1 JSONL. Default: {DEFAULT_STAGE1_PATH}",
    )
    parser.add_argument(
        "--stage2-path",
        type=Path,
        default=DEFAULT_STAGE2_PATH,
        help=f"Output stage2 JSONL. Default: {DEFAULT_STAGE2_PATH}",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Funnel summary JSON updated for stage2 stats. Default: {DEFAULT_SUMMARY_PATH}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report_path.resolve()
    test_path = args.test_path.resolve()
    stage1_path = args.stage1_path.resolve()
    stage2_path = args.stage2_path.resolve()
    summary_path = args.summary_path.resolve()

    _ensure_report_exists(report_path, test_path)
    allowed_instances = _load_allowed_instances(report_path)
    if not allowed_instances:
        raise ValueError(
            "No allowed instance IDs found in report (expected status `ok` or `skipped`)."
        )

    total_rows, kept_rows = _filter_stage1(
        stage1_path,
        stage2_path,
        allowed_instances=allowed_instances,
    )
    stage2_instances, stage2_comments = _compute_jsonl_stage_stats(stage2_path)
    _update_stage2_summary(
        summary_path,
        instances=stage2_instances,
        comments=stage2_comments,
    )
    print(
        "[make_dataset] Wrote stage2 dataset: "
        f"{stage2_path} | kept {kept_rows}/{total_rows} rows | "
        f"allowed instances from report: {len(allowed_instances)} | "
        f"updated summary stage2: instances={stage2_instances}, comments={stage2_comments}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
