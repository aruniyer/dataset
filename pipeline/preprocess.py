"""Modular preprocessing utilities for SWE-CARE instances.

Preprocessing keeps only HIGH-quality comments from filter outputs and emits
instance-level data where `reference_review_comments` has been reduced to the
retained subset.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

DEFAULT_STEPS = ("select_high", "validate_comments", "drop_empty")
VALID_MISMATCH_POLICIES = {"warn_skip", "fail_fast"}


class PreprocessError(RuntimeError):
    """Raised when preprocessing cannot continue."""


@dataclass
class PreprocessRecord:
    """Per-instance preprocessing state passed through modular steps."""

    instance_id: str
    filter_result: dict[str, Any]
    dataset_instance: dict[str, Any] | None
    mismatch_policy: str
    high_filter_comments: list[dict[str, Any]] = field(default_factory=list)
    kept_comments: list[dict[str, Any]] = field(default_factory=list)
    kept_comment_indices: list[int] = field(default_factory=list)


@dataclass
class RunReport:
    """Aggregated preprocessing metrics and audit details."""

    created_at: str
    dataset: str
    split: str
    filter_dir: str
    output_dir: str
    steps: list[str]
    mismatch_policy: str
    total_filter_files: int = 0
    total_filter_instances: int = 0
    total_dataset_instances: int = 0
    instances_processed: int = 0
    instances_kept: int = 0
    instances_dropped_missing_dataset: int = 0
    instances_dropped_zero_high: int = 0
    comments_seen: int = 0
    comments_high_candidates: int = 0
    comments_low_dropped: int = 0
    comments_kept: int = 0
    comments_dropped_mismatch: int = 0
    missing_dataset_instance_ids: list[str] = field(default_factory=list)
    dropped_instance_ids: list[str] = field(default_factory=list)
    dropped_comments: list[dict[str, Any]] = field(default_factory=list)
    kept_instance_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report payload."""
        return asdict(self)


def build_run_report(
    dataset: str,
    split: str,
    filter_dir: Path,
    output_dir: Path,
    steps: Iterable[str],
    mismatch_policy: str,
) -> RunReport:
    """Create a run report initialized with static configuration."""
    return RunReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset,
        split=split,
        filter_dir=str(filter_dir),
        output_dir=str(output_dir),
        steps=list(steps),
        mismatch_policy=mismatch_policy,
    )


def parse_step_names(steps: str | Iterable[str]) -> list[str]:
    """Parse and validate preprocessing steps."""
    if isinstance(steps, str):
        parsed = [step.strip() for step in steps.split(",") if step.strip()]
    else:
        parsed = [str(step).strip() for step in steps if str(step).strip()]

    if not parsed:
        raise ValueError("No preprocessing steps provided.")

    unknown = [step for step in parsed if step not in STEP_REGISTRY]
    if unknown:
        unknown_text = ", ".join(sorted(set(unknown)))
        raise ValueError(f"Unknown preprocessing step(s): {unknown_text}")

    return parsed


def load_filter_results(filter_dir: Path) -> dict[str, dict[str, Any]]:
    """Load per-instance filter outputs keyed by instance_id."""
    if not filter_dir.exists():
        raise FileNotFoundError(f"Filter directory does not exist: {filter_dir}")

    results: dict[str, dict[str, Any]] = {}
    for json_path in sorted(filter_dir.glob("*.json")):
        if json_path.name == "summary.json":
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        instance_id = data.get("instance_id")
        if not instance_id:
            raise ValueError(f"Missing instance_id in filter file: {json_path}")
        if instance_id in results:
            raise ValueError(f"Duplicate filter result for instance_id: {instance_id}")

        results[instance_id] = data

    return results


def load_dataset_instances(dataset: str, split: str) -> dict[str, dict[str, Any]]:
    """Load SWE-CARE rows for a split and map by instance_id."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load SWE-CARE. "
            "Install project dependencies (e.g., `uv sync`) before running preprocessing."
        ) from exc

    ds = load_dataset(dataset, split=split)
    by_instance: dict[str, dict[str, Any]] = {}
    for row in ds:
        row_dict = dict(row)
        instance_id = row_dict.get("instance_id")
        if not instance_id:
            continue
        by_instance[instance_id] = row_dict
    return by_instance


def _parse_comment_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _comment_sort_key(comment: dict[str, Any]) -> tuple[int, int]:
    idx = _parse_comment_index(comment.get("comment_index"))
    if idx is None:
        return (1, 0)
    return (0, idx)


def _handle_bad_comment(
    record: PreprocessRecord,
    report: RunReport,
    filter_comment: dict[str, Any],
    reason: str,
) -> None:
    dropped = {
        "instance_id": record.instance_id,
        "comment_index": filter_comment.get("comment_index"),
        "reason": reason,
        "path": filter_comment.get("path", ""),
        "text": filter_comment.get("text", ""),
    }
    report.comments_dropped_mismatch += 1
    report.dropped_comments.append(dropped)

    if record.mismatch_policy == "fail_fast":
        raise PreprocessError(
            f"{record.instance_id}: dropped comment_index="
            f"{filter_comment.get('comment_index')} ({reason})"
        )


def step_select_high(
    record: PreprocessRecord,
    report: RunReport,
) -> PreprocessRecord:
    """Keep only HIGH comments from filter output."""
    comments = record.filter_result.get("comments", [])
    report.comments_seen += len(comments)

    high_comments: list[dict[str, Any]] = []
    for comment in comments:
        quality = str(comment.get("quality", "")).upper()
        if quality == "HIGH":
            high_comments.append(comment)
        else:
            report.comments_low_dropped += 1

    report.comments_high_candidates += len(high_comments)
    record.high_filter_comments = sorted(high_comments, key=_comment_sort_key)
    return record


def step_validate_comments(
    record: PreprocessRecord,
    report: RunReport,
) -> PreprocessRecord | None:
    """Match retained HIGH comments to dataset comments by comment_index."""
    if record.dataset_instance is None:
        report.instances_dropped_missing_dataset += 1
        report.missing_dataset_instance_ids.append(record.instance_id)
        return None

    dataset_comments = record.dataset_instance.get("reference_review_comments") or []
    kept_comments: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    seen_indices: set[int] = set()

    for high_comment in record.high_filter_comments:
        idx = _parse_comment_index(high_comment.get("comment_index"))
        if idx is None:
            _handle_bad_comment(record, report, high_comment, "invalid_comment_index")
            continue
        if idx in seen_indices:
            _handle_bad_comment(record, report, high_comment, "duplicate_comment_index")
            continue
        if idx < 0 or idx >= len(dataset_comments):
            _handle_bad_comment(record, report, high_comment, "comment_index_out_of_range")
            continue

        dataset_comment = dataset_comments[idx]
        filter_path = high_comment.get("path") or ""
        dataset_path = dataset_comment.get("path") or ""
        if filter_path and filter_path != dataset_path:
            _handle_bad_comment(record, report, high_comment, "path_mismatch")
            continue

        filter_text = high_comment.get("text") or ""
        dataset_text = dataset_comment.get("text") or ""
        if filter_text and filter_text != dataset_text:
            _handle_bad_comment(record, report, high_comment, "text_mismatch")
            continue

        kept_comments.append(copy.deepcopy(dataset_comment))
        kept_indices.append(idx)
        seen_indices.add(idx)

    record.kept_comments = kept_comments
    record.kept_comment_indices = kept_indices
    report.comments_kept += len(kept_comments)
    return record


def step_drop_empty(
    record: PreprocessRecord,
    report: RunReport,
) -> PreprocessRecord | None:
    """Drop instances with no retained comments."""
    if not record.kept_comments:
        report.instances_dropped_zero_high += 1
        return None
    return record


StepFunction = Callable[[PreprocessRecord, RunReport], PreprocessRecord | None]

STEP_REGISTRY: dict[str, StepFunction] = {
    "select_high": step_select_high,
    "validate_comments": step_validate_comments,
    "drop_empty": step_drop_empty,
}


def run_steps(
    record: PreprocessRecord,
    report: RunReport,
    step_names: Iterable[str],
) -> PreprocessRecord | None:
    """Run configured preprocessing steps for one instance."""
    current: PreprocessRecord | None = record
    for step_name in step_names:
        step_fn = STEP_REGISTRY[step_name]
        if current is None:
            return None
        current = step_fn(current, report)
    return current


def preprocess_instances(
    filter_results: dict[str, dict[str, Any]],
    dataset_instances: dict[str, dict[str, Any]],
    step_names: Iterable[str],
    mismatch_policy: str,
    report: RunReport,
) -> tuple[list[dict[str, Any]], list[str], RunReport]:
    """Preprocess instances from loaded maps."""
    if mismatch_policy not in VALID_MISMATCH_POLICIES:
        raise ValueError(
            f"Invalid mismatch policy '{mismatch_policy}'. "
            f"Expected one of: {sorted(VALID_MISMATCH_POLICIES)}"
        )

    steps = parse_step_names(step_names)
    report.total_filter_files = len(filter_results)
    report.total_filter_instances = len(filter_results)
    report.total_dataset_instances = len(dataset_instances)

    output_instances: list[dict[str, Any]] = []
    dropped_ids: set[str] = set(report.dropped_instance_ids)

    for instance_id in sorted(filter_results):
        report.instances_processed += 1
        record = PreprocessRecord(
            instance_id=instance_id,
            filter_result=filter_results[instance_id],
            dataset_instance=dataset_instances.get(instance_id),
            mismatch_policy=mismatch_policy,
        )

        processed = run_steps(record, report, steps)
        if processed is None:
            if instance_id not in dropped_ids:
                report.dropped_instance_ids.append(instance_id)
                dropped_ids.add(instance_id)
            continue

        if processed.dataset_instance is None:
            report.instances_dropped_missing_dataset += 1
            report.missing_dataset_instance_ids.append(instance_id)
            if instance_id not in dropped_ids:
                report.dropped_instance_ids.append(instance_id)
                dropped_ids.add(instance_id)
            continue

        output_instance = copy.deepcopy(processed.dataset_instance)
        output_instance["reference_review_comments"] = processed.kept_comments
        output_instances.append(output_instance)

    output_instances.sort(key=lambda row: row.get("instance_id", ""))
    kept_instance_ids = [row["instance_id"] for row in output_instances if row.get("instance_id")]

    report.instances_kept = len(output_instances)
    report.kept_instance_ids = kept_instance_ids
    report.missing_dataset_instance_ids = sorted(set(report.missing_dataset_instance_ids))
    report.dropped_instance_ids = sorted(set(report.dropped_instance_ids))
    report.dropped_comments.sort(
        key=lambda item: (
            item.get("instance_id", ""),
            _parse_comment_index(item.get("comment_index")) or -1,
            item.get("reason", ""),
        )
    )

    return output_instances, kept_instance_ids, report


def run_preprocessing(
    dataset: str,
    split: str,
    filter_dir: Path,
    output_dir: Path,
    steps: str | Iterable[str] = DEFAULT_STEPS,
    mismatch_policy: str = "warn_skip",
) -> tuple[list[dict[str, Any]], list[str], RunReport]:
    """Load inputs and preprocess into cleaned instance rows."""
    step_names = parse_step_names(steps)
    report = build_run_report(
        dataset=dataset,
        split=split,
        filter_dir=filter_dir,
        output_dir=output_dir,
        steps=step_names,
        mismatch_policy=mismatch_policy,
    )

    filter_results = load_filter_results(filter_dir)
    dataset_instances = load_dataset_instances(dataset, split)
    return preprocess_instances(
        filter_results=filter_results,
        dataset_instances=dataset_instances,
        step_names=step_names,
        mismatch_policy=mismatch_policy,
        report=report,
    )


def write_instances_jsonl(path: Path, instances: Iterable[dict[str, Any]]) -> None:
    """Write cleaned instances as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for instance in instances:
            handle.write(json.dumps(instance, ensure_ascii=False, default=str))
            handle.write("\n")


def write_instance_ids(
    path: Path,
    instance_ids: Iterable[str],
    prefix: str | None = None,
) -> None:
    """Write kept instance IDs, one per line."""
    normalized_prefix = ""
    if prefix:
        normalized_prefix = prefix.strip().strip("/")

    def _format_id(instance_id: str) -> str:
        raw_id = instance_id.strip()
        if not normalized_prefix:
            return raw_id.lower()
        prefix_token = f"{normalized_prefix}/"
        if raw_id.startswith(prefix_token):
            return raw_id.lower()
        return f"{prefix_token}{raw_id}".lower()

    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [_format_id(instance_id) for instance_id in instance_ids]
    body = "\n".join(ids)
    if body:
        body += "\n"
    path.write_text(body, encoding="utf-8")


def write_report(path: Path, report: RunReport) -> None:
    """Write JSON report for auditing preprocessing behavior."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
