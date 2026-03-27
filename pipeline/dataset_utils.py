"""Dataset loading utilities for the SWE-CARE dataset.

Provides functions to load individual instances or filtered batches
from the inclusionAI/SWE-CARE Hugging Face dataset.
"""

import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

_dataset_cache: dict[str, object] = {}


def _get_dataset(split: str = "dev"):
    """Load and cache the SWE-CARE dataset for a given split."""
    if split not in _dataset_cache:
        logger.info("Loading SWE-CARE dataset (split=%s)...", split)
        _dataset_cache[split] = load_dataset("inclusionAI/SWE-CARE", split=split)
        logger.info("Loaded %d instances.", len(_dataset_cache[split]))
    return _dataset_cache[split]


def load_instance(instance_id: str, split: str = "dev") -> dict | None:
    """Load a single instance by its instance_id.

    Returns:
        The instance dict, or None if not found.
    """
    ds = _get_dataset(split)
    for row in ds:
        if row["instance_id"] == instance_id:
            return row
    logger.warning("Instance '%s' not found in split '%s'", instance_id, split)
    return None


def load_instances(
    split: str = "dev",
    repo: str | None = None,
    difficulty: str | None = None,
    max_comments: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Load instances with optional filtering.

    Args:
        split: Dataset split ('dev' or 'test').
        repo: Filter by repository name (e.g. 'tobymao/sqlglot').
        difficulty: Filter by difficulty level.
        max_comments: Only include instances with at most this many comments.
        limit: Maximum number of instances to return.

    Returns:
        List of instance dicts matching the filters.
    """
    ds = _get_dataset(split)
    results = []

    for row in ds:
        if repo and row["repo"] != repo:
            continue
        if difficulty and row["metadata"]["difficulty"] != difficulty:
            continue
        if max_comments is not None:
            if len(row["reference_review_comments"]) > max_comments:
                continue
        results.append(row)
        if limit and len(results) >= limit:
            break

    logger.info(
        "Filtered %d instances (repo=%s, difficulty=%s, limit=%s)",
        len(results), repo, difficulty, limit,
    )
    return results


def get_instance_summary(instance: dict) -> str:
    """Return a human-readable one-line summary of an instance."""
    meta = instance["metadata"]
    num_comments = len(instance["reference_review_comments"])
    return (
        f"{instance['instance_id']} | "
        f"{instance['repo']} | "
        f"{meta['difficulty']} | "
        f"{num_comments} comment(s) | "
        f"{meta['problem_domain']}"
    )
