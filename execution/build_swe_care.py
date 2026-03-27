"""Build Docker images for SWE-CARE instances (test split for now)."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from .build_log_checks import analyze_build_log_text, has_python_retry_signal
from .docker_env import (
    DEFAULT_APT_PROXY,
    DEFAULT_PIP_EXTRA_INDEX_URL,
    DEFAULT_PIP_INDEX_URL,
    DEFAULT_PIP_TRUSTED_HOST,
    DEFAULT_SWE_CARE_BUILD_OVERRIDES_ROOT,
    BuildResult,
    build_image,
    resolve_build_script_overrides,
)
from .python_version import (
    DEFAULT_PYTHON_VERSION,
    DEFAULT_REPO_CACHE_DIR,
    SUPPORTED_PYTHON_VERSIONS,
    resolve_python_version,
)
from .specs import PREnvironmentSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildTask:
    """One unique Docker image build task."""

    task_id: str
    repo: str
    pull_number: int | None
    commit: str
    languages: tuple[str, ...]
    instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class LanguageOverrides:
    """Language override rules keyed by instance_id and repo."""

    instance_overrides: dict[str, str]
    repo_overrides: dict[str, str]

    @property
    def count(self) -> int:
        return len(self.instance_overrides) + len(self.repo_overrides)


_REPO_LOCKS: dict[str, threading.Lock] = {}
_REPO_LOCKS_GUARD = threading.Lock()
_TAG_COMPONENT_RE = re.compile(r"[^a-z0-9_.-]+")
_INSTANCE_SELECTOR_RE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+__[0-9a-fA-F]{7,}$")
_INSTANCE_IMAGE_ID_RE = re.compile(
    r"^(?:reviewbench/)?(?P<repo>[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+)-\d+@(?P<commit>[0-9a-fA-F]{7,})$"
)
_BUILD_LANGUAGE_PRIORITY = ("python", "javascript", "typescript")
DEFAULT_LANGUAGE_OVERRIDES_FILE = (
    Path(__file__).resolve().parent / "assets" / "swe_care_language_overrides.json"
)
DEFAULT_SWE_CARE_IGNORES_FILE = (
    Path(__file__).resolve().parent / "assets" / "swe_care_ignores.json"
)


@dataclass(frozen=True)
class DatasetIgnores:
    """Dataset ignore selectors keyed by instance_id and repo."""

    instance_ids: frozenset[str]
    repos: frozenset[str]

    @property
    def count(self) -> int:
        return len(self.instance_ids) + len(self.repos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Docker images for SWE-CARE dataset instances."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="inclusionAI/SWE-CARE",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test"],
        help="Dataset split (currently only test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of rows from the split.",
    )
    parser.add_argument(
        "--language-overrides-file",
        type=Path,
        default=DEFAULT_LANGUAGE_OVERRIDES_FILE,
        help=(
            "JSON file to monkey-patch dataset language values. Supports instance-level "
            "and repo-level overrides. Empty file means no overrides. "
            "Default: execution/assets/swe_care_language_overrides.json"
        ),
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help=(
            "Optional repo filter in owner/name format. "
            "When set, only dataset rows from this repo are built."
        ),
    )
    parser.add_argument(
        "--instance",
        dest="instances",
        action="append",
        default=None,
        help=(
            "Optional instance selector(s). Supports org__repo__hash, "
            "reviewbench/org__repo-pr@hash, and Org__Repo-pr@hash. "
            "Can be repeated and/or provided as comma-separated values. "
            "You may also pass a file path; non-empty lines in the file are parsed "
            "as selectors (comma-separated values per line are allowed)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/hf-datasets"),
        help="HuggingFace datasets cache dir.",
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=DEFAULT_REPO_CACHE_DIR,
        help=(
            "Local repo cache used during python-version inference "
            "and as local seed sources for docker build contexts."
        ),
    )
    parser.add_argument(
        "--build-overrides-root",
        type=Path,
        default=DEFAULT_SWE_CARE_BUILD_OVERRIDES_ROOT,
        help=(
            "Directory containing optional repo/commit-specific manual build overrides "
            "(.python-version, Dockerfile, setup_repo.sh, install_deps.sh)."
        ),
    )
    parser.add_argument(
        "--context-root",
        type=Path,
        default=Path(__file__).resolve().parent / "dockerfiles",
        help="Where generated Docker build contexts are written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "swe_care_builds",
        help="Output root for logs and result files.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max build threads.",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        default="auto",
        help="Python version override, or 'auto' (default).",
    )
    parser.add_argument(
        "--use-merged-commit",
        action="store_true",
        help="Use merged_commit instead of commit_to_review.head_commit.",
    )
    parser.add_argument(
        "--include-non-python",
        action="store_true",
        help="Include non-Python rows (default: skip).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Pass --no-cache to docker build.",
    )
    parser.add_argument(
        "--pull-base",
        action="store_true",
        help="Pass --pull to docker build.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if the image tag already exists.",
    )
    parser.add_argument(
        "--build-timeout-sec",
        type=int,
        default=7200,
        help="Timeout for one docker build in seconds (default: 7200). Set 0 to disable.",
    )
    parser.add_argument(
        "--apt-proxy",
        type=str,
        default=DEFAULT_APT_PROXY,
        help="Optional apt proxy URL (for example: http://host.docker.internal:3142).",
    )
    parser.add_argument(
        "--pip-index-url",
        type=str,
        default=DEFAULT_PIP_INDEX_URL,
        help="Optional pip index URL (for example: https://pypi.org/simple).",
    )
    parser.add_argument(
        "--pip-extra-index-url",
        type=str,
        default=DEFAULT_PIP_EXTRA_INDEX_URL,
        help="Optional extra pip index URL.",
    )
    parser.add_argument(
        "--pip-trusted-host",
        type=str,
        default=DEFAULT_PIP_TRUSTED_HOST,
        help="Optional pip trusted host list (space-separated).",
    )
    return parser.parse_args()


def _configure_logging(run_dir: Path) -> Path:
    run_log = run_dir / "run.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(run_log, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return run_log


def _get_repo_lock(repo: str) -> threading.Lock:
    with _REPO_LOCKS_GUARD:
        lock = _REPO_LOCKS.get(repo)
        if lock is None:
            lock = threading.Lock()
            _REPO_LOCKS[repo] = lock
        return lock


def _get_target_commit(instance: dict, use_merged_commit: bool) -> str | None:
    if use_merged_commit:
        return instance.get("merged_commit")
    return instance.get("commit_to_review", {}).get("head_commit")


def _normalize_split_expr(split: str, limit: int | None) -> str:
    if limit is None:
        return split
    return f"{split}[:{limit}]"


def _task_id_for(repo: str, commit: str) -> str:
    return f"{repo.replace('/', '__')}__{commit[:7]}"


def _normalize_instance_selector(raw_selector: str) -> str | None:
    selector = raw_selector.strip()
    if not selector:
        return None

    if _INSTANCE_SELECTOR_RE.fullmatch(selector) is not None:
        repo_part, commit_part = selector.rsplit("__", 1)
        return f"{repo_part}__{commit_part[:7]}"

    image_id_match = _INSTANCE_IMAGE_ID_RE.fullmatch(selector)
    if image_id_match is None:
        return None
    repo_part = image_id_match.group("repo")
    commit_part = image_id_match.group("commit")
    return f"{repo_part}__{commit_part[:7]}"


def _parse_instance_filters_from_file(path: Path) -> tuple[list[str], list[str]]:
    selectors: list[str] = []
    invalid: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                for token in stripped.split(","):
                    selector = token.strip()
                    if not selector:
                        continue
                    normalized = _normalize_instance_selector(selector)
                    if normalized is None:
                        invalid.append(f"{selector} ({path}:{line_no})")
                        continue
                    selectors.append(normalized)
    except OSError as exc:
        invalid.append(f"{path} ({exc})")
    return selectors, invalid


def _parse_instance_filters(raw_values: list[str] | None) -> tuple[tuple[str, ...], list[str]]:
    if not raw_values:
        return (), []

    selectors: list[str] = []
    invalid: list[str] = []
    for raw_value in raw_values:
        for token in raw_value.split(","):
            selector = token.strip()
            if not selector:
                continue

            path_candidate = Path(selector).expanduser()
            if path_candidate.is_file():
                from_file_selectors, from_file_invalid = _parse_instance_filters_from_file(
                    path_candidate
                )
                selectors.extend(from_file_selectors)
                invalid.extend(from_file_invalid)
                continue

            normalized = _normalize_instance_selector(selector)
            if normalized is None:
                invalid.append(selector)
                continue
            selectors.append(normalized)
    deduped_selectors = tuple(dict.fromkeys(selectors))
    return deduped_selectors, invalid


def _selector_for_row(row: dict, *, use_merged_commit: bool) -> str | None:
    repo = row.get("repo")
    commit = _get_target_commit(row, use_merged_commit)
    if not isinstance(repo, str) or "/" not in repo:
        return None
    if not isinstance(commit, str) or len(commit) < 7:
        return None
    return _task_id_for(repo, commit)


def _filter_rows_by_instance_selectors(
    rows: list[dict],
    selectors: tuple[str, ...],
    *,
    use_merged_commit: bool,
) -> tuple[list[dict], list[str]]:
    if not selectors:
        return rows, []

    selector_lookup = {selector.lower(): selector for selector in selectors}
    selected_rows: list[dict] = []
    matched_selector_keys: set[str] = set()
    for row in rows:
        instance_id = row.get("instance_id")
        if isinstance(instance_id, str):
            instance_key = instance_id.lower()
            if instance_key in selector_lookup:
                selected_rows.append(row)
                matched_selector_keys.add(instance_key)
                continue
        selector = _selector_for_row(row, use_merged_commit=use_merged_commit)
        if selector is not None:
            selector_key = selector.lower()
            if selector_key in selector_lookup:
                selected_rows.append(row)
                matched_selector_keys.add(selector_key)

    missing = [selector for selector in selectors if selector.lower() not in matched_selector_keys]
    return selected_rows, missing


def _normalize_repo_selector(value: str, *, source: str) -> str:
    selector = value.strip()
    if not selector:
        raise ValueError(f"{source} cannot be empty")

    if "/" in selector:
        owner, repo_name = selector.split("/", 1)
        if owner and repo_name:
            return f"{owner}/{repo_name}"
        raise ValueError(
            f"{source} must be in owner/name or owner__name format: {value!r}"
        )

    if selector.count("__") == 1:
        owner, repo_name = selector.split("__", 1)
        if owner and repo_name:
            return f"{owner}/{repo_name}"

    raise ValueError(f"{source} must be in owner/name or owner__name format: {value!r}")


def _parse_instance_ignore_entries(raw: object, *, source: str) -> set[str]:
    if isinstance(raw, dict):
        entries: list[object] = list(raw.keys())
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"{source} must be a list or object")

    parsed: set[str] = set()
    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            instance_id = entry.strip()
        elif isinstance(entry, dict):
            raw_instance_id = entry.get("instance_id", entry.get("id", entry.get("instance")))
            if not isinstance(raw_instance_id, str):
                raise ValueError(
                    f"{source} entry at index {idx} must include string instance_id/id/instance"
                )
            instance_id = raw_instance_id.strip()
        else:
            raise ValueError(f"{source} entry at index {idx} must be string or object")

        if not instance_id:
            raise ValueError(f"{source} entry at index {idx} has empty instance id")
        parsed.add(instance_id)
    return parsed


def _parse_repo_ignore_entries(raw: object, *, source: str) -> set[str]:
    if isinstance(raw, dict):
        entries: list[object] = list(raw.keys())
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"{source} must be a list or object")

    parsed: set[str] = set()
    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            repo_selector = entry
        elif isinstance(entry, dict):
            raw_repo = entry.get("repo", entry.get("repository"))
            if not isinstance(raw_repo, str):
                raise ValueError(
                    f"{source} entry at index {idx} must include string repo/repository"
                )
            repo_selector = raw_repo
        else:
            raise ValueError(f"{source} entry at index {idx} must be string or object")

        parsed.add(_normalize_repo_selector(repo_selector, source=f"{source}[{idx}]"))
    return parsed


def _parse_ignore_rule_list(raw: object, *, source: str) -> DatasetIgnores:
    if not isinstance(raw, list):
        raise ValueError(f"{source} must be a list")

    instance_ids: set[str] = set()
    repos: set[str] = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{source} entry at index {idx} must be an object")

        raw_instance_id = entry.get("instance_id", entry.get("id", entry.get("instance")))
        raw_repo = entry.get("repo", entry.get("repository"))
        has_instance = isinstance(raw_instance_id, str) and bool(raw_instance_id.strip())
        has_repo = isinstance(raw_repo, str) and bool(raw_repo.strip())
        if has_instance == has_repo:
            raise ValueError(
                f"{source} entry at index {idx} must specify exactly one selector "
                "(instance_id/id/instance or repo/repository)"
            )

        if has_instance:
            instance_ids.add(raw_instance_id.strip())
            continue

        repos.add(_normalize_repo_selector(raw_repo, source=f"{source}[{idx}]"))

    return DatasetIgnores(instance_ids=frozenset(instance_ids), repos=frozenset(repos))


def _parse_dataset_ignores(raw: object) -> DatasetIgnores:
    if raw is None:
        return DatasetIgnores(instance_ids=frozenset(), repos=frozenset())
    if isinstance(raw, list):
        return _parse_ignore_rule_list(raw, source="rules")
    if not isinstance(raw, dict):
        raise ValueError("SWE-CARE ignores file must be a JSON object or list")
    if not raw:
        return DatasetIgnores(instance_ids=frozenset(), repos=frozenset())

    instance_ids: set[str] = set()
    repos: set[str] = set()
    instance_container_keys = ("instance_ignores", "instance_ids", "instances")
    repo_container_keys = ("repo_ignores", "repos", "repository_ignores")
    has_container = any(key in raw for key in (*instance_container_keys, *repo_container_keys, "rules"))

    if has_container:
        allowed_keys = {*instance_container_keys, *repo_container_keys, "rules"}
        unknown_keys = sorted(
            key for key in raw.keys() if isinstance(key, str) and key not in allowed_keys
        )
        if unknown_keys:
            raise ValueError(
                "unknown top-level keys in SWE-CARE ignores file: "
                f"{', '.join(unknown_keys)}"
            )
        for key in instance_container_keys:
            if key in raw:
                instance_ids.update(_parse_instance_ignore_entries(raw[key], source=key))
        for key in repo_container_keys:
            if key in raw:
                repos.update(_parse_repo_ignore_entries(raw[key], source=key))
        if "rules" in raw:
            from_rules = _parse_ignore_rule_list(raw["rules"], source="rules")
            instance_ids.update(from_rules.instance_ids)
            repos.update(from_rules.repos)
        return DatasetIgnores(instance_ids=frozenset(instance_ids), repos=frozenset(repos))

    # Backward-compatible shape: object mapping selector -> reason.
    for selector in raw.keys():
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError("ignore mapping keys must be non-empty strings")
        normalized_selector = selector.strip()
        if _INSTANCE_SELECTOR_RE.fullmatch(normalized_selector):
            instance_ids.add(normalized_selector)
            continue
        repos.add(_normalize_repo_selector(normalized_selector, source="ignore_mapping"))
    return DatasetIgnores(instance_ids=frozenset(instance_ids), repos=frozenset(repos))


def _load_dataset_ignores(path: Path) -> DatasetIgnores:
    if not path.exists():
        return DatasetIgnores(instance_ids=frozenset(), repos=frozenset())
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return DatasetIgnores(instance_ids=frozenset(), repos=frozenset())
    raw = json.loads(raw_text)
    return _parse_dataset_ignores(raw)


def _apply_dataset_ignores(
    rows: list[dict],
    ignores: DatasetIgnores,
) -> tuple[list[dict], int, list[str], list[str]]:
    if ignores.count == 0:
        return rows, 0, [], []

    present_instance_ids = {
        instance_id
        for row in rows
        for instance_id in [row.get("instance_id")]
        if isinstance(instance_id, str) and instance_id
    }
    present_repos = {
        repo
        for row in rows
        for repo in [row.get("repo")]
        if isinstance(repo, str) and repo
    }

    missing_instance_ids = sorted(
        instance_id for instance_id in ignores.instance_ids if instance_id not in present_instance_ids
    )
    missing_repos = sorted(repo for repo in ignores.repos if repo not in present_repos)

    filtered_rows = [
        row
        for row in rows
        if row.get("instance_id") not in ignores.instance_ids and row.get("repo") not in ignores.repos
    ]
    ignored_count = len(rows) - len(filtered_rows)
    return filtered_rows, ignored_count, missing_instance_ids, missing_repos


def _normalize_tag_component(value: str) -> str:
    return _TAG_COMPONENT_RE.sub("-", value.strip().lower())


def _infer_task_languages(raw_language: str) -> tuple[str, ...]:
    normalized = raw_language.strip().lower()
    detected: set[str] = set()
    alias = {
        "python": "python",
        "py": "python",
        "javascript": "javascript",
        "js": "javascript",
        "node": "javascript",
        "nodejs": "javascript",
        "ecmascript": "javascript",
        "typescript": "typescript",
        "ts": "typescript",
        "tsx": "typescript",
    }
    for token in re.findall(r"[a-z]+", normalized):
        lang = alias.get(token)
        if lang is not None:
            detected.add(lang)
    ordered = tuple(lang for lang in _BUILD_LANGUAGE_PRIORITY if lang in detected)
    return ordered


def _parse_override_language_value(value: object, *, selector: str) -> str:
    if isinstance(value, str):
        language = value.strip()
    elif isinstance(value, dict):
        language_value = value.get("language", value.get("languages"))
        if not isinstance(language_value, str):
            raise ValueError(
                f"language override for '{selector}' must contain a string language"
            )
        language = language_value.strip()
    else:
        raise ValueError(
            f"language override for '{selector}' must be a string or object"
        )
    if not language:
        raise ValueError(f"language override for '{selector}' cannot be empty")
    return language


def _parse_named_overrides(
    raw: object, *, selector_name: str, enforce_repo_format: bool
) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"{selector_name} overrides must be a JSON object")

    parsed: dict[str, str] = {}
    for selector, value in raw.items():
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError(f"{selector_name} override keys must be non-empty strings")
        normalized_selector = selector.strip()
        if enforce_repo_format and "/" not in normalized_selector:
            raise ValueError(
                f"repo override key must be in owner/name format: {normalized_selector!r}"
            )
        parsed[normalized_selector] = _parse_override_language_value(
            value, selector=normalized_selector
        )
    return parsed


def _parse_rule_list_overrides(raw: object) -> LanguageOverrides:
    if not isinstance(raw, list):
        raise ValueError("language overrides list must be a JSON array")

    instance_overrides: dict[str, str] = {}
    repo_overrides: dict[str, str] = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"language overrides list item at index {idx} must be an object"
            )
        instance_id = item.get("instance_id", item.get("id"))
        repo = item.get("repo", item.get("repository"))
        has_instance = isinstance(instance_id, str) and bool(instance_id.strip())
        has_repo = isinstance(repo, str) and bool(repo.strip())
        if has_instance == has_repo:
            raise ValueError(
                f"language overrides list item at index {idx} must specify exactly one "
                "target selector: instance_id/id or repo/repository"
            )
        language = _parse_override_language_value(
            item.get("language", item.get("languages")), selector=f"list_item_{idx}"
        )
        if has_instance:
            instance_overrides[instance_id.strip()] = language
            continue
        normalized_repo = repo.strip()
        if "/" not in normalized_repo:
            raise ValueError(
                f"language overrides list item at index {idx} has invalid repo "
                f"(expected owner/name): {normalized_repo!r}"
            )
        repo_overrides[normalized_repo] = language

    return LanguageOverrides(
        instance_overrides=instance_overrides,
        repo_overrides=repo_overrides,
    )


def _parse_language_overrides(raw: object) -> LanguageOverrides:
    if raw is None:
        return LanguageOverrides(instance_overrides={}, repo_overrides={})
    if isinstance(raw, list):
        return _parse_rule_list_overrides(raw)

    if isinstance(raw, dict):
        if not raw:
            return LanguageOverrides(instance_overrides={}, repo_overrides={})

        instance_container_keys = (
            "language_overrides",
            "instance_id_overrides",
            "instance_overrides",
            "overrides",
        )
        repo_container_keys = ("repo_overrides", "repository_overrides")
        has_container = any(
            key in raw
            for key in (*instance_container_keys, *repo_container_keys, "rules")
        )
        if has_container:
            allowed_keys = {
                *instance_container_keys,
                *repo_container_keys,
                "rules",
            }
            unknown_keys = sorted(
                key for key in raw.keys() if isinstance(key, str) and key not in allowed_keys
            )
            if unknown_keys:
                raise ValueError(
                    "unknown top-level keys in language overrides file: "
                    f"{', '.join(unknown_keys)}"
                )

            instance_overrides: dict[str, str] = {}
            repo_overrides: dict[str, str] = {}
            for key in instance_container_keys:
                if key in raw:
                    instance_overrides.update(
                        _parse_named_overrides(
                            raw[key],
                            selector_name="instance_id",
                            enforce_repo_format=False,
                        )
                    )
            for key in repo_container_keys:
                if key in raw:
                    repo_overrides.update(
                        _parse_named_overrides(
                            raw[key],
                            selector_name="repo",
                            enforce_repo_format=True,
                        )
                    )
            if "rules" in raw:
                from_rules = _parse_rule_list_overrides(raw["rules"])
                instance_overrides.update(from_rules.instance_overrides)
                repo_overrides.update(from_rules.repo_overrides)
            return LanguageOverrides(
                instance_overrides=instance_overrides,
                repo_overrides=repo_overrides,
            )

        # Backward-compatible shape: object mapping instance_id -> language.
        instance_overrides = _parse_named_overrides(
            raw, selector_name="instance_id", enforce_repo_format=False
        )
        return LanguageOverrides(
            instance_overrides=instance_overrides,
            repo_overrides={},
        )

    raise ValueError("language overrides file must be a JSON object or list")


def _load_language_overrides(path: Path) -> LanguageOverrides:
    if not path.exists():
        raise FileNotFoundError(f"language overrides file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return LanguageOverrides(instance_overrides={}, repo_overrides={})
    raw = json.loads(raw_text)
    return _parse_language_overrides(raw)


def _apply_language_overrides(
    rows: list[dict], overrides: LanguageOverrides
) -> tuple[list[dict], int, list[str], list[str]]:
    patched_rows = [dict(row) for row in rows]
    if not overrides.instance_overrides and not overrides.repo_overrides:
        return patched_rows, 0, [], []

    instance_index: dict[str, list[int]] = {}
    repo_index: dict[str, list[int]] = {}
    for idx, row in enumerate(patched_rows):
        instance_id = row.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            instance_index.setdefault(instance_id, []).append(idx)
        repo = row.get("repo")
        if isinstance(repo, str) and repo:
            repo_index.setdefault(repo, []).append(idx)

    touched_rows: set[int] = set()
    missing_repos: list[str] = []
    for repo, language_value in overrides.repo_overrides.items():
        row_indexes = repo_index.get(repo)
        if not row_indexes:
            missing_repos.append(repo)
            continue
        for row_idx in row_indexes:
            patched_rows[row_idx]["language"] = language_value
            touched_rows.add(row_idx)

    missing_instance_ids: list[str] = []
    for instance_id, language_value in overrides.instance_overrides.items():
        row_indexes = instance_index.get(instance_id)
        if not row_indexes:
            missing_instance_ids.append(instance_id)
            continue
        for row_idx in row_indexes:
            patched_rows[row_idx]["language"] = language_value
            touched_rows.add(row_idx)
    return patched_rows, len(touched_rows), missing_instance_ids, missing_repos


def _swe_care_image_tag(task: BuildTask) -> str:
    if task.pull_number is None:
        raise ValueError("pull_number is required for SWE-CARE image naming")
    org, repo_name = task.repo.split("/", 1)
    return (
        f"reviewbench/{_normalize_tag_component(org)}__"
        f"{_normalize_tag_component(repo_name)}-{task.pull_number}:latest"
    )


def _image_exists(image_tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _precheck_rows(
    rows: list[dict],
    *,
    include_non_python: bool,
    use_merged_commit: bool,
) -> tuple[list[BuildTask], list[dict]]:
    skipped: list[dict] = []
    grouped: dict[tuple[str, int | None, str], list[str]] = {}
    grouped_languages: dict[tuple[str, int | None, str], set[str]] = {}
    short_commit_index: dict[tuple[str, str], str] = {}
    short_commit_collisions: set[tuple[str, str]] = set()

    for idx, row in enumerate(rows):
        instance_id = row.get("instance_id", f"row_{idx}")
        repo = row.get("repo")
        pull_number = row.get("pull_number")
        commit = _get_target_commit(row, use_merged_commit)
        language = str(row.get("language", ""))
        detected_languages = _infer_task_languages(language)

        if not isinstance(repo, str) or "/" not in repo:
            skipped.append(
                {
                    "instance_id": instance_id,
                    "status": "skipped_invalid_repo",
                    "reason": f"invalid repo: {repo!r}",
                }
            )
            continue
        if not isinstance(commit, str) or len(commit) < 7:
            skipped.append(
                {
                    "instance_id": instance_id,
                    "status": "skipped_invalid_commit",
                    "reason": f"invalid commit: {commit!r}",
                }
            )
            continue
        if not isinstance(pull_number, int) or pull_number <= 0:
            skipped.append(
                {
                    "instance_id": instance_id,
                    "status": "skipped_invalid_pull_number",
                    "reason": f"invalid pull_number: {pull_number!r}",
                }
            )
            continue
        if (not include_non_python) and "python" not in detected_languages:
            skipped.append(
                {
                    "instance_id": instance_id,
                    "status": "skipped_non_python",
                    "reason": f"language={language!r}",
                }
            )
            continue

        key = (repo, pull_number, commit)
        grouped.setdefault(key, []).append(instance_id)
        effective_languages = detected_languages or ("python",)
        grouped_languages.setdefault(key, set()).update(effective_languages)

        short_key = (repo, commit[:7])
        previous = short_commit_index.get(short_key)
        if previous is None:
            short_commit_index[short_key] = commit
        elif previous != commit:
            short_commit_collisions.add(short_key)

    tasks: list[BuildTask] = []
    for (repo, pull_number, commit), instance_ids in grouped.items():
        short_key = (repo, commit[:7])
        if short_key in short_commit_collisions:
            for instance_id in instance_ids:
                skipped.append(
                    {
                        "instance_id": instance_id,
                        "status": "skipped_short_commit_collision",
                        "reason": f"repo {repo} has multiple commits with short id {commit[:7]}",
                    }
                )
            continue

        language_pool = grouped_languages.get((repo, pull_number, commit), {"python"})
        task_languages = tuple(
            lang for lang in _BUILD_LANGUAGE_PRIORITY if lang in language_pool
        ) or ("python",)
        tasks.append(
            BuildTask(
                task_id=_task_id_for(repo, commit),
                repo=repo,
                pull_number=pull_number,
                commit=commit,
                languages=task_languages,
                instance_ids=tuple(instance_ids),
            )
        )
    return tasks, skipped


def _resolve_python_version_for_task(task: BuildTask, args: argparse.Namespace) -> tuple[str, str]:
    if args.python_version.lower() != "auto":
        return args.python_version, "cli_override"
    if "python" not in task.languages:
        return DEFAULT_PYTHON_VERSION, "default_non_python_language"
    manual_overrides = resolve_build_script_overrides(
        task.repo,
        task.commit,
        overrides_root=args.build_overrides_root,
    )
    if manual_overrides.python_version:
        logger.info(
            "Using manual python override for %s@%s: %s (%s)",
            task.repo,
            task.commit[:12],
            manual_overrides.python_version,
            manual_overrides.source_dir,
        )
        return manual_overrides.python_version, "manual_override_python_version"

    # `resolve_python_version` mutates local git checkout state, so serialize by repo.
    lock = _get_repo_lock(task.repo)
    with lock:
        try:
            resolution = resolve_python_version(
                task.repo,
                task.commit,
                pull_number=task.pull_number,
                cache_dir=args.repos_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Python version auto-resolution failed for %s@%s; using default %s. Error: %s",
                task.repo,
                task.commit,
                DEFAULT_PYTHON_VERSION,
                exc,
            )
            return DEFAULT_PYTHON_VERSION, "default_fallback_resolution_error"
    return resolution.python_version, resolution.source


def _run_basic_checks(build: BuildResult, task: BuildTask, image_tag: str, log_file: Path) -> dict:
    setup_script = build.context_dir / "setup_repo.sh"
    log_text = log_file.read_text(errors="ignore") if log_file.exists() else ""
    log_signals = analyze_build_log_text(log_text)
    checks = {
        "context_has_dockerfile": build.dockerfile_path.exists(),
        "context_has_setup_script": setup_script.exists(),
        "context_has_install_script": (build.context_dir / "install_deps.sh").exists(),
        "build_log_exists": log_file.exists(),
        "image_exists": _image_exists(image_tag),
        "setup_mentions_commit": setup_script.exists() and task.commit in setup_script.read_text(),
        "build_log_has_soft_errors": log_signals.has_soft_errors,
        "build_log_retryable_soft_error": log_signals.retryable_with_python_fallback,
        "build_log_soft_error_kinds": list(log_signals.kinds),
        "build_log_soft_error_evidence": list(log_signals.evidence),
    }
    checks["hard_checks_passed"] = all(
        (
            checks["context_has_dockerfile"],
            checks["context_has_setup_script"],
            checks["context_has_install_script"],
            checks["build_log_exists"],
            checks["image_exists"],
            checks["setup_mentions_commit"],
        )
    )
    checks["all_passed"] = checks["hard_checks_passed"] and (
        not checks["build_log_has_soft_errors"]
    )
    return checks


def _python_retry_candidates(initial: str, *, allow_fallback: bool) -> tuple[str, ...]:
    normalized = (initial or DEFAULT_PYTHON_VERSION).strip()
    if not allow_fallback or normalized not in SUPPORTED_PYTHON_VERSIONS:
        return (normalized,)

    # Try older versions first, then newer versions as a fallback.
    start_idx = SUPPORTED_PYTHON_VERSIONS.index(normalized)
    older = list(SUPPORTED_PYTHON_VERSIONS[start_idx + 1 :])
    newer = list(reversed(SUPPORTED_PYTHON_VERSIONS[:start_idx]))
    return tuple([normalized, *older, *newer])


def _build_one(task: BuildTask, args: argparse.Namespace, run_dir: Path) -> dict:
    started = time.time()
    build_timeout_sec: int | None = (
        args.build_timeout_sec if args.build_timeout_sec and args.build_timeout_sec > 0 else None
    )
    build_logs_dir = run_dir / "build_logs"
    build_logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = build_logs_dir / f"{task.task_id}.log"
    log_file.write_text(
        (
            f"task_id={task.task_id}\n"
            f"repo={task.repo}\n"
            f"pull_number={task.pull_number}\n"
            f"commit={task.commit}\n\n"
        )
    )

    result = {
        "task_id": task.task_id,
        "repo": task.repo,
        "pull_number": task.pull_number,
        "commit": task.commit,
        "languages": list(task.languages),
        "instance_ids": list(task.instance_ids),
        "status": "pending",
        "log_file": str(log_file),
        "build_timeout_sec": build_timeout_sec,
    }
    attempt_errors: list[dict] = []

    try:
        python_version, version_source = _resolve_python_version_for_task(task, args)
        image_tag = _swe_care_image_tag(task)
        allow_py_fallback = (
            args.python_version.lower() == "auto"
            and "python" in task.languages
            and not version_source.startswith("manual_override")
        )
        py_candidates = _python_retry_candidates(
            python_version or DEFAULT_PYTHON_VERSION,
            allow_fallback=allow_py_fallback,
        )
        result["python_version_source"] = version_source
        result["python_version_candidates"] = list(py_candidates)
        result["image_tag"] = image_tag

        if _image_exists(image_tag) and not args.force_rebuild:
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write("Skipping build because image already exists.\n")
            result["status"] = "skipped_existing_image"
            result["elapsed_sec"] = round(time.time() - started, 2)
            return result

        for attempt_idx, attempt_py in enumerate(py_candidates, start=1):
            spec = PREnvironmentSpec(
                repo=task.repo,
                pull_number=task.pull_number,
                commit=task.commit,
                languages=task.languages,
                python_version=attempt_py,
                image_tag=image_tag,
            )
            result["python_version"] = spec.python_version
            result["context_dir"] = str(args.context_root / spec.docker_context_dirname)

            if attempt_idx > 1:
                with log_file.open("a", encoding="utf-8") as fh:
                    fh.write(
                        (
                            f"\nRetrying build with python version "
                            f"({attempt_py}); attempt {attempt_idx}/{len(py_candidates)}\n"
                        )
                    )

            try:
                build = build_image(
                    spec,
                    context_root=args.context_root,
                    cached_repos_dir=args.repos_dir,
                    build_overrides_root=args.build_overrides_root,
                    no_cache=args.no_cache,
                    pull_base=args.pull_base,
                    build_log_path=log_file,
                    build_timeout_sec=build_timeout_sec,
                    apt_proxy=args.apt_proxy,
                    pip_index_url=args.pip_index_url,
                    pip_extra_index_url=args.pip_extra_index_url,
                    pip_trusted_host=args.pip_trusted_host,
                )
                checks = _run_basic_checks(build, task, image_tag, log_file)
                result["checks"] = checks
                if checks["build_log_has_soft_errors"]:
                    soft_error = (
                        "soft build error(s): "
                        + ", ".join(checks["build_log_soft_error_kinds"])
                    )
                    attempt_errors.append(
                        {
                            "python_version": attempt_py,
                            "error": soft_error,
                            "evidence": checks["build_log_soft_error_evidence"],
                        }
                    )
                    should_retry = (
                        allow_py_fallback
                        and attempt_idx < len(py_candidates)
                        and bool(checks["build_log_retryable_soft_error"])
                    )
                    if should_retry:
                        with log_file.open("a", encoding="utf-8") as fh:
                            fh.write(
                                (
                                    "\nDetected soft build errors. "
                                    f"Retrying with next python candidate.\n"
                                )
                            )
                        continue
                    result["status"] = "soft_error"
                else:
                    result["status"] = "ok" if checks["all_passed"] else "check_failed"
                result["elapsed_sec"] = round(time.time() - started, 2)
                if attempt_errors:
                    result["attempt_errors"] = attempt_errors
                if attempt_idx > 1:
                    result["python_version_source"] = f"{version_source}+fallback_retry"
                return result
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                attempt_errors.append({"python_version": attempt_py, "error": error_text})
                log_text = log_file.read_text(errors="ignore") if log_file.exists() else ""
                should_retry = (
                    allow_py_fallback
                    and attempt_idx < len(py_candidates)
                    and (
                        has_python_retry_signal(error_text)
                        or has_python_retry_signal(log_text)
                    )
                )
                if not should_retry:
                    raise

        # Should be unreachable because non-retriable failures are raised immediately.
        raise RuntimeError("all python-version attempts failed")
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = str(exc)
        if attempt_errors:
            result["attempt_errors"] = attempt_errors
        result["elapsed_sec"] = round(time.time() - started, 2)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"build failed:\n{exc}\n")
        return result


def main() -> int:
    args = parse_args()
    split_expr = _normalize_split_expr(args.split, args.limit)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.split}_{timestamp}"
    run_log = _configure_logging(run_dir)
    instance_filters, invalid_instance_filters = _parse_instance_filters(args.instances)
    if invalid_instance_filters:
        raise ValueError(
            "invalid --instance selector(s); expected one of: "
            "org__repo__hash, reviewbench/org__repo-pr@hash, Org__Repo-pr@hash, "
            "or a file path containing those selectors. Example: "
            "pandas-dev__pandas__22a6bff. Invalid values: "
            f"{', '.join(invalid_instance_filters[:5])}"
        )

    logger.info("Loading %s (%s)", args.dataset, split_expr)
    ds = load_dataset(args.dataset, split=split_expr, cache_dir=str(args.cache_dir))
    rows = list(ds)
    language_override_count = 0
    language_override_instance_count = 0
    language_override_repo_count = 0
    language_overrides_applied_rows = 0
    language_override_missing_instance_ids: list[str] = []
    language_override_missing_repos: list[str] = []
    if args.language_overrides_file is not None:
        overrides = _load_language_overrides(args.language_overrides_file)
        language_override_instance_count = len(overrides.instance_overrides)
        language_override_repo_count = len(overrides.repo_overrides)
        language_override_count = overrides.count
        (
            rows,
            language_overrides_applied_rows,
            language_override_missing_instance_ids,
            language_override_missing_repos,
        ) = _apply_language_overrides(rows, overrides)
        logger.info(
            "Applied language overrides (%s): %d instance entries, %d repo entries, %d row patches",
            args.language_overrides_file,
            language_override_instance_count,
            language_override_repo_count,
            language_overrides_applied_rows,
        )
        if language_override_missing_instance_ids:
            logger.warning(
                "Language overrides referenced %d missing instance_id(s). Example: %s",
                len(language_override_missing_instance_ids),
                ", ".join(language_override_missing_instance_ids[:5]),
            )
        if language_override_missing_repos:
            logger.warning(
                "Language overrides referenced %d missing repo key(s). Example: %s",
                len(language_override_missing_repos),
                ", ".join(language_override_missing_repos[:5]),
            )
    total_rows_loaded = len(rows)
    logger.info("Loaded %d rows", total_rows_loaded)

    dataset_ignores_file = DEFAULT_SWE_CARE_IGNORES_FILE
    dataset_ignores = _load_dataset_ignores(dataset_ignores_file)
    dataset_ignore_instance_count = len(dataset_ignores.instance_ids)
    dataset_ignore_repo_count = len(dataset_ignores.repos)
    dataset_ignores_applied_rows = 0
    dataset_ignore_missing_instance_ids: list[str] = []
    dataset_ignore_missing_repos: list[str] = []
    if dataset_ignores_file.exists():
        (
            rows,
            dataset_ignores_applied_rows,
            dataset_ignore_missing_instance_ids,
            dataset_ignore_missing_repos,
        ) = _apply_dataset_ignores(rows, dataset_ignores)
        logger.info(
            "Applied SWE-CARE ignores (%s): %d instance selectors, %d repo selectors, "
            "%d excluded row(s), %d remaining row(s)",
            dataset_ignores_file,
            dataset_ignore_instance_count,
            dataset_ignore_repo_count,
            dataset_ignores_applied_rows,
            len(rows),
        )
        if dataset_ignore_missing_instance_ids:
            logger.warning(
                "SWE-CARE ignores referenced %d missing instance selector(s). Example: %s",
                len(dataset_ignore_missing_instance_ids),
                ", ".join(dataset_ignore_missing_instance_ids[:5]),
            )
        if dataset_ignore_missing_repos:
            logger.warning(
                "SWE-CARE ignores referenced %d missing repo selector(s). Example: %s",
                len(dataset_ignore_missing_repos),
                ", ".join(dataset_ignore_missing_repos[:5]),
            )
    else:
        logger.info("SWE-CARE ignores file not found, skipping excludes: %s", dataset_ignores_file)

    repo_filter = (args.repo or "").strip() or None
    if repo_filter is not None:
        if "/" not in repo_filter:
            raise ValueError("--repo must be in owner/name format")
        rows = [row for row in rows if row.get("repo") == repo_filter]
        logger.info(
            "Applied repo filter %s: selected %d/%d rows",
            repo_filter,
            len(rows),
            total_rows_loaded,
        )

    instance_filter_missing: list[str] = []
    if instance_filters:
        rows_before_instance_filter = len(rows)
        rows, instance_filter_missing = _filter_rows_by_instance_selectors(
            rows,
            instance_filters,
            use_merged_commit=args.use_merged_commit,
        )
        logger.info(
            "Applied instance filter (%d selectors): selected %d/%d rows",
            len(instance_filters),
            len(rows),
            rows_before_instance_filter,
        )
        if instance_filter_missing:
            logger.warning(
                "Instance filter referenced %d selector(s) with no dataset match. Example: %s",
                len(instance_filter_missing),
                ", ".join(instance_filter_missing[:5]),
            )

    tasks, skipped_rows = _precheck_rows(
        rows,
        include_non_python=args.include_non_python,
        use_merged_commit=args.use_merged_commit,
    )
    logger.info(
        "Prepared %d unique build tasks (skipped rows during precheck: %d)",
        len(tasks),
        len(skipped_rows),
    )

    task_results: list[dict] = []
    if tasks:
        max_workers = max(1, args.max_workers)
        logger.info("Starting builds with max_workers=%d", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_build_one, task, args, run_dir): task for task in tasks
            }
            for idx, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                res = future.result()
                task_results.append(res)
                logger.info(
                    "[%d/%d] %s %s",
                    idx,
                    len(tasks),
                    res["status"],
                    task.task_id,
                )

    results = {
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "build_overrides_root": str(args.build_overrides_root),
        "language_overrides_file": (
            str(args.language_overrides_file) if args.language_overrides_file else None
        ),
        "language_override_count": language_override_count,
        "language_override_instance_count": language_override_instance_count,
        "language_override_repo_count": language_override_repo_count,
        "language_overrides_applied_rows": language_overrides_applied_rows,
        "language_override_missing_instance_ids": language_override_missing_instance_ids,
        "language_override_missing_repos": language_override_missing_repos,
        "dataset_ignores_file": str(dataset_ignores_file),
        "dataset_ignore_instance_count": dataset_ignore_instance_count,
        "dataset_ignore_repo_count": dataset_ignore_repo_count,
        "dataset_ignores_applied_rows": dataset_ignores_applied_rows,
        "dataset_ignore_missing_instance_ids": dataset_ignore_missing_instance_ids,
        "dataset_ignore_missing_repos": dataset_ignore_missing_repos,
        "repo_filter": repo_filter,
        "instance_filters": list(instance_filters),
        "instance_filter_missing": instance_filter_missing,
        "build_timeout_sec": args.build_timeout_sec,
        "run_dir": str(run_dir),
        "run_log": str(run_log),
        "task_results": task_results,
        "skipped_rows": skipped_rows,
    }
    results_file = run_dir / "results.json"
    results_file.write_text(json.dumps(results, indent=2))

    status_counts: dict[str, int] = {}
    for row in task_results:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    for row in skipped_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    summary = {
        "total_rows_loaded": total_rows_loaded,
        "total_rows_selected": len(rows),
        "build_overrides_root": str(args.build_overrides_root),
        "language_overrides_file": (
            str(args.language_overrides_file) if args.language_overrides_file else None
        ),
        "language_override_count": language_override_count,
        "language_override_instance_count": language_override_instance_count,
        "language_override_repo_count": language_override_repo_count,
        "language_overrides_applied_rows": language_overrides_applied_rows,
        "language_override_missing_instance_ids": language_override_missing_instance_ids,
        "language_override_missing_repos": language_override_missing_repos,
        "dataset_ignores_file": str(dataset_ignores_file),
        "dataset_ignore_instance_count": dataset_ignore_instance_count,
        "dataset_ignore_repo_count": dataset_ignore_repo_count,
        "dataset_ignores_applied_rows": dataset_ignores_applied_rows,
        "dataset_ignore_missing_instance_ids": dataset_ignore_missing_instance_ids,
        "dataset_ignore_missing_repos": dataset_ignore_missing_repos,
        "repo_filter": repo_filter,
        "instance_filters": list(instance_filters),
        "instance_filter_missing": instance_filter_missing,
        "unique_tasks": len(tasks),
        "status_counts": status_counts,
        "results_file": str(results_file),
        "run_log": str(run_log),
    }
    summary_file = run_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    logger.info("Run complete. Summary: %s", summary_file)
    failure_statuses = ("error", "check_failed", "soft_error")
    failed_count = sum(status_counts.get(status, 0) for status in failure_statuses)
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
