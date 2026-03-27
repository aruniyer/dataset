"""Heuristics for interpreting docker build logs."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PYTHON_RETRY_PATTERNS = (
    re.compile(r"requires python .+ but yours is", re.IGNORECASE),
    re.compile(r"No matching distribution found for", re.IGNORECASE),
    re.compile(r"Could not find a version that satisfies the requirement", re.IGNORECASE),
    re.compile(r"requires a different python", re.IGNORECASE),
    re.compile(r"requires at least Python \d+\.\d+", re.IGNORECASE),
    re.compile(r"Python version >= \d+\.\d+ required", re.IGNORECASE),
    re.compile(r"Could not build wheels for", re.IGNORECASE),
    re.compile(r"Failed building wheel for", re.IGNORECASE),
    re.compile(r"Cannot install on Python version", re.IGNORECASE),
    re.compile(r"module 'collections' has no attribute 'Iterable'", re.IGNORECASE),
    re.compile(r"cannot import name 'Mapping' from 'collections'", re.IGNORECASE),
    re.compile(r"pkgutil.*ImpImporter", re.IGNORECASE),
    re.compile(r"has no attribute 'ImpImporter'", re.IGNORECASE),
    re.compile(r"setuptools\.extern\.six", re.IGNORECASE),
    re.compile(r"metadata-generation-failed", re.IGNORECASE),
    re.compile(r"global flags not at the start of the expression", re.IGNORECASE),
)

# These are "soft failures": docker build exits 0, but dependency setup is incomplete.
_SOFT_ERROR_RULES = (
    (
        "repo_package_install_failed",
        re.compile(r"Warning: repository package install failed; continuing", re.IGNORECASE),
    ),
    (
        "repo_package_install_skipped_build_compat",
        re.compile(
            r"Info: skipping repository package install after build-time compatibility failure\.",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class BuildLogSignals:
    """Derived signals from one docker build log."""

    has_soft_errors: bool
    retryable_with_python_fallback: bool
    kinds: tuple[str, ...]
    evidence: tuple[str, ...]


def has_python_retry_signal(text: str) -> bool:
    """Return True when text suggests trying another Python version."""
    return any(pattern.search(text) for pattern in _PYTHON_RETRY_PATTERNS)


def analyze_build_log_text(text: str, *, evidence_limit: int = 8) -> BuildLogSignals:
    """Detect soft errors in a docker build log."""
    if not text:
        return BuildLogSignals(
            has_soft_errors=False,
            retryable_with_python_fallback=False,
            kinds=(),
            evidence=(),
        )

    kinds: list[str] = []
    evidence: list[str] = []
    seen_kinds: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for kind, pattern in _SOFT_ERROR_RULES:
            if not pattern.search(line):
                continue
            if kind not in seen_kinds:
                seen_kinds.add(kind)
                kinds.append(kind)
            if len(evidence) < evidence_limit:
                evidence.append(line[:500])

    has_soft_errors = bool(kinds)
    retryable = has_soft_errors and has_python_retry_signal(text)

    return BuildLogSignals(
        has_soft_errors=has_soft_errors,
        retryable_with_python_fallback=retryable,
        kinds=tuple(kinds),
        evidence=tuple(evidence),
    )
