"""Data model for PR-specific Docker execution environments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

KNOWN_LANGUAGES = (
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "csharp",
)
IMPLEMENTED_LANGUAGES = ("python", "javascript", "typescript")
_LANGUAGE_SLUGS = {
    "python": "py",
    "javascript": "js",
    "typescript": "ts",
    "java": "java",
    "go": "go",
    "rust": "rust",
    "csharp": "cs",
}

COMMON_APT_PACKAGES = (
    "git",
    "curl",
    "ca-certificates",
    "build-essential",
    "pkg-config",
)
PYTHON_APT_PACKAGES = (
    "libffi-dev",
    "libssl-dev",
    "python3-dev",
)
BASE_APT_PACKAGES = COMMON_APT_PACKAGES + PYTHON_APT_PACKAGES
DEFAULT_APT_PACKAGES = BASE_APT_PACKAGES

_REPO_APT_GROUP_PACKAGES: dict[str, tuple[str, ...]] = {
    "postgres": ("libpq-dev",),
    "numeric": ("gfortran", "libopenblas-dev", "liblapack-dev"),
    "plot": ("libfreetype6-dev", "libpng-dev", "libcairo2-dev", "libqhull-dev"),
    "av": (
        "ffmpeg",
        "libavformat-dev",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavutil-dev",
        "libavfilter-dev",
        "libsdl-pango-dev",
        "libsrtp2-dev",
        "libswscale-dev",
        "libswresample-dev",
    ),
    "xmlsec": ("xmlsec1", "libxml2-dev", "libxmlsec1-dev", "libxmlsec1-openssl"),
}

_REPO_APT_GROUP_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"(airflow|dbt|mlflow|netbox|posthog|rasa|saleor|sentry|superset)",
            re.IGNORECASE,
        ),
        ("postgres",),
    ),
    (
        re.compile(
            r"(dask|dgl|gensim|jax|lightning|modin|numba|numpy|pandas|ray|scikit-learn|scipy|torch|pytorch|xorbits)",
            re.IGNORECASE,
        ),
        ("numeric",),
    ),
    (
        re.compile(
            r"(matplotlib|manim|altair|plot|seaborn)",
            re.IGNORECASE,
        ),
        ("plot",),
    ),
    (
        re.compile(
            r"(manim|fiftyone|video|audio|ffmpeg|opencv|openmined|pysyft)",
            re.IGNORECASE,
        ),
        ("av",),
    ),
    (
        re.compile(
            r"(home-assistant|saleor|sentry|scrapy|xmlsec)",
            re.IGNORECASE,
        ),
        ("xmlsec",),
    ),
)

DEFAULT_EXTRA_PIP_PACKAGES = ("pytest",)


def _normalize_slug(value: str) -> str:
    value = value.strip().lower()
    return re.sub(r"[^a-z0-9_.-]+", "-", value)


def _normalize_languages(languages: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen = set()
    for raw in languages:
        lang = raw.strip().lower()
        if not lang:
            continue
        if lang not in seen:
            seen.add(lang)
            ordered.append(lang)
    return tuple(ordered)


def _dedupe_ordered(items: tuple[str, ...]) -> tuple[str, ...]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def infer_apt_packages_for_repo(
    repo: str,
    *,
    languages: tuple[str, ...] = ("python",),
) -> tuple[str, ...]:
    """Infer apt dependencies from repository identity.

    This keeps base packages small and adds repo-specific groups when likely needed.
    """
    normalized_languages = _normalize_languages(languages)
    packages: list[str] = list(COMMON_APT_PACKAGES)
    if any(lang in normalized_languages for lang in ("javascript", "typescript")):
        packages.append("nodejs")
        packages.append("npm")
    if "python" in normalized_languages:
        for pkg in PYTHON_APT_PACKAGES:
            if pkg not in packages:
                packages.append(pkg)
    if "python" not in normalized_languages:
        return tuple(packages)

    for pattern, group_names in _REPO_APT_GROUP_RULES:
        if not pattern.search(repo):
            continue
        for group in group_names:
            for pkg in _REPO_APT_GROUP_PACKAGES[group]:
                if pkg not in packages:
                    packages.append(pkg)
    return tuple(packages)


@dataclass(frozen=True)
class PREnvironmentSpec:
    """Configuration for building one PR-specific Docker environment.

    The model already supports multi-language metadata. Build/render logic is
    currently implemented for Python and basic JS/TS environments.
    """

    repo: str
    pull_number: int | None = None
    commit: str | None = None
    languages: tuple[str, ...] = ("python",)
    python_version: str = "3.11"
    image_tag: str | None = None
    apt_packages: tuple[str, ...] = field(default_factory=tuple)
    extra_pip_packages: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_EXTRA_PIP_PACKAGES
    )
    workdir: str = "/workspace"

    def __post_init__(self) -> None:
        if not _REPO_PATTERN.match(self.repo):
            raise ValueError(
                "repo must be in 'owner/name' format, for example 'pallets/flask'"
            )
        if self.pull_number is not None and self.pull_number <= 0:
            raise ValueError("pull_number must be positive when provided")
        if self.pull_number is None and not self.commit:
            raise ValueError("Provide at least one of: pull_number, commit")
        if not self.workdir.startswith("/"):
            raise ValueError("workdir must be an absolute path inside the container")
        normalized_languages = _normalize_languages(self.languages)
        if not normalized_languages:
            raise ValueError("At least one language must be provided")
        unknown = [lang for lang in normalized_languages if lang not in KNOWN_LANGUAGES]
        if unknown:
            raise ValueError(
                f"Unknown language(s): {unknown}. Known languages: {KNOWN_LANGUAGES}"
            )
        object.__setattr__(self, "languages", normalized_languages)
        if not self.apt_packages:
            object.__setattr__(
                self,
                "apt_packages",
                infer_apt_packages_for_repo(self.repo, languages=normalized_languages),
            )
        else:
            object.__setattr__(self, "apt_packages", _dedupe_ordered(self.apt_packages))
        if "python" in normalized_languages and not self.python_version.strip():
            raise ValueError("python_version cannot be empty when Python is enabled")
        if not self.apt_packages:
            raise ValueError("apt_packages must contain at least one package")
        if any(not item.strip() for item in self.apt_packages):
            raise ValueError("apt_packages cannot contain empty items")
        if any(not item.strip() for item in self.extra_pip_packages):
            raise ValueError("extra_pip_packages cannot contain empty items")

    @property
    def repo_slug(self) -> str:
        """Filesystem-safe repo slug."""
        return _normalize_slug(self.repo.replace("/", "__"))

    @property
    def org_slug(self) -> str:
        """Filesystem-safe organization/user slug."""
        org, _ = self.repo.split("/", 1)
        return _normalize_slug(org)

    @property
    def repo_name_slug(self) -> str:
        """Filesystem-safe repository-name slug."""
        _, repo_name = self.repo.split("/", 1)
        return _normalize_slug(repo_name)

    @property
    def checked_out_commit_id(self) -> str:
        """Commit identifier expected to be checked out in the container.

        Falls back to `pr<n>` when a concrete commit hash is not provided.
        """
        if self.commit:
            commit = _normalize_slug(self.commit)
            return commit[:7]
        if self.pull_number is not None:
            return f"pr{self.pull_number}"
        return self.ref_slug

    @property
    def docker_context_dirname(self) -> str:
        """Directory name for generated Docker build context."""
        return f"{self.org_slug}__{self.repo_name_slug}__{self.checked_out_commit_id}"

    @property
    def ref_slug(self) -> str:
        """Short ref identifier for image names and context folders."""
        if self.pull_number is not None:
            return f"pr{self.pull_number}"
        assert self.commit is not None
        return f"commit-{self.commit[:12]}"

    @property
    def language_slug(self) -> str:
        """Short combined language identifier (for tags and context paths)."""
        return "-".join(_LANGUAGE_SLUGS.get(lang, lang) for lang in self.languages)

    @property
    def primary_language(self) -> str:
        """Primary language used to select the build implementation."""
        return self.languages[0]

    @property
    def unimplemented_languages(self) -> tuple[str, ...]:
        """Languages known by spec but not yet implemented in builders."""
        return tuple(lang for lang in self.languages if lang not in IMPLEMENTED_LANGUAGES)

    @property
    def resolved_image_tag(self) -> str:
        """Resolved docker tag to build."""
        if self.image_tag:
            return self.image_tag
        if self.languages == ("python",):
            py = _normalize_slug(self.python_version.replace(".", ""))
            return f"reviewbench/{self.repo_slug}:py{py}-{self.ref_slug}"
        parts = [self.language_slug]
        if "python" in self.languages:
            py = _normalize_slug(self.python_version.replace(".", ""))
            parts.append(f"py{py}")
        parts.append(self.ref_slug)
        return f"reviewbench/{self.repo_slug}:{'-'.join(parts)}"


# Backward-compatible alias with a broader name for future non-PR use cases.
RepoEnvironmentSpec = PREnvironmentSpec
