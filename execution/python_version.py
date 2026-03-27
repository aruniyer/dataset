"""Resolve Python versions for repos at specific commits."""

from __future__ import annotations

import configparser
import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from pipeline import repo_manager

logger = logging.getLogger(__name__)

DEFAULT_PYTHON_VERSION = "3.11"
DEFAULT_REPO_CACHE_DIR = Path(__file__).resolve().parent.parent / "repos"
SUPPORTED_PYTHON_VERSIONS = ("3.13", "3.12", "3.11", "3.10", "3.9", "3.8", "3.7", "3.6")
PREFERRED_PYTHON_VERSIONS = ("3.11", "3.10", "3.9", "3.8", "3.7", "3.6")

_PYTHON_VERSION_RE = re.compile(r"\b(3\.(?:6|7|8|9|10|11|12|13))(?:\.\d+)?\b")
_TOX_ENV_RE = re.compile(r"\bpy(3(?:6|7|8|9|10|11|12|13))\b")
_TOX_BASEPY_RE = re.compile(r"\bpython(?:-?)(3\.(?:6|7|8|9|10|11|12|13))(?:\.\d+)?\b")
_CONDA_ENV_FILENAMES = ("environment.yml", "environment.yaml", "conda.yml", "conda.yaml")


@dataclass(frozen=True)
class PythonVersionResolution:
    """Resolved Python version with provenance."""

    python_version: str
    source: str


def resolve_python_version(
    repo: str,
    commit: str,
    *,
    pull_number: int | None = None,
    cache_dir: str | Path = DEFAULT_REPO_CACHE_DIR,
    default: str = DEFAULT_PYTHON_VERSION,
) -> PythonVersionResolution:
    """Resolve Python version for `repo` at a specific `commit`."""
    repo_path = repo_manager.clone_repo(repo, cache_dir=cache_dir)
    if pull_number is not None:
        repo_manager.fetch_pr_commits(repo_path, pull_number)
    repo_manager.checkout_commit(repo_path, commit)
    return infer_python_version_from_repo(repo_path, default=default)


def infer_python_version_from_repo(
    repo_path: str | Path,
    *,
    default: str = DEFAULT_PYTHON_VERSION,
) -> PythonVersionResolution:
    """Infer Python version from repository metadata files."""
    repo_path = Path(repo_path)

    direct = _resolve_from_direct_version_files(repo_path)
    if direct:
        return direct

    pyproject = _resolve_from_pyproject(repo_path)
    if pyproject:
        return pyproject

    setup_cfg = _resolve_from_setup_cfg(repo_path)
    if setup_cfg:
        return setup_cfg

    setup_py = _resolve_from_setup_py(repo_path)
    if setup_py:
        return setup_py

    pipfile = _resolve_from_pipfile(repo_path)
    if pipfile:
        return pipfile

    conda_env = _resolve_from_conda_environment(repo_path)
    if conda_env:
        return conda_env

    tox = _resolve_from_tox(repo_path)
    if tox:
        return tox

    return PythonVersionResolution(default, "default")


def _resolve_from_direct_version_files(repo_path: Path) -> PythonVersionResolution | None:
    python_version_file = repo_path / ".python-version"
    if python_version_file.exists():
        version = _pick_version_from_text(python_version_file.read_text())
        if version:
            return PythonVersionResolution(version, ".python-version")

    tool_versions = repo_path / ".tool-versions"
    if tool_versions.exists():
        for line in tool_versions.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("python "):
                version = _pick_version_from_text(stripped)
                if version:
                    return PythonVersionResolution(version, ".tool-versions")

    runtime_txt = repo_path / "runtime.txt"
    if runtime_txt.exists():
        version = _pick_version_from_text(runtime_txt.read_text())
        if version:
            return PythonVersionResolution(version, "runtime.txt")

    return None


def _resolve_from_pyproject(repo_path: Path) -> PythonVersionResolution | None:
    pyproject = repo_path / "pyproject.toml"
    if not pyproject.exists():
        return None

    try:
        data = tomllib.loads(pyproject.read_text())
    except (tomllib.TOMLDecodeError, OSError) as exc:
        logger.debug("Could not parse pyproject.toml: %s", exc)
        return None

    candidates: list[tuple[str, str]] = []

    project_requires = data.get("project", {}).get("requires-python")
    if isinstance(project_requires, str):
        candidates.append((project_requires, "pyproject.toml:project.requires-python"))

    poetry_python = (
        data.get("tool", {})
        .get("poetry", {})
        .get("dependencies", {})
        .get("python")
    )
    if isinstance(poetry_python, str):
        candidates.append((poetry_python, "pyproject.toml:tool.poetry.dependencies.python"))

    pdm_python = data.get("tool", {}).get("pdm", {}).get("python_requires")
    if isinstance(pdm_python, str):
        candidates.append((pdm_python, "pyproject.toml:tool.pdm.python_requires"))

    hatch_requires = data.get("tool", {}).get("hatch", {}).get("metadata", {}).get(
        "requires-python"
    )
    if isinstance(hatch_requires, str):
        candidates.append((hatch_requires, "pyproject.toml:tool.hatch.metadata.requires-python"))

    for raw_spec, source in candidates:
        resolved = _pick_version_from_spec(raw_spec)
        if resolved:
            return PythonVersionResolution(resolved, source)

    return None


def _resolve_from_setup_cfg(repo_path: Path) -> PythonVersionResolution | None:
    setup_cfg = repo_path / "setup.cfg"
    if not setup_cfg.exists():
        return None

    parser = configparser.ConfigParser()
    try:
        parser.read(setup_cfg)
    except configparser.Error as exc:
        logger.debug("Could not parse setup.cfg: %s", exc)
        return None

    if parser.has_option("options", "python_requires"):
        raw_spec = parser.get("options", "python_requires")
        resolved = _pick_version_from_spec(raw_spec)
        if resolved:
            return PythonVersionResolution(resolved, "setup.cfg:options.python_requires")

    return None


def _resolve_from_setup_py(repo_path: Path) -> PythonVersionResolution | None:
    setup_py = repo_path / "setup.py"
    if not setup_py.exists():
        return None

    text = setup_py.read_text(errors="ignore")
    match = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not match:
        return None

    resolved = _pick_version_from_spec(match.group(1))
    if resolved:
        return PythonVersionResolution(resolved, "setup.py:python_requires")
    return None


def _resolve_from_pipfile(repo_path: Path) -> PythonVersionResolution | None:
    pipfile = repo_path / "Pipfile"
    if not pipfile.exists():
        return None

    try:
        data = tomllib.loads(pipfile.read_text())
    except (tomllib.TOMLDecodeError, OSError) as exc:
        logger.debug("Could not parse Pipfile: %s", exc)
        return None

    requires = data.get("requires", {})
    for key in ("python_version", "python_full_version"):
        value = requires.get(key)
        if isinstance(value, str):
            resolved = _pick_version_from_text(value)
            if resolved:
                return PythonVersionResolution(resolved, f"Pipfile:requires.{key}")

    return None


def _resolve_from_conda_environment(repo_path: Path) -> PythonVersionResolution | None:
    for filename in _CONDA_ENV_FILENAMES:
        env_file = repo_path / filename
        if not env_file.exists():
            continue

        text = env_file.read_text(errors="ignore")
        for raw_line in text.splitlines():
            spec = _extract_python_spec_from_conda_dep_line(raw_line)
            if spec is None:
                continue
            if not spec:
                # "python" without a version pin doesn't help inference.
                continue

            resolved = _pick_version_from_spec(spec)
            if resolved:
                return PythonVersionResolution(
                    resolved,
                    f"{filename}:dependencies.python",
                )
    return None


def _extract_python_spec_from_conda_dep_line(line: str) -> str | None:
    """Extract python dependency spec from one conda env dependency line."""
    stripped = line.split("#", 1)[0].strip()
    if not stripped.startswith("-"):
        return None

    dep = stripped[1:].strip()
    if not dep:
        return None

    if dep[0] in {'"', "'"} and dep[-1] == dep[0]:
        dep = dep[1:-1].strip()
        if not dep:
            return None

    lowered = dep.lower()
    if not lowered.startswith("python"):
        return None

    if len(dep) == len("python"):
        return ""

    next_char = dep[len("python")]
    # Avoid matching packages like python-dateutil.
    if next_char.isalnum() or next_char in {"_", "-"}:
        return None

    spec = dep[len("python") :].strip()
    if spec.startswith(":"):
        spec = spec[1:].strip()
    return spec


def _resolve_from_tox(repo_path: Path) -> PythonVersionResolution | None:
    tox_ini = repo_path / "tox.ini"
    if not tox_ini.exists():
        return None

    text = tox_ini.read_text(errors="ignore")
    env_versions = []
    for env in _TOX_ENV_RE.findall(text):
        normalized = f"{env[0]}.{env[1:]}"
        env_versions.append(normalized)

    supported_env_versions = sorted(
        {v for v in env_versions if v in SUPPORTED_PYTHON_VERSIONS},
        key=Version,
        reverse=True,
    )
    if supported_env_versions:
        return PythonVersionResolution(supported_env_versions[0], "tox.ini:envlist.highest")

    text_versions = _PYTHON_VERSION_RE.findall(text)
    text_versions.extend(_TOX_BASEPY_RE.findall(text))
    supported_text_versions = sorted(
        {v for v in text_versions if v in SUPPORTED_PYTHON_VERSIONS},
        key=Version,
        reverse=True,
    )
    if supported_text_versions:
        return PythonVersionResolution(supported_text_versions[0], "tox.ini:text.highest")
    return None


def _pick_version_from_spec(raw_spec: str) -> str | None:
    spec = raw_spec.strip()
    if not spec:
        return None

    if "||" in spec:
        candidates = []
        for part in (p.strip() for p in spec.split("||")):
            resolved = _pick_version_from_spec(part)
            if resolved:
                candidates.append(resolved)
        if not candidates:
            return None
        candidate_set = set(candidates)
        for version in PREFERRED_PYTHON_VERSIONS:
            if version in candidate_set:
                return version
        return None

    normalized = _normalize_spec(spec)
    if not normalized:
        return None

    try:
        spec_set = SpecifierSet(normalized)
    except InvalidSpecifier:
        return _pick_version_from_text(spec)

    for version in PREFERRED_PYTHON_VERSIONS:
        if Version(version) in spec_set:
            return version
    return None


def _normalize_spec(spec: str) -> str:
    cleaned = spec.strip().replace(" ", "")
    if not cleaned:
        return ""

    if cleaned.startswith("^"):
        base = _pick_version_from_text(cleaned[1:])
        if not base:
            return ""
        major = int(base.split(".")[0])
        return f">={base},<{major + 1}.0"

    if cleaned.startswith("~") and not cleaned.startswith("~="):
        base = _pick_version_from_text(cleaned[1:])
        if not base:
            return ""
        major, minor = base.split(".")
        return f">={base},<{major}.{int(minor) + 1}"

    if re.fullmatch(r"\d+\.\d+(\.\d+)?", cleaned):
        short = _pick_version_from_text(cleaned)
        if short:
            return f"=={short}.*"

    return cleaned


def _pick_version_from_text(text: str) -> str | None:
    matches = _PYTHON_VERSION_RE.findall(text)
    if not matches:
        return None

    match_set = set(matches)
    for version in PREFERRED_PYTHON_VERSIONS:
        if version in match_set:
            return version
    return None
