"""Build Docker images for PR-specific repository execution environments."""

from __future__ import annotations

import logging
import os
import re
import shutil
import shlex
import stat
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from .specs import PREnvironmentSpec

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_ROOT = Path(__file__).resolve().parent / "dockerfiles"
DEFAULT_REPO_CACHE_ROOT = Path(__file__).resolve().parent.parent / "repos"
DEFAULT_UV_BINARY = Path(__file__).resolve().parent / "assets" / "uv"
DEFAULT_CLAUDE_BINARY = Path(__file__).resolve().parent / "assets" / "claude"
DEFAULT_APT_PROXY = "http://host.docker.internal:3142"
DEFAULT_PIP_INDEX_URL = "http://host.docker.internal:3141/root/pypi/+simple"
DEFAULT_PIP_EXTRA_INDEX_URL: str | None = None
DEFAULT_PIP_TRUSTED_HOST = "host.docker.internal"
DEFAULT_SWE_CARE_BUILD_OVERRIDES_ROOT = (
    Path(__file__).resolve().parent / "assets" / "swe_care_build_overrides"
)
_REPO_SEED_LOCKS: dict[str, threading.Lock] = {}
_REPO_SEED_LOCKS_GUARD = threading.Lock()
_BUILDKIT_FRONTEND_GRPC_ERROR = "frontend grpc server closed unexpectedly"
_OVERRIDE_COMMIT_DIR_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


@dataclass(frozen=True)
class BuildResult:
    """Result of building one docker image."""

    image_tag: str
    context_dir: Path
    dockerfile_path: Path
    command: tuple[str, ...]
    build_log_path: Path | None = None


@dataclass(frozen=True)
class BuildScriptOverrides:
    """Optional repo/commit-specific manual build script overrides."""

    source_dir: Path | None = None
    python_version: str | None = None
    dockerfile_path: Path | None = None
    setup_repo_path: Path | None = None
    install_deps_path: Path | None = None
    post_install_path: Path | None = None

    @property
    def has_any(self) -> bool:
        return any(
            (
                self.python_version,
                self.dockerfile_path is not None,
                self.setup_repo_path is not None,
                self.install_deps_path is not None,
                self.post_install_path is not None,
            )
        )


def spec_from_instance(
    instance: dict,
    *,
    use_merged_commit: bool = False,
    languages: tuple[str, ...] = ("python",),
    python_version: str = "3.11",
    image_tag: str | None = None,
    apt_packages: tuple[str, ...] | None = None,
    extra_pip_packages: tuple[str, ...] | None = None,
) -> PREnvironmentSpec:
    """Create a PREnvironmentSpec from a dataset-style instance dict."""
    if "repo" not in instance:
        raise ValueError("instance is missing required key: repo")

    if use_merged_commit:
        commit = instance.get("merged_commit")
    else:
        commit = instance.get("commit_to_review", {}).get("head_commit")

    if not commit:
        raise ValueError("instance does not include a usable commit hash")

    kwargs: dict = {}
    if apt_packages is not None:
        kwargs["apt_packages"] = apt_packages
    if extra_pip_packages is not None:
        kwargs["extra_pip_packages"] = extra_pip_packages

    return PREnvironmentSpec(
        repo=instance["repo"],
        pull_number=instance.get("pull_number"),
        commit=commit,
        languages=languages,
        python_version=python_version,
        image_tag=image_tag,
        **kwargs,
    )


def prepare_build_context(
    spec: PREnvironmentSpec,
    context_root: Path = DEFAULT_CONTEXT_ROOT,
    cached_repos_dir: Path | None = DEFAULT_REPO_CACHE_ROOT,
    build_overrides_root: Path | None = None,
) -> Path:
    """Generate a docker build context for the given spec."""
    _assert_language_implementation(spec)

    context_root.mkdir(parents=True, exist_ok=True)
    context_dir = context_root / spec.docker_context_dirname
    context_dir.mkdir(parents=True, exist_ok=True)

    dockerfile_path = context_dir / "Dockerfile"
    setup_script = context_dir / "setup_repo.sh"
    install_script = context_dir / "install_deps.sh"
    post_install_script = context_dir / "post_install.sh"
    uv_binary = context_dir / "uv"
    claude_binary = context_dir / "claude"
    seeded_repo_dir = context_dir / "repo_seed"
    overrides = resolve_build_script_overrides(
        spec.repo,
        spec.commit,
        overrides_root=build_overrides_root,
    )

    seed_used = _prepare_cached_repo_seed(
        spec,
        seed_dir=seeded_repo_dir,
        cached_repos_dir=cached_repos_dir,
    )

    if overrides.dockerfile_path is not None:
        dockerfile_path.write_text(overrides.dockerfile_path.read_text(encoding="utf-8"))
    else:
        dockerfile_path.write_text(_render_dockerfile(spec))
    if overrides.setup_repo_path is not None:
        _write_executable(setup_script, overrides.setup_repo_path.read_text(encoding="utf-8"))
    else:
        _write_executable(setup_script, _render_setup_repo_script(spec))
    if overrides.install_deps_path is not None:
        _write_executable(install_script, overrides.install_deps_path.read_text(encoding="utf-8"))
    else:
        _write_executable(install_script, _render_install_deps_script(spec))
    if overrides.post_install_path is not None:
        _write_executable(
            post_install_script,
            overrides.post_install_path.read_text(encoding="utf-8"),
        )
    else:
        _write_executable(post_install_script, _render_post_install_script())
    if "python" in spec.languages:
        if not DEFAULT_UV_BINARY.is_file():
            raise FileNotFoundError(f"uv binary not found at {DEFAULT_UV_BINARY}")
        if uv_binary.exists() or uv_binary.is_symlink():
            uv_binary.unlink()
        try:
            os.link(DEFAULT_UV_BINARY, uv_binary)
        except OSError:
            shutil.copy2(DEFAULT_UV_BINARY, uv_binary)

    if (
        "python" in spec.languages
        or "javascript" in spec.languages
        or "typescript" in spec.languages
    ):
        if not DEFAULT_CLAUDE_BINARY.is_file():
            raise FileNotFoundError(f"claude binary not found at {DEFAULT_CLAUDE_BINARY}")
        if claude_binary.exists() or claude_binary.is_symlink():
            claude_binary.unlink()
        try:
            os.link(DEFAULT_CLAUDE_BINARY, claude_binary)
        except OSError:
            shutil.copy2(DEFAULT_CLAUDE_BINARY, claude_binary)

    logger.info(
        "Prepared docker build context at %s (repo seed: %s, override: %s)",
        context_dir,
        "cache" if seed_used else "network-clone fallback",
        str(overrides.source_dir) if overrides.source_dir is not None else "none",
    )
    return context_dir


def build_image(
    spec: PREnvironmentSpec,
    *,
    context_root: Path = DEFAULT_CONTEXT_ROOT,
    cached_repos_dir: Path | None = DEFAULT_REPO_CACHE_ROOT,
    build_overrides_root: Path | None = None,
    no_cache: bool = False,
    pull_base: bool = False,
    build_log_path: Path | None = None,
    build_timeout_sec: int | None = None,
    apt_proxy: str | None = DEFAULT_APT_PROXY,
    pip_index_url: str | None = DEFAULT_PIP_INDEX_URL,
    pip_extra_index_url: str | None = DEFAULT_PIP_EXTRA_INDEX_URL,
    pip_trusted_host: str | None = DEFAULT_PIP_TRUSTED_HOST,
) -> BuildResult:
    """Build the docker image for a PR/repo commit."""
    _assert_language_implementation(spec)
    _ensure_docker_available()
    context_dir = prepare_build_context(
        spec,
        context_root=context_root,
        cached_repos_dir=cached_repos_dir,
        build_overrides_root=build_overrides_root,
    )
    dockerfile_path = context_dir / "Dockerfile"
    cmd = ["docker", "build", "-f", str(dockerfile_path), "-t", spec.resolved_image_tag]
    build_arg_values = (
        ("REVIEWBENCH_APT_PROXY", apt_proxy),
        ("REVIEWBENCH_PIP_INDEX_URL", pip_index_url),
        ("REVIEWBENCH_PIP_EXTRA_INDEX_URL", pip_extra_index_url),
        ("REVIEWBENCH_PIP_TRUSTED_HOST", pip_trusted_host),
    )
    resolved_build_args: list[tuple[str, str]] = []
    for build_arg_name, raw_value in build_arg_values:
        if raw_value is None:
            continue
        value = raw_value.strip()
        if not value:
            continue
        resolved_build_args.append((build_arg_name, value))
        cmd.extend(["--build-arg", f"{build_arg_name}={value}"])
    if any("host.docker.internal" in value for _, value in resolved_build_args):
        cmd.extend(["--add-host", "host.docker.internal:host-gateway"])
    if no_cache:
        cmd.append("--no-cache")
    if pull_base:
        cmd.append("--pull")
    cmd.append(str(context_dir))

    logger.info("Building image %s", spec.resolved_image_tag)
    build_env = os.environ.copy()
    build_env.setdefault("DOCKER_BUILDKIT", "1")
    first_retryable_result: subprocess.CompletedProcess[str] | None = None
    result: subprocess.CompletedProcess[str] | None = None
    try:
        for attempt in (1, 2):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=build_timeout_sec,
                env=build_env,
            )
            if result.returncode == 0:
                break
            if (
                attempt == 1
                and _has_retryable_buildkit_frontend_error(result.stdout, result.stderr)
            ):
                first_retryable_result = result
                logger.warning(
                    "Docker build hit transient BuildKit frontend error for %s; retrying once.",
                    spec.resolved_image_tag,
                )
                continue
            break
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        timeout_note = (
            f"\n\nBuild timed out after {build_timeout_sec} seconds."
            if build_timeout_sec is not None
            else "\n\nBuild timed out."
        )
        if build_log_path is not None:
            _write_build_log(build_log_path, cmd, stdout, stderr + timeout_note)
        raise RuntimeError(
            (
                f"Docker build timed out for {spec.resolved_image_tag} "
                f"after {build_timeout_sec} seconds.\n"
                f"stdout:\n{stdout[-4000:]}\n"
                f"stderr:\n{stderr[-4000:]}"
            )
        ) from exc
    if result is None:
        raise RuntimeError(
            f"Docker build did not produce a result for {spec.resolved_image_tag}"
        )
    if build_log_path is not None:
        if first_retryable_result is None:
            _write_build_log(build_log_path, cmd, result.stdout, result.stderr)
        else:
            retry_note = (
                "Retry note: first attempt failed with retryable BuildKit frontend "
                "error and was retried once.\n\n"
            )
            stdout_with_attempts = (
                "===== ATTEMPT 1 STDOUT (retryable failure) =====\n"
                f"{first_retryable_result.stdout}\n\n"
                "===== ATTEMPT 2 STDOUT =====\n"
                f"{result.stdout}"
            )
            stderr_with_attempts = (
                retry_note
                + "===== ATTEMPT 1 STDERR (retryable failure) =====\n"
                + f"{first_retryable_result.stderr}\n\n"
                + "===== ATTEMPT 2 STDERR =====\n"
                + f"{result.stderr}"
            )
            _write_build_log(
                build_log_path,
                cmd,
                stdout_with_attempts,
                stderr_with_attempts,
            )
    if result.returncode != 0:
        msg = (
            f"Docker build failed for {spec.resolved_image_tag}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
        if first_retryable_result is not None:
            msg = (
                f"{msg}\n\n"
                "First attempt failed with retryable BuildKit frontend error and was retried once."
            )
        raise RuntimeError(msg)

    return BuildResult(
        image_tag=spec.resolved_image_tag,
        context_dir=context_dir,
        dockerfile_path=dockerfile_path,
        command=tuple(cmd),
        build_log_path=build_log_path,
    )


def _ensure_docker_available() -> None:
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("docker CLI was not found in PATH") from exc

    if result.returncode != 0:
        raise RuntimeError(
            "docker daemon is not reachable. Make sure Docker is running."
        )


def resolve_build_script_overrides(
    repo: str,
    commit: str | None,
    *,
    overrides_root: Path | None = DEFAULT_SWE_CARE_BUILD_OVERRIDES_ROOT,
) -> BuildScriptOverrides:
    """Resolve optional manual build overrides for a repo+commit pair."""
    if overrides_root is None:
        return BuildScriptOverrides()

    repo_override_dir = _resolve_repo_override_dir(Path(overrides_root), repo)
    if repo_override_dir is None:
        return BuildScriptOverrides()

    selected_dir = _select_override_dir_for_commit(repo_override_dir, commit)
    if selected_dir is None:
        selected_dir = repo_override_dir if _has_override_files(repo_override_dir) else None
    if selected_dir is None:
        return BuildScriptOverrides()

    python_version = _load_python_version_override(selected_dir / ".python-version")
    dockerfile_path = selected_dir / "Dockerfile"
    setup_repo_path = selected_dir / "setup_repo.sh"
    install_deps_path = selected_dir / "install_deps.sh"
    post_install_path = selected_dir / "post_install.sh"
    return BuildScriptOverrides(
        source_dir=selected_dir,
        python_version=python_version,
        dockerfile_path=dockerfile_path if dockerfile_path.is_file() else None,
        setup_repo_path=setup_repo_path if setup_repo_path.is_file() else None,
        install_deps_path=install_deps_path if install_deps_path.is_file() else None,
        post_install_path=post_install_path if post_install_path.is_file() else None,
    )


def _resolve_repo_override_dir(overrides_root: Path, repo: str) -> Path | None:
    """Resolve repo override directory, matching repo slug case-insensitively."""
    expected_dir_name = repo.replace("/", "__")
    exact_dir = overrides_root / expected_dir_name
    if exact_dir.is_dir():
        return exact_dir

    if not overrides_root.is_dir():
        return None

    lowered_expected = expected_dir_name.lower()
    candidates = [
        child
        for child in overrides_root.iterdir()
        if child.is_dir() and child.name.lower() == lowered_expected
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous case-insensitive override directories under {overrides_root}: "
            f"{', '.join(sorted(path.name for path in candidates))}"
        )
    return candidates[0]


def _has_override_files(directory: Path) -> bool:
    return any(
        (directory / name).is_file()
        for name in (
            ".python-version",
            "Dockerfile",
            "setup_repo.sh",
            "install_deps.sh",
            "post_install.sh",
        )
    )


def _select_override_dir_for_commit(repo_override_dir: Path, commit: str | None) -> Path | None:
    if not commit:
        return None
    normalized_commit = commit.strip().lower()
    if len(normalized_commit) < 7:
        return None

    exact_dir = repo_override_dir / normalized_commit
    if exact_dir.is_dir():
        return exact_dir

    candidates: list[Path] = []
    for child in repo_override_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.strip()
        if _OVERRIDE_COMMIT_DIR_RE.fullmatch(name) is None:
            continue
        if normalized_commit.startswith(name.lower()):
            candidates.append(child)

    if not candidates:
        return None
    candidates.sort(key=lambda path: len(path.name), reverse=True)
    top_len = len(candidates[0].name)
    top_candidates = [path for path in candidates if len(path.name) == top_len]
    unique_top = {path.name.lower() for path in top_candidates}
    if len(unique_top) > 1:
        raise ValueError(
            f"Ambiguous override directories under {repo_override_dir}: "
            f"{', '.join(sorted(unique_top))}"
        )
    return candidates[0]


def _load_python_version_override(path: Path) -> str | None:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            return line
    raise ValueError(f"python version override file is empty: {path}")


def _assert_language_implementation(spec: PREnvironmentSpec) -> None:
    implemented_selected = any(
        lang in ("python", "javascript", "typescript") for lang in spec.languages
    )
    if not implemented_selected:
        raise NotImplementedError(
            "No implemented language selected. "
            "Implemented languages: python, javascript, typescript."
        )
    if spec.unimplemented_languages:
        logger.info(
            "Additional languages declared but currently ignored by builder: %s",
            spec.unimplemented_languages,
        )


def _render_dockerfile(spec: PREnvironmentSpec) -> str:
    if "python" in spec.languages:
        return _render_dockerfile_python(spec)
    if "javascript" in spec.languages or "typescript" in spec.languages:
        return _render_dockerfile_js_ts(spec)
    raise NotImplementedError(f"No Dockerfile renderer for language: {spec.primary_language}")


def _render_dockerfile_python(spec: PREnvironmentSpec) -> str:
    apt_packages = _dedupe(spec.apt_packages)
    apt_lines = " \\\n    ".join(apt_packages)

    return f"""# syntax=docker/dockerfile:1.4
FROM python:{spec.python_version}-slim

ARG REVIEWBENCH_APT_PROXY={DEFAULT_APT_PROXY}
ARG REVIEWBENCH_PIP_INDEX_URL={DEFAULT_PIP_INDEX_URL}
ARG REVIEWBENCH_PIP_EXTRA_INDEX_URL=
ARG REVIEWBENCH_PIP_TRUSTED_HOST={DEFAULT_PIP_TRUSTED_HOST}

ENV DEBIAN_FRONTEND=noninteractive \\
    PIP_DISABLE_PIP_VERSION_CHECK=1 \\
    PIP_NO_CACHE_DIR=1 \\
    REVIEWBENCH_PIP_INDEX_URL=${{REVIEWBENCH_PIP_INDEX_URL}} \\
    REVIEWBENCH_PIP_EXTRA_INDEX_URL=${{REVIEWBENCH_PIP_EXTRA_INDEX_URL}} \\
    REVIEWBENCH_PIP_TRUSTED_HOST=${{REVIEWBENCH_PIP_TRUSTED_HOST}} \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

RUN APT_PROXY_FILE=/etc/apt/apt.conf.d/99reviewbench-proxy && \\
    APT_UPDATE_LOG=/tmp/reviewbench-apt-update.log && \\
    if [ -n "$REVIEWBENCH_APT_PROXY" ]; then \\
      echo "Acquire::http::Proxy \\"$REVIEWBENCH_APT_PROXY\\";" > "$APT_PROXY_FILE" && \\
      echo "Acquire::https::Proxy \\"$REVIEWBENCH_APT_PROXY\\";" >> "$APT_PROXY_FILE" ; \\
    fi && \\
    apt_status=0; \\
    apt-get update > "$APT_UPDATE_LOG" 2>&1 || apt_status=$?; \\
    cat "$APT_UPDATE_LOG" && \\
    if [ "$apt_status" -ne 0 ] || ( [ -n "$REVIEWBENCH_APT_PROXY" ] && grep -Eiq "Could not resolve 'host\\\\.docker\\\\.internal'|Failed to fetch .*host\\\\.docker\\\\.internal|Temporary failure in name resolution|Name or service not known|Connection refused|ProxyError|timed out|Could not connect" "$APT_UPDATE_LOG" ); then \\
      if [ -n "$REVIEWBENCH_APT_PROXY" ]; then \\
        echo "Warning: apt proxy unavailable ($REVIEWBENCH_APT_PROXY); retrying direct." >&2 && \\
        rm -f "$APT_PROXY_FILE" && \\
        apt-get update; \\
      else \\
        exit "$apt_status"; \\
      fi; \\
    fi && \\
    rm -f "$APT_UPDATE_LOG" && \\
    apt-get install -y --no-install-recommends \\
    {apt_lines} && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY uv /usr/local/bin/uv
COPY claude /usr/local/bin/claude
COPY setup_repo.sh /tmp/setup_repo.sh
COPY install_deps.sh /tmp/install_deps.sh
COPY post_install.sh /tmp/post_install.sh

RUN --mount=type=bind,source=repo_seed,target=/tmp/repo_seed,readonly \\
    chmod +x /usr/local/bin/uv /usr/local/bin/claude /tmp/setup_repo.sh /tmp/install_deps.sh /tmp/post_install.sh && \\
    /tmp/setup_repo.sh && \\
    /tmp/install_deps.sh && \\
    /tmp/post_install.sh

WORKDIR {spec.workdir}
CMD ["/bin/bash"]
"""


def _render_dockerfile_js_ts(spec: PREnvironmentSpec) -> str:
    apt_packages = _dedupe(spec.apt_packages)
    apt_lines = " \\\n    ".join(apt_packages)
    return f"""# syntax=docker/dockerfile:1.4
FROM debian:bookworm-slim

ARG REVIEWBENCH_APT_PROXY={DEFAULT_APT_PROXY}

ENV DEBIAN_FRONTEND=noninteractive

RUN APT_PROXY_FILE=/etc/apt/apt.conf.d/99reviewbench-proxy && \\
    APT_UPDATE_LOG=/tmp/reviewbench-apt-update.log && \\
    if [ -n "$REVIEWBENCH_APT_PROXY" ]; then \\
      echo "Acquire::http::Proxy \\"$REVIEWBENCH_APT_PROXY\\";" > "$APT_PROXY_FILE" && \\
      echo "Acquire::https::Proxy \\"$REVIEWBENCH_APT_PROXY\\";" >> "$APT_PROXY_FILE" ; \\
    fi && \\
    apt_status=0; \\
    apt-get update > "$APT_UPDATE_LOG" 2>&1 || apt_status=$?; \\
    cat "$APT_UPDATE_LOG" && \\
    if [ "$apt_status" -ne 0 ] || ( [ -n "$REVIEWBENCH_APT_PROXY" ] && grep -Eiq "Could not resolve 'host\\\\.docker\\\\.internal'|Failed to fetch .*host\\\\.docker\\\\.internal|Temporary failure in name resolution|Name or service not known|Connection refused|ProxyError|timed out|Could not connect" "$APT_UPDATE_LOG" ); then \\
      if [ -n "$REVIEWBENCH_APT_PROXY" ]; then \\
        echo "Warning: apt proxy unavailable ($REVIEWBENCH_APT_PROXY); retrying direct." >&2 && \\
        rm -f "$APT_PROXY_FILE" && \\
        apt-get update; \\
      else \\
        exit "$apt_status"; \\
      fi; \\
    fi && \\
    rm -f "$APT_UPDATE_LOG" && \\
    apt-get install -y --no-install-recommends \\
    {apt_lines} && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY claude /usr/local/bin/claude
COPY setup_repo.sh /tmp/setup_repo.sh
COPY install_deps.sh /tmp/install_deps.sh
COPY post_install.sh /tmp/post_install.sh

RUN --mount=type=bind,source=repo_seed,target=/tmp/repo_seed,readonly \\
    chmod +x /usr/local/bin/claude /tmp/setup_repo.sh /tmp/install_deps.sh /tmp/post_install.sh && \\
    /tmp/setup_repo.sh && \\
    /tmp/install_deps.sh && \\
    /tmp/post_install.sh

WORKDIR {spec.workdir}
CMD ["/bin/bash"]
"""


def _render_post_install_script() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euxo pipefail",
            "",
            "# Optional post-install hook. Default behavior is no-op.",
            "exit 0",
        ]
    ) + "\n"


def _render_setup_repo_script(spec: PREnvironmentSpec) -> str:
    repo_url = f"https://github.com/{spec.repo}.git"
    lines = [
        "#!/usr/bin/env bash",
        "set -euxo pipefail",
        "",
        f"REPO_URL={shlex.quote(repo_url)}",
        'TARGET_DIR="/workspace"',
        'cd "$TARGET_DIR"',
        'if [[ -d "/tmp/repo_seed/.git" ]]; then',
        '  cp -a /tmp/repo_seed/. "$TARGET_DIR"/',
        "else",
        '  git clone --no-checkout "$REPO_URL" .',
        "fi",
        'cd "$TARGET_DIR"',
        'if git remote | grep -q "^origin$"; then',
        '  git remote set-url origin "$REPO_URL"',
        "else",
        '  git remote add origin "$REPO_URL"',
        "fi",
        "git config advice.detachedHead false",
    ]

    if spec.pull_number is not None:
        lines.append(f"git fetch --no-tags --force origin pull/{spec.pull_number}/head:pr-head")

    if spec.commit:
        lines.append(f"git fetch --no-tags --force origin {shlex.quote(spec.commit)} || true")
        lines.append(f"git checkout --force {shlex.quote(spec.commit)}")
    elif spec.pull_number is not None:
        lines.append("git checkout --force pr-head")
    else:
        lines.append("git fetch --no-tags --force origin HEAD:origin-head")
        lines.append("git checkout --force origin-head")

    lines.extend(
        [
            "git reset --hard",
            "git clean -fdx",
            "git submodule sync --recursive || true",
            "git submodule update --init --recursive || true",
            "git rev-parse HEAD > /tmp/checked_out_commit.txt",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_install_deps_script(spec: PREnvironmentSpec) -> str:
    # Future extension: merge outputs from multiple language installers.
    if "python" in spec.languages:
        return _render_install_deps_script_python(spec)
    if "javascript" in spec.languages or "typescript" in spec.languages:
        return _render_install_deps_script_js_ts(spec)
    raise NotImplementedError(f"No dependency installer for language: {spec.primary_language}")


def _render_install_deps_script_js_ts(spec: PREnvironmentSpec) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euxo pipefail",
        "",
        'cd "/workspace"',
        'echo "Using JS/TS installer path for languages: '
        + ",".join(spec.languages)
        + '"',
        "if command -v node >/dev/null 2>&1; then",
        "  node --version",
        "elif command -v nodejs >/dev/null 2>&1; then",
        "  nodejs --version",
        "else",
        '  echo "Error: node/nodejs is not installed in the container." >&2',
        "  exit 1",
        "fi",
        "if command -v npm >/dev/null 2>&1; then",
        "  npm --version",
        "else",
        '  echo "Error: npm is not installed in the container." >&2',
        "  exit 1",
        "fi",
    ]
    return "\n".join(lines) + "\n"


def _render_install_deps_script_python(spec: PREnvironmentSpec) -> str:
    pip_pkgs = _dedupe(spec.extra_pip_packages)
    pip_install_line = ""
    if pip_pkgs:
        quoted_pkgs = " ".join(shlex.quote(pkg) for pkg in pip_pkgs)
        pip_install_line = (
            f'run_repo_pip_install "strict" "/tmp/reviewbench-extra-pip.log" {quoted_pkgs}'
        )

    lines = [
        "#!/usr/bin/env bash",
        "set -euxo pipefail",
        "ulimit -n 65536",
        "",
        'cd "/workspace"',
        'cat > /tmp/reviewbench-pip-constraints.txt << "EOF"',
        "pip<24.1",
        "setuptools<66",
        "wheel<0.46",
        "EOF",
        'cat > /tmp/reviewbench-pip-constraints-legacy.txt << "EOF"',
        "pip<24.1",
        "setuptools<58",
        "wheel<0.46",
        "EOF",
        'cat > /tmp/reviewbench-pip-constraints-cython-legacy.txt << "EOF"',
        "pip<24.1",
        "setuptools<66",
        "wheel<0.46",
        "Cython<3",
        "EOF",
        'cat > /tmp/reviewbench-pip-constraints-global.txt << "EOF"',
        "setuptools<82",
        "EOF",
        'cat > /tmp/reviewbench-pip-constraints-modern.txt << "EOF"',
        "pip<25.3",
        "setuptools>=75,<82",
        "wheel<0.46",
        "EOF",
        "REVIEWBENCH_CONSTRAINT_GLOBAL=/tmp/reviewbench-pip-constraints-global.txt",
        "REVIEWBENCH_CONSTRAINT_LEGACY=/tmp/reviewbench-pip-constraints-legacy.txt",
        "REVIEWBENCH_CONSTRAINT_CYTHON_LEGACY=/tmp/reviewbench-pip-constraints-cython-legacy.txt",
        'if [[ -x "/usr/local/bin/python" ]]; then',
        '  REVIEWBENCH_PYTHON_BIN="/usr/local/bin/python"',
        "elif command -v python >/dev/null 2>&1; then",
        '  REVIEWBENCH_PYTHON_BIN="$(command -v python)"',
        "elif command -v python3 >/dev/null 2>&1; then",
        '  REVIEWBENCH_PYTHON_BIN="$(command -v python3)"',
        "else",
        '  echo "Error: no Python interpreter found in container." >&2',
        "  exit 1",
        "fi",
        "export REVIEWBENCH_PYTHON_BIN",
        'echo "Using Python interpreter: ${REVIEWBENCH_PYTHON_BIN}"',
        'REVIEWBENCH_IS_PY313_PLUS=$("$REVIEWBENCH_PYTHON_BIN" -c '"'"'import sys; print("1" if sys.version_info >= (3, 13) else "0")'"'"')',
        'REVIEWBENCH_IS_PY37_OR_LOWER=$("$REVIEWBENCH_PYTHON_BIN" -c '"'"'import sys; print("1" if sys.version_info < (3, 8) else "0")'"'"')',
        'REVIEWBENCH_CAN_USE_LEGACY_SETUPTOOLS=$("$REVIEWBENCH_PYTHON_BIN" -c '"'"'import sys; print("1" if sys.version_info < (3, 13) else "0")'"'"')',
        'if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" == "1" ]]; then',
        '  echo "Info: using python -m pip for Python <=3.7 (uv unsupported)."',
        "fi",
        'if [[ "$REVIEWBENCH_IS_PY313_PLUS" == "1" ]]; then',
        "  REVIEWBENCH_CONSTRAINT_STRICT=/tmp/reviewbench-pip-constraints-modern.txt",
        "else",
        "  REVIEWBENCH_CONSTRAINT_STRICT=/tmp/reviewbench-pip-constraints.txt",
        "fi",
        "export UV_CONSTRAINT=\"$REVIEWBENCH_CONSTRAINT_STRICT\"",
        "export PIP_CONSTRAINT=\"$REVIEWBENCH_CONSTRAINT_STRICT\"",
        'export MAKEFLAGS="-j8"',
        'if [[ -n "${REVIEWBENCH_PIP_INDEX_URL:-}" ]]; then',
        '  export PIP_INDEX_URL="$REVIEWBENCH_PIP_INDEX_URL"',
        '  export UV_INDEX_URL="$REVIEWBENCH_PIP_INDEX_URL"',
        "fi",
        'if [[ -n "${REVIEWBENCH_PIP_EXTRA_INDEX_URL:-}" ]]; then',
        '  export PIP_EXTRA_INDEX_URL="$REVIEWBENCH_PIP_EXTRA_INDEX_URL"',
        '  export UV_EXTRA_INDEX_URL="$REVIEWBENCH_PIP_EXTRA_INDEX_URL"',
        "fi",
        'if [[ -n "${REVIEWBENCH_PIP_TRUSTED_HOST:-}" ]]; then',
        '  export PIP_TRUSTED_HOST="$REVIEWBENCH_PIP_TRUSTED_HOST"',
        '  export UV_INSECURE_HOST="${REVIEWBENCH_PIP_TRUSTED_HOST// /,}"',
        "fi",
        "uv --version",
        'if [[ "$REVIEWBENCH_IS_PY313_PLUS" == "1" ]]; then',
        '  if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" == "1" ]]; then',
        '    "$REVIEWBENCH_PYTHON_BIN" -m pip install --upgrade '"'"'pip<25.3'"'"' '"'"'setuptools>=75,<82'"'"' '"'"'wheel<0.46'"'"' '"'"'pycparser'"'"' \\',
        '      || echo "Warning: failed to pin pip/setuptools/wheel toolchain"',
        "  else",
        '    uv pip install --python "$REVIEWBENCH_PYTHON_BIN" --upgrade '"'"'pip<25.3'"'"' '"'"'setuptools>=75,<82'"'"' '"'"'wheel<0.46'"'"' '"'"'pycparser'"'"' \\',
        '      || echo "Warning: failed to pin pip/setuptools/wheel toolchain"',
        "  fi",
        "else",
        '  if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" == "1" ]]; then',
        '    "$REVIEWBENCH_PYTHON_BIN" -m pip install --upgrade '"'"'pip<24.1'"'"' '"'"'setuptools<66'"'"' '"'"'wheel<0.46'"'"' '"'"'pycparser'"'"' \\',
        '      || echo "Warning: failed to pin pip/setuptools/wheel toolchain"',
        "  else",
        '    uv pip install --python "$REVIEWBENCH_PYTHON_BIN" --upgrade '"'"'pip<24.1'"'"' '"'"'setuptools<66'"'"' '"'"'wheel<0.46'"'"' '"'"'pycparser'"'"' \\',
        '      || echo "Warning: failed to pin pip/setuptools/wheel toolchain"',
        "  fi",
        "fi",
        "",
        "run_pip_install_with_mode() {",
        '  local mode="$1"',
        '  local log_file="$2"',
        "  shift 2",
        "  set +e",
        '  if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" == "1" ]]; then',
        '    case "$mode" in',
        "      strict)",
        '        PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_STRICT" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee "$log_file"',
        "        ;;",
        "      legacy)",
        '        PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_LEGACY" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee "$log_file"',
        "        ;;",
        "      cython_legacy)",
        '        PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_CYTHON_LEGACY" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee "$log_file"',
        "        ;;",
        "      none)",
        '        PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_GLOBAL" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee "$log_file"',
        "        ;;",
        "      *)",
        '        echo "Unknown package install mode: ${mode}" >&2',
        "        set -e",
        "        return 2",
        "        ;;",
        "    esac",
        "  else",
        '    case "$mode" in',
        "      strict)",
        '        UV_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_STRICT" uv pip install --python "$REVIEWBENCH_PYTHON_BIN" "$@" 2>&1 | tee "$log_file"',
        "        ;;",
        "      legacy)",
        '        UV_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_LEGACY" uv pip install --python "$REVIEWBENCH_PYTHON_BIN" "$@" 2>&1 | tee "$log_file"',
        "        ;;",
        "      cython_legacy)",
        '        UV_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_CYTHON_LEGACY" uv pip install --python "$REVIEWBENCH_PYTHON_BIN" "$@" 2>&1 | tee "$log_file"',
        "        ;;",
        "      none)",
        '        UV_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_GLOBAL" uv pip install --python "$REVIEWBENCH_PYTHON_BIN" "$@" 2>&1 | tee "$log_file"',
        "        ;;",
        "      *)",
        '        echo "Unknown package install mode: ${mode}" >&2',
        "        set -e",
        "        return 2",
        "        ;;",
        "    esac",
        "  fi",
        "  local status=${PIPESTATUS[0]}",
        '  local active_index="${UV_INDEX_URL:-${PIP_INDEX_URL:-}}"',
        '  if [[ "$status" -ne 0 && -n "$active_index" ]] \\',
        '    && grep -Eiq "Temporary failure in name resolution|Name or service not known|Failed to establish a new connection|Connection refused|ProxyError|ConnectTimeoutError|timed out|Could not fetch URL" "$log_file"; then',
        '    echo "Warning: package index ${active_index} failed; retrying with default index." | tee -a "$log_file"',
        '    if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" == "1" ]]; then',
        '      case "$mode" in',
        "        strict)",
        '          env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INSECURE_HOST PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_STRICT" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee -a "$log_file"',
        "          ;;",
        "        legacy)",
        '          env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INSECURE_HOST PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_LEGACY" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee -a "$log_file"',
        "          ;;",
        "        cython_legacy)",
        '          env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INSECURE_HOST PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_CYTHON_LEGACY" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee -a "$log_file"',
        "          ;;",
        "        none)",
        '          env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INSECURE_HOST PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_GLOBAL" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee -a "$log_file"',
        "          ;;",
        "      esac",
        "    else",
        '      case "$mode" in',
        "        strict)",
        '          env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INSECURE_HOST -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST UV_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_STRICT" uv pip install --python "$REVIEWBENCH_PYTHON_BIN" "$@" 2>&1 | tee -a "$log_file"',
        "          ;;",
        "        legacy)",
        '          env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INSECURE_HOST -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST UV_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_LEGACY" uv pip install --python "$REVIEWBENCH_PYTHON_BIN" "$@" 2>&1 | tee -a "$log_file"',
        "          ;;",
        "        cython_legacy)",
        '          env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INSECURE_HOST -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST UV_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_CYTHON_LEGACY" uv pip install --python "$REVIEWBENCH_PYTHON_BIN" "$@" 2>&1 | tee -a "$log_file"',
        "          ;;",
        "        none)",
        '          env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_INSECURE_HOST -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TRUSTED_HOST UV_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_GLOBAL" uv pip install --python "$REVIEWBENCH_PYTHON_BIN" "$@" 2>&1 | tee -a "$log_file"',
        "          ;;",
        "      esac",
        "    fi",
        "    status=${PIPESTATUS[0]}",
        "  fi",
        '  if [[ "$status" -ne 0 && "$REVIEWBENCH_IS_PY37_OR_LOWER" != "1" ]] \\',
        '    && grep -Eiq "invalid package format|The metadata at .* is invalid" "$log_file"; then',
        '    echo "Info: uv rejected package metadata; retrying install with pip resolver." | tee -a "$log_file"',
        '    case "$mode" in',
        "      strict)",
        '        PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_STRICT" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee -a "$log_file"',
        "        ;;",
        "      legacy)",
        '        PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_LEGACY" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee -a "$log_file"',
        "        ;;",
        "      cython_legacy)",
        '        PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_CYTHON_LEGACY" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee -a "$log_file"',
        "        ;;",
        "      none)",
        '        PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_GLOBAL" "$REVIEWBENCH_PYTHON_BIN" -m pip install "$@" 2>&1 | tee -a "$log_file"',
        "        ;;",
        "      *)",
        '        echo "Unknown package install mode: ${mode}" >&2',
        "        set -e",
        "        return 2",
        "        ;;",
        "    esac",
        "    status=${PIPESTATUS[0]}",
        "  fi",
        "  set -e",
        '  return "$status"',
        "}",
        "",
        "upgrade_python_packages() {",
        '  local log_file="$1"',
        "  shift",
        "  set +e",
        '  if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" == "1" ]]; then',
        '    PIP_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_GLOBAL" "$REVIEWBENCH_PYTHON_BIN" -m pip install --upgrade "$@" 2>&1 | tee -a "$log_file"',
        "  else",
        '    UV_CONSTRAINT="$REVIEWBENCH_CONSTRAINT_GLOBAL" uv pip install --python "$REVIEWBENCH_PYTHON_BIN" --upgrade "$@" 2>&1 | tee -a "$log_file"',
        "  fi",
        "  local status=${PIPESTATUS[0]}",
        "  set -e",
        '  return "$status"',
        "}",
        "",
        "ensure_apt_packages() {",
        '  local log_file="$1"',
        "  shift",
        '  if [[ "$#" -eq 0 ]]; then',
        "    return 0",
        "  fi",
        "  if ! command -v apt-get >/dev/null 2>&1; then",
        '    echo "Warning: apt-get is unavailable; cannot install system packages: $*" | tee -a "$log_file"',
        "    return 1",
        "  fi",
        '  echo "Info: installing system packages: $*" | tee -a "$log_file"',
        "  set +e",
        '  DEBIAN_FRONTEND=noninteractive apt-get update 2>&1 | tee -a "$log_file"',
        "  local status=${PIPESTATUS[0]}",
        '  if [[ "$status" -eq 0 ]]; then',
        '    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@" 2>&1 | tee -a "$log_file"',
        "    status=${PIPESTATUS[0]}",
        "  fi",
        "  apt-get clean >/dev/null 2>&1 || true",
        "  rm -rf /var/lib/apt/lists/* || true",
        "  set -e",
        '  return "$status"',
        "}",
        "",
        "install_talib_deb_for_freqtrade() {",
        f'  if [[ "{spec.repo.lower()}" != "freqtrade/freqtrade" ]]; then',
        "    return 0",
        "  fi",
        '  if [[ -f "/usr/include/ta-lib/ta_defs.h" || -f "/usr/local/include/ta-lib/ta_defs.h" || -f "/usr/include/ta_defs.h" || -f "/usr/local/include/ta_defs.h" ]]; then',
        '    echo "Info: TA-Lib headers already present; skipping deb install."',
        "    return 0",
        "  fi",
        '  local talib_python_version=""',
        "  local candidate_file",
        "  while IFS= read -r candidate_file; do",
        '    local req_line=""',
        '    req_line="$(grep -Eim1 "^[[:space:]]*TA-Lib([[:space:]]*[<>=!~]{1,2}[[:space:]]*[0-9][0-9A-Za-z._-]*)?" "$candidate_file" || true)"',
        '    if [[ -z "$req_line" ]]; then',
        "      continue",
        "    fi",
        '    talib_python_version="$(echo "$req_line" | sed -nE '"'"'s/.*[<>=!~]{1,2}[[:space:]]*([0-9]+(\\.[0-9]+){1,2}).*/\\1/p'"'"' | head -n 1)"',
        '    if [[ -n "$talib_python_version" ]]; then',
        "      break",
        "    fi",
        "  done < <(find . -type f \\( -name 'requirements*.txt' -o -name '*requirements*.txt' -o -name 'constraints*.txt' -o -name '*constraints*.txt' -o -name 'pyproject.toml' -o -name 'setup.cfg' -o -name 'setup.py' \\) | sort -u)",
        '  local talib_c_version="0.6.4"',
        '  if [[ "$talib_python_version" =~ ^0\\.(4|5)(\\.|$) ]]; then',
        '    talib_c_version="0.4.0"',
        "  fi",
        '  echo "Info: detected ta-lib-python version ${talib_python_version:-unknown}; selecting TA-Lib C ${talib_c_version}."',
        '  local deb_url="https://github.com/ta-lib/ta-lib/releases/download/v${talib_c_version}/ta-lib_${talib_c_version}_amd64.deb"',
        '  local deb_file="/tmp/ta-lib_${talib_c_version}_amd64.deb"',
        '  local src_dir="/tmp/ta-lib-src-${talib_c_version}"',
        '  local src_tar="/tmp/ta-lib-${talib_c_version}-src.tar.gz"',
        '  local install_log="/tmp/reviewbench-ta-lib-install.log"',
        '  echo "Info: installing TA-Lib from ${deb_url}"',
        "  set +e",
        '  curl -fsSL --retry 3 --retry-delay 2 --retry-connrefused -o "$deb_file" "$deb_url" 2>&1 | tee "$install_log"',
        "  local status=${PIPESTATUS[0]}",
        '  if [[ "$status" -eq 0 ]]; then',
        '    dpkg -i "$deb_file" 2>&1 | tee -a "$install_log"',
        "    status=${PIPESTATUS[0]}",
        "  fi",
        '  if [[ "$status" -ne 0 ]] && command -v apt-get >/dev/null 2>&1; then',
        '    DEBIAN_FRONTEND=noninteractive apt-get update 2>&1 | tee -a "$install_log"',
        '    DEBIAN_FRONTEND=noninteractive apt-get install -y -f 2>&1 | tee -a "$install_log"',
        '    dpkg -i "$deb_file" 2>&1 | tee -a "$install_log"',
        "    status=${PIPESTATUS[0]}",
        "  fi",
        '  if [[ "$status" -ne 0 && "$talib_c_version" == "0.4.0" ]]; then',
        '    echo "Warning: TA-Lib 0.4.0 deb install failed; building TA-Lib 0.4.0 from source." | tee -a "$install_log"',
        '    ensure_apt_packages "$install_log" file || true',
        "    rm -f \"$src_tar\"",
        "    rm -rf \"$src_dir\"",
        "    mkdir -p \"$src_dir\"",
        '    curl -fsSL --retry 3 --retry-delay 2 --retry-connrefused -o "$src_tar" "https://github.com/ta-lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz" 2>&1 | tee -a "$install_log"',
        "    status=${PIPESTATUS[0]}",
        '    if [[ "$status" -ne 0 ]]; then',
        '      curl -fsSL --retry 3 --retry-delay 2 --retry-connrefused -o "$src_tar" "https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz" 2>&1 | tee -a "$install_log"',
        "      status=${PIPESTATUS[0]}",
        "    fi",
        '    if [[ "$status" -eq 0 ]]; then',
        '      tar -xzf "$src_tar" -C "$src_dir" --strip-components=1 2>&1 | tee -a "$install_log"',
        "      status=${PIPESTATUS[0]}",
        "    fi",
        '    if [[ "$status" -eq 0 ]]; then',
        '      (cd "$src_dir" && ./configure --prefix=/usr && MAKEFLAGS="-j1" make -j1 && make install) 2>&1 | tee -a "$install_log"',
        "      status=${PIPESTATUS[0]}",
        "      ldconfig >/dev/null 2>&1 || true",
        "    fi",
        '    rm -f "$src_tar" || true',
        '    rm -rf "$src_dir" || true',
        "  fi",
        '  rm -f "$deb_file" || true',
        "  set -e",
        '  if [[ "$status" -ne 0 ]]; then',
        '    echo "Error: failed to install TA-Lib deb package." >&2',
        '    tail -n 80 "$install_log" >&2 || true',
        '    return "$status"',
        "  fi",
        '  if [[ ! -f "/usr/include/ta-lib/ta_defs.h" && ! -f "/usr/local/include/ta-lib/ta_defs.h" && ! -f "/usr/include/ta_defs.h" && ! -f "/usr/local/include/ta_defs.h" ]]; then',
        '    echo "Error: TA-Lib headers still missing after deb install." >&2',
        "    return 1",
        "  fi",
        "}",
        "",
        "normalize_requirement_file() {",
        '  local file_path="$1"',
        '  if [[ ! -f "$file_path" ]]; then',
        "    return 0",
        "  fi",
        '  sed -E -i "s#git\\+git://github.com/#git+https://github.com/#g" "$file_path"',
        '  sed -E -i "s#git://github.com/#https://github.com/#g" "$file_path"',
        '  sed -E -i "s|^([[:space:]]*)sklearn([[:space:]]*)(#.*)?\\$|\\1scikit-learn\\3|" "$file_path"',
        '  sed -E -i "s@^([[:space:]]*)(-e|--editable)[[:space:]]+\\.[[:space:]]*(#.*)?\\$@\\1# reviewbench-skip-local-editable\\3@" "$file_path"',
        '  sed -E -i "s|^([[:space:]]*)\\.[[:space:]]*(#.*)?\\$|# reviewbench-skip-local-path\\2|" "$file_path"',
        "}",
        "",
        "prepare_requirements_file() {",
        '  local src_file="$1"',
        '  local dst_file="$2"',
        '  cp "$src_file" "$dst_file"',
        '  normalize_requirement_file "$dst_file"',
        "}",
        "",
        "iter_requirement_files() {",
        '  local primary_file="$1"',
        '  if [[ -f "$primary_file" ]]; then',
        '    echo "$primary_file"',
        "  fi",
        '  while IFS= read -r candidate; do',
        '    if [[ "$candidate" != "$primary_file" ]]; then',
        '      echo "$candidate"',
        "    fi",
        '  done < <(find . -type f \\( -iname "*requirement*.txt" -o -iname "*requirements*.txt" -o -iname "*constraint*.txt" -o -iname "*constraints*.txt" \\) | sort -u)',
        "}",
        "",
        "normalize_repo_requirement_files() {",
        '  local primary_file="$1"',
        '  while IFS= read -r req_candidate; do',
        '    normalize_requirement_file "$req_candidate"',
        '  done < <(iter_requirement_files "$primary_file")',
        "}",
        "",
        "apply_fix_to_requirement_files() {",
        '  local primary_file="$1"',
        '  local sed_expr="$2"',
        '  while IFS= read -r req_candidate; do',
        '    sed -E -i "$sed_expr" "$req_candidate"',
        '  done < <(iter_requirement_files "$primary_file")',
        "}",
        "",
        "apply_known_requirement_fixes() {",
        '  local req_file="$1"',
        '  local missing_pkg="$2"',
        '  case "$missing_pkg" in',
        '    "braintree~=4.8.0")',
        '      apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)braintree~=4\\.8\\.0([[:space:]]*)(#.*)?\\$|\\1braintree>=4.17.1,<5\\3|"',
        "      ;;",
        '    "RestrictedPython~=5.1")',
        '      apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)RestrictedPython~=5\\.1([[:space:]]*)(#.*)?\\$|\\1RestrictedPython>=5.2,<8\\3|"',
        "      ;;",
        '    "pandas-ta=="*)',
        '      apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)pandas-ta([<>=!~].*)?([[:space:]]*)(#.*)?\\$|# reviewbench-skip-pandas-ta\\4|"',
        "      ;;",
        '    "pandas-ta")',
        '      apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)pandas-ta([<>=!~].*)?([[:space:]]*)(#.*)?\\$|# reviewbench-skip-pandas-ta\\4|"',
        "      ;;",
        '    "ccxt=="*)',
        '      apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)ccxt==[^[:space:]#]+([[:space:]]*)(#.*)?\\$|\\1ccxt\\3|"',
        "      ;;",
        '    "numpy=="*)',
        '      if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" == "1" ]]; then',
        '        apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)numpy==[^[:space:]#]+([[:space:]]*)(#.*)?\\$|\\1numpy<=1.21.6\\3|"',
        '      else',
        '        apply_fix_to_requirement_files "$req_file" "s@^([[:space:]]*)numpy==1\\.(1[0-9]|2[0-3])\\.[0-9]+([^#]*)(#.*)?\\$@\\1numpy>=1.24,<2\\3\\4@"',
        "      fi",
        "      ;;",
        '    "grpcio=="*)',
        '      if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" != "1" ]]; then',
        '        apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)grpcio==1\\.[0-4][0-9]\\.[0-9]+([^#]*)(#.*)?\\$|\\1grpcio>=1.56,<2\\2\\3|"',
        "      fi",
        "      ;;",
        '    "xmlsec=="*)',
        '      if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" != "1" ]]; then',
        '        apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)xmlsec==1\\.3\\.(1[0-3])([^#]*)(#.*)?\\$|\\1xmlsec>=1.3.14,<2\\3\\4|"',
        "      fi",
        "      ;;",
        '    "yarl=="*)',
        '      if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" != "1" ]]; then',
        '        apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)yarl==1\\.[0-8]\\.[0-9]+([^#]*)(#.*)?\\$|\\1yarl>=1.9,<2\\2\\3|"',
        "      fi",
        "      ;;",
        '    "aiohttp=="*)',
        '      if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" != "1" ]]; then',
        '        apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)aiohttp==3\\.[0-8]\\.[0-9]+([^#]*)(#.*)?\\$|\\1aiohttp>=3.9,<4\\2\\3|"',
        "      fi",
        "      ;;",
        '    "frozenlist=="*)',
        '      if [[ "$REVIEWBENCH_IS_PY37_OR_LOWER" != "1" ]]; then',
        '        apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)frozenlist==1\\.[0-3]\\.[0-9]+([^#]*)(#.*)?\\$|\\1frozenlist>=1.4,<2\\2\\3|"',
        "      fi",
        "      ;;",
        '    "opentelemetry-exporter-prometheus>=1.12.0rc1")',
        '      apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)opentelemetry-exporter-prometheus>=1\\.12\\.0rc1([[:space:]]*)(#.*)?\\$|\\1opentelemetry-exporter-prometheus>=0.54b1\\3|"',
        "      ;;",
        '    "tensorflow==1.13.1")',
        '      apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)tensorflow==1\\.13\\.1([[:space:]]*)(#.*)?\\$|# reviewbench-skip-tensorflow-1.13.1\\3|"',
        "      ;;",
        '    "clickhouse-driver=="*)',
        '      if [[ "$missing_pkg" =~ ^clickhouse-driver==0\\.2\\.[01]$ ]]; then',
        '        apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)clickhouse-driver==0\\.2\\.[01]([^#]*)(#.*)?\\$|\\1clickhouse-driver>=0.2.9,<0.3\\2\\3|"',
        "      fi",
        "      ;;",
        '    "lxml=="*)',
        '      if [[ "$missing_pkg" =~ ^lxml==4\\.(6|7|8)\\.[0-9]+$ ]]; then',
        '        apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)lxml==4\\.[678]\\.[0-9]+([^#]*)(#.*)?\\$|\\1lxml>=4.9.3,<5\\2\\3|"',
        "      fi",
        "      ;;",
        '    "aiohasupervisor=="*)',
        '      apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)aiohasupervisor==[^[:space:]#]+([[:space:]]*)(#.*)?\\$|\\1aiohasupervisor\\3|"',
        "      ;;",
        '    "hass-nabucasa=="*)',
        '      apply_fix_to_requirement_files "$req_file" "s|^([[:space:]]*)hass-nabucasa==[^[:space:]#]+([[:space:]]*)(#.*)?\\$|\\1hass-nabucasa\\3|"',
        "      ;;",
        "  esac",
        "}",
        "",
        "run_pip_requirements_install() {",
        '  local req_file="$1"',
        '  local log_file="$2"',
        '  local mode="$3"',
        "  shift 3",
        '  run_pip_install_with_mode "$mode" "$log_file" "$@" -r "$req_file"',
        "}",
        "",
        "run_repo_pip_install() {",
        '  local mode="$1"',
        '  local log_file="$2"',
        "  shift 2",
        '  run_pip_install_with_mode "$mode" "$log_file" "$@"',
        "}",
        "",
        "ensure_numpy_installed() {",
        '  local pip_log="$1"',
        '  if "$REVIEWBENCH_PYTHON_BIN" -c "import numpy" >/dev/null 2>&1; then',
        "    return 0",
        "  fi",
        '  echo "Info: numpy is missing; installing numpy before build." | tee -a "$pip_log"',
        '  run_repo_pip_install "strict" "$pip_log" "numpy<2" \\',
        '    || run_repo_pip_install "none" "$pip_log" "numpy<2"',
        "}",
        "",
        "ninja_meets_minimum_version() {",
        '  local min_version="${1:-1.8.2}"',
        "  if ! command -v ninja >/dev/null 2>&1; then",
        "    return 1",
        "  fi",
        '  local current_version=""',
        '  current_version="$(ninja --version 2>/dev/null || true)"',
        '  if [[ -z "$current_version" ]]; then',
        "    return 1",
        "  fi",
        '  "$REVIEWBENCH_PYTHON_BIN" - "$current_version" "$min_version" <<'"'"'PY'"'"' >/dev/null 2>&1',
        "import re",
        "import sys",
        "",
        "def parse(v: str):",
        "    m = re.search(r'(\\d+)\\.(\\d+)\\.(\\d+)', v)",
        "    if not m:",
        "        return None",
        "    return tuple(int(x) for x in m.groups())",
        "",
        "cur = parse(sys.argv[1])",
        "min_v = parse(sys.argv[2])",
        "if cur is None or min_v is None:",
        "    raise SystemExit(1)",
        "raise SystemExit(0 if cur >= min_v else 1)",
        "PY",
        "}",
        "",
        "install_scipy_build_prerequisites() {",
        '  local pip_log="$1"',
        f'  if [[ "{spec.repo.lower()}" != "scipy/scipy" ]]; then',
        "    return 0",
        "  fi",
        '  echo "Info: ensuring scipy build prerequisites (numpy, Cython, meson, meson-python, ninja, pybind11, pythran)." | tee -a "$pip_log"',
        '  ensure_numpy_installed "$pip_log" || true',
        '  ensure_apt_packages "$pip_log" ninja-build || true',
        '  run_repo_pip_install "strict" "$pip_log" Cython meson meson-python ninja pybind11 pythran \\',
        '    || run_repo_pip_install "none" "$pip_log" Cython meson meson-python ninja pybind11 pythran || true',
        '  if ! ninja_meets_minimum_version "1.8.2"; then',
        '    echo "Info: installing newer ninja (>=1.8.2) for scipy build." | tee -a "$pip_log"',
        '    run_repo_pip_install "strict" "$pip_log" '"'"'ninja>=1.8.2'"'"' \\',
        '      || run_repo_pip_install "none" "$pip_log" '"'"'ninja>=1.8.2'"'"' || true',
        "  fi",
        '  if command -v ninja >/dev/null 2>&1; then',
        '    echo "Info: ninja version: $(ninja --version 2>/dev/null || echo unknown)" | tee -a "$pip_log"',
        "  fi",
        "}",
        "",
        "patch_pandas_legacy_pytz_requirement() {",
        '  local pip_log="$1"',
        "  local patched=0",
        "  while IFS= read -r candidate; do",
        '    if grep -Eq "pytz[[:space:]]*>=[[:space:]]*2011k" "$candidate"; then',
        '      sed -E -i "s/pytz[[:space:]]*>=[[:space:]]*2011k/pytz>=2014.1/g" "$candidate"',
        "      patched=1",
        "    fi",
        '  done < <(find . -type f \\( -name "setup.py" -o -name "setup.cfg" -o -name "pyproject.toml" -o -iname "*requirement*.txt" -o -iname "*requirements*.txt" -o -iname "*constraint*.txt" -o -iname "*constraints*.txt" \\) | sort -u)',
        '  if [[ "$patched" -eq 1 ]]; then',
        '    echo "Info: patched legacy requirement spec pytz>=2011k -> pytz>=2014.1." | tee -a "$pip_log"',
        "  fi",
        "}",
        "",
        "try_repo_install_with_mode() {",
        '  local mode="$1"',
        '  local pip_log="$2"',
        '  run_repo_pip_install "$mode" "$pip_log" -e ".[test]" \\',
        '    || run_repo_pip_install "$mode" "$pip_log" -e ".[tests]" \\',
        '    || run_repo_pip_install "$mode" "$pip_log" -e ".[dev]" \\',
        '    || run_repo_pip_install "$mode" "$pip_log" -e .',
        "}",
        "",
        "try_repo_install_with_mode_no_build_isolation() {",
        '  local mode="$1"',
        '  local pip_log="$2"',
        '  run_repo_pip_install "$mode" "$pip_log" --no-build-isolation -e ".[test]" \\',
        '    || run_repo_pip_install "$mode" "$pip_log" --no-build-isolation -e ".[tests]" \\',
        '    || run_repo_pip_install "$mode" "$pip_log" --no-build-isolation -e ".[dev]" \\',
        '    || run_repo_pip_install "$mode" "$pip_log" --no-build-isolation -e .',
        "}",
        "",
        "try_repo_install_with_mode_for_repo() {",
        '  local mode="$1"',
        '  local pip_log="$2"',
        f'  if [[ "{spec.repo.lower()}" == "pandas-dev/pandas" || "{spec.repo.lower()}" == "scipy/scipy" ]]; then',
        '    try_repo_install_with_mode_no_build_isolation "$mode" "$pip_log"',
        "  else",
        '    try_repo_install_with_mode "$mode" "$pip_log"',
        "  fi",
        "}",
        "",
        "install_required_requirements() {",
        '  local req_file="$1"',
        '  if [[ ! -f "$req_file" ]]; then',
        "    return 0",
        "  fi",
        '  echo "Installing requirements from ${req_file}"',
        "  local req_basename req_dir",
        '  req_basename="$(basename "$req_file")"',
        '  req_dir="$(dirname "$req_file")"',
        '  local patched_req="${req_dir}/.reviewbench-${req_basename}.patched.txt"',
        '  local pip_log="/tmp/reviewbench-${req_basename}.pip.log"',
        '  local constraint_mode="strict"',
        "",
        '  normalize_repo_requirement_files "$req_file"',
        '  prepare_requirements_file "$req_file" "$patched_req"',
        f'  if [[ "{spec.repo.lower()}" == "posthog/posthog" ]]; then',
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)clickhouse-driver==0\\.2\\.[01]([^#]*)(#.*)?\\$|\\1clickhouse-driver>=0.2.9,<0.3\\2\\3|"',
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)lxml==4\\.[0-8]\\.[0-9]+([^#]*)(#.*)?\\$|\\1lxml>=4.9.3,<5\\2\\3|"',
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)grpcio==1\\.[0-4][0-9]\\.[0-9]+([^#]*)(#.*)?\\$|\\1grpcio>=1.56,<2\\2\\3|"',
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)xmlsec==1\\.3\\.(1[0-3])([^#]*)(#.*)?\\$|\\1xmlsec>=1.3.14,<2\\3\\4|"',
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)yarl==1\\.[0-8]\\.[0-9]+([^#]*)(#.*)?\\$|\\1yarl>=1.9,<2\\2\\3|"',
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)aiohttp==3\\.[0-8]\\.[0-9]+([^#]*)(#.*)?\\$|\\1aiohttp>=3.9,<4\\2\\3|"',
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)frozenlist==1\\.[0-3]\\.[0-9]+([^#]*)(#.*)?\\$|\\1frozenlist>=1.4,<2\\2\\3|"',
        '    apply_fix_to_requirement_files "$patched_req" "s@^([[:space:]]*)numpy==1\\.(1[0-9]|2[0-3])\\.[0-9]+([^#]*)(#.*)?\\$@\\1numpy>=1.24,<2\\3\\4@"',
        "  fi",
        "",
        '  if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
            "    return 0",
        "  fi",
        "",
        '  if grep -q "sklearn'"'"' PyPI package is deprecated" "$pip_log"; then',
        "    export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "cannot import name '"'"'Feature'"'"' from '"'"'setuptools'"'"'" "$pip_log"; then',
        '    if [[ "$REVIEWBENCH_CAN_USE_LEGACY_SETUPTOOLS" == "1" ]]; then',
        '      upgrade_python_packages "$pip_log" '"'"'setuptools<46'"'"' || true',
        '      constraint_mode="legacy"',
        '      if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "        return 0",
        "      fi",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none"; then',
        "      constraint_mode=\"none\"",
        "      return 0",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode" --no-build-isolation; then',
        "      return 0",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none" --no-build-isolation; then',
        "      constraint_mode=\"none\"",
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "setuptools.extern.six" "$pip_log"; then',
        '    if [[ "$REVIEWBENCH_CAN_USE_LEGACY_SETUPTOOLS" == "1" ]]; then',
        '      upgrade_python_packages "$pip_log" '"'"'setuptools<58'"'"' || true',
        '      constraint_mode="legacy"',
        '      if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "        return 0",
        "      fi",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none"; then',
        "      constraint_mode=\"none\"",
        "      return 0",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode" --no-build-isolation; then',
        "      return 0",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none" --no-build-isolation; then',
        "      constraint_mode=\"none\"",
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "error in pandas setup command" "$pip_log"; then',
        '    if [[ "$REVIEWBENCH_CAN_USE_LEGACY_SETUPTOOLS" == "1" ]]; then',
        '      upgrade_python_packages "$pip_log" '"'"'setuptools<58'"'"' || true',
        '      constraint_mode="legacy"',
        '      if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "        return 0",
        "      fi",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none"; then',
        "      constraint_mode=\"none\"",
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -Eq \"pkgutil\\.ImpImporter|has no attribute 'ImpImporter'\" \"$pip_log\"; then",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none"; then',
        "      constraint_mode=\"none\"",
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "Cannot install setuptools>=" "$pip_log"; then',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none"; then',
        "      constraint_mode=\"none\"",
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "The user requested (constraint) setuptools<" "$pip_log"; then',
        '    constraint_mode="none"',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "Could not build wheels for psycopg2" "$pip_log"; then',
        '    sed -E -i "s|^([[:space:]]*)psycopg2([<>=!~].*)?([[:space:]]*)(#.*)?\\$|\\1psycopg2-binary\\2\\4|" "$patched_req"',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -Eq \"couldn't run 'pg_config'|No such file or directory: 'pg_config'\" \"$pip_log\"; then",
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)psycopg\\[c,pool\\]==([^[:space:]#]+)([[:space:]]*)(#.*)?\\$|\\1psycopg[binary,pool]==\\2\\3|"',
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)psycopg-c([<>=!~].*)?([[:space:]]*)(#.*)?\\$|# reviewbench-skip-psycopg-c\\4|"',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "Could not build wheels for TA-Lib" "$pip_log"; then',
        '    apply_fix_to_requirement_files "$patched_req" "s|^([[:space:]]*)TA-Lib([<>=!~].*)?([[:space:]]*)(#.*)?\\$|# reviewbench-skip-ta-lib\\4|"',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -q \"ModuleNotFoundError: No module named 'numpy'\" \"$pip_log\" && grep -Eiq \"ta-lib|TA-Lib\" \"$pip_log\"; then",
        '    ensure_numpy_installed "$pip_log" || true',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode" --no-build-isolation; then',
        "      return 0",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none" --no-build-isolation; then',
        '      constraint_mode="none"',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -q \"No module named 'pkg_resources'\" \"$pip_log\"; then",
        '    upgrade_python_packages "$pip_log" '"'"'setuptools<82'"'"' || true',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode" --no-build-isolation; then',
        "      return 0",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none" --no-build-isolation; then',
        '      constraint_mode="none"',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -Eiq "Please make sure the libxml2 and libxslt development packages|libxml2 and libxslt development packages" "$pip_log"; then',
        '    ensure_apt_packages "$pip_log" libxml2-dev libxslt1-dev zlib1g-dev || true',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -Eiq "xmlsec1 is not installed or not in path|error: xmlsec1|xmlsec1.*not found" "$pip_log"; then',
        '    ensure_apt_packages "$pip_log" xmlsec1 libxml2-dev libxmlsec1-dev libxmlsec1-openssl pkg-config || true',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "      return 0",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none"; then',
        '      constraint_mode="none"',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  while IFS= read -r failed_line; do",
        '    failed_pkg="${failed_line#*\\`}"',
        '    failed_pkg="${failed_pkg%\\`}"',
        '    apply_known_requirement_fixes "$patched_req" "$failed_pkg"',
        "  done < <(grep -oE 'Failed to build `[^`]+`' \"$pip_log\" | sort -u)",
        "",
        '  if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "    return 0",
        "  fi",
        "",
        "  while IFS= read -r missing_line; do",
        '    missing_pkg="${missing_line##*for }"',
        '    apply_known_requirement_fixes "$patched_req" "$missing_pkg"',
        '  done < <(grep -oE "No matching distribution found for [^ ]+" "$pip_log" | sort -u)',
        "",
        "  while IFS= read -r missing_line; do",
        '    missing_pkg="${missing_line##*requirement }"',
        '    apply_known_requirement_fixes "$patched_req" "$missing_pkg"',
        '  done < <(grep -oE "Could not find a version that satisfies the requirement [^ ]+" "$pip_log" | sort -u)',
        "",
        "  while IFS= read -r missing_line; do",
        '    missing_pkg="${missing_line##*of }"',
        '    apply_known_requirement_fixes "$patched_req" "$missing_pkg"',
        '  done < <(grep -oE "there is no version of [^ ]+" "$pip_log" | sort -u)',
        "",
        '  if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
    "    return 0",
  "  fi",
        "",
        "  if grep -q \"No module named 'pkg_resources'\" \"$pip_log\"; then",
        '    upgrade_python_packages "$pip_log" '"'"'setuptools<82'"'"' || true',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode" --no-build-isolation; then',
        "      return 0",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none" --no-build-isolation; then',
        '      constraint_mode="none"',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "AttributeError: cython_sources" "$pip_log"; then',
        '    upgrade_python_packages "$pip_log" '"'"'Cython<3'"'"' || true',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "cython_legacy"; then',
        "      return 0",
        "    fi",
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "cython_legacy" --no-build-isolation; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "ResolutionImpossible" "$pip_log"; then',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "$constraint_mode"; then',
        "      return 0",
        "    fi",
        '    if [[ "$constraint_mode" != "none" ]]; then',
        '      if run_pip_requirements_install "$patched_req" "$pip_log" "none"; then',
        "        return 0",
        "      fi",
        "    fi",
        "  fi",
        "",
        '  if [[ "$constraint_mode" != "none" ]]; then',
        '    if run_pip_requirements_install "$patched_req" "$pip_log" "none"; then',
        "    return 0",
        "  fi",
        "  fi",
        "",
        '  echo "Required install failed: ${req_file}" >&2',
        '  tail -n 60 "$pip_log" >&2 || true',
        "  return 1",
        "}",
        "",
        "install_optional_requirements() {",
        '  local req_file="$1"',
        '  if [[ -f "$req_file" ]]; then',
        '    echo "Installing optional requirements from ${req_file}"',
        "    local req_basename req_dir",
        '    req_basename="$(basename "$req_file")"',
        '    req_dir="$(dirname "$req_file")"',
        '    local patched_req="${req_dir}/.reviewbench-${req_basename}.patched.txt"',
        '    local pip_log="/tmp/reviewbench-${req_basename}.optional.log"',
        '    normalize_repo_requirement_files "$req_file"',
        '    prepare_requirements_file "$req_file" "$patched_req"',
        '    run_pip_requirements_install "$patched_req" "$pip_log" "strict" || echo "Optional install failed: ${req_file}"',
        "  fi",
        "}",
        "",
        "install_repo_package() {",
        '  if [[ ! -f "pyproject.toml" && ! -f "setup.py" && ! -f "setup.cfg" ]]; then',
        "    return 0",
        "  fi",
        '  local pip_log="/tmp/reviewbench-repo-install.log"',
        "",
    ]

    lines.extend(
        [
        f'  if [[ "{spec.repo.lower()}" == "numpy/numpy" && -f "setup.py" ]]; then',
        '    if ! "$REVIEWBENCH_PYTHON_BIN" -c "import Cython" >/dev/null 2>&1; then',
        '      echo "Info: Cython is missing; installing Cython<3 before numpy setup.py build_ext -i." | tee -a "$pip_log"',
        '      upgrade_python_packages "$pip_log" '"'"'Cython<3'"'"' || run_repo_pip_install "none" "$pip_log" '"'"'Cython<3'"'"' || true',
        "    fi",
        '    echo "Info: numpy setup.py detected; running python setup.py build_ext -i as first choice." | tee -a "$pip_log"',
        "    set +e",
        '    python setup.py build_ext -i -j 4 2>&1 | tee -a "$pip_log"',
        "    local build_ext_status=${PIPESTATUS[0]}",
        "    set -e",
        '    if [[ "$build_ext_status" -eq 0 ]]; then',
        "      return 0",
        "    fi",
        '    echo "Warning: setup.py build_ext -i failed (exit ${build_ext_status}); not using standard install flow when setup.py exists." | tee -a "$pip_log"',
        '    echo "Info: Cannot install on Python version; requesting python-version fallback after numpy setup.py build_ext -i failure." | tee -a "$pip_log"',
        "    return 1",
        "  fi",
        "",
        f'  if [[ "{spec.repo.lower()}" == "pandas-dev/pandas" ]]; then',
        '    patch_pandas_legacy_pytz_requirement "$pip_log"',
        '    run_repo_pip_install "strict" "$pip_log" '"'"'Cython<3'"'"' meson meson-python || true',
        '    if [[ -f "requirements-dev.txt" ]]; then',
        '      run_pip_requirements_install "requirements-dev.txt" "$pip_log" "strict" || true',
        "    fi",
        '    ensure_numpy_installed "$pip_log" || true',
        '    if try_repo_install_with_mode_for_repo "strict" "$pip_log"; then',
        "      return 0",
        "    fi",
        f'  elif [[ "{spec.repo.lower()}" == "scipy/scipy" ]]; then',
        '    install_scipy_build_prerequisites "$pip_log" || true',
        '    if try_repo_install_with_mode_for_repo "strict" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  else",
        '    if try_repo_install_with_mode_for_repo "strict" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        f'  if [[ "{spec.repo.lower()}" == "pandas-dev/pandas" ]]; then',
        '    if grep -q "No module named '"'"'pkg_resources'"'"'" "$pip_log"; then',
        '      upgrade_python_packages "$pip_log" '"'"'setuptools<81'"'"' || true',
        '      if run_repo_pip_install "none" "$pip_log" --no-build-isolation -e .; then',
        "        return 0",
        "      fi",
        "    fi",
        "",
        '    if grep -q "No module named '"'"'distutils.msvccompiler'"'"'" "$pip_log"; then',
        "      export SETUPTOOLS_USE_DISTUTILS=local",
        '      upgrade_python_packages "$pip_log" '"'"'setuptools<70'"'"' || true',
        '      if run_repo_pip_install "none" "$pip_log" --no-build-isolation -e .; then',
        "        return 0",
        "      fi",
        "    fi",
        '    if run_repo_pip_install "none" "$pip_log" --no-build-isolation -e .; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "Submodule .* not initialized" "$pip_log"; then',
        "    git submodule sync --recursive || true",
        "    git submodule update --init --recursive || true",
        '    if try_repo_install_with_mode_for_repo "strict" "$pip_log" || try_repo_install_with_mode_for_repo "none" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "The user requested (constraint) setuptools<" "$pip_log"; then',
        '    if try_repo_install_with_mode_for_repo "none" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        f"  if [[ \"{spec.repo.lower()}\" == \"pandas-dev/pandas\" ]] \\",
        "    && grep -Eq \"pytz>=2011k|ParserSyntaxError: Expected end or semicolon \\(after version specifier\\)|InvalidRequirement: Expected end or semicolon \\(after version specifier\\)|Failed to parse metadata from built wheel\" \"$pip_log\"; then",
        '    patch_pandas_legacy_pytz_requirement "$pip_log"',
        '    if try_repo_install_with_mode_for_repo "none" "$pip_log" || try_repo_install_with_mode_for_repo "strict" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        f'  if [[ "{spec.repo.lower()}" == "pandas-dev/pandas" ]] \\',
        "    && grep -Eq \"No module named 'numpy'|KeyError: 'numpy'\" \"$pip_log\"; then",
        '    if ensure_numpy_installed "$pip_log"; then',
        '      if try_repo_install_with_mode_for_repo "strict" "$pip_log" || try_repo_install_with_mode_for_repo "none" "$pip_log"; then',
        "        return 0",
        "      fi",
        "    fi",
        "  fi",
        "",
        "  if grep -Eq \"pkgutil\\.ImpImporter|has no attribute 'ImpImporter'\" \"$pip_log\"; then",
        '    if try_repo_install_with_mode_for_repo "none" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "setuptools.extern.six" "$pip_log"; then',
        '    if [[ "$REVIEWBENCH_CAN_USE_LEGACY_SETUPTOOLS" == "1" ]]; then',
        '      upgrade_python_packages "$pip_log" '"'"'setuptools<58'"'"' || true',
        '      if try_repo_install_with_mode_for_repo "legacy" "$pip_log"; then',
        "        return 0",
        "      fi",
        "    fi",
        '    if try_repo_install_with_mode_for_repo "none" "$pip_log"; then',
        "      return 0",
        "    fi",
        '    if run_repo_pip_install "none" "$pip_log" --no-build-isolation -e .; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "AttributeError: cython_sources" "$pip_log"; then',
        '    upgrade_python_packages "$pip_log" '"'"'Cython<3'"'"' || true',
        '    if try_repo_install_with_mode_for_repo "cython_legacy" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -Eq \"No module named 'Cython'|No such file or directory: 'cython'|Cython either isn't installed or it failed|Running cythonize failed|Program 'cython' not found or not executable|Program cython found: NO|cython: not found\" \"$pip_log\"; then",
        '    echo "Info: Cython is missing; installing Cython<3 and tempita and retrying." | tee -a "$pip_log"',
        '    upgrade_python_packages "$pip_log" '"'"'Cython<3'"'"' tempita || true',
        '    if try_repo_install_with_mode_for_repo "cython_legacy" "$pip_log" || try_repo_install_with_mode_for_repo "none" "$pip_log" || try_repo_install_with_mode_for_repo "strict" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -Eq \"No module named 'tempita'\" \"$pip_log\"; then",
        '    echo "Info: tempita is missing; installing tempita and retrying." | tee -a "$pip_log"',
        '    upgrade_python_packages "$pip_log" '"'"'Cython<3'"'"' tempita || true',
        '    if try_repo_install_with_mode_for_repo "cython_legacy" "$pip_log" || try_repo_install_with_mode_no_build_isolation "none" "$pip_log" || try_repo_install_with_mode_for_repo "none" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -Eq \"Cython-generated file .* not found\" \"$pip_log\"; then",
        '    upgrade_python_packages "$pip_log" '"'"'Cython<3'"'"' || true',
        '    if run_repo_pip_install "cython_legacy" "$pip_log" --no-build-isolation -e .; then',
        "      return 0",
        "    fi",
        '    if run_repo_pip_install "none" "$pip_log" --no-build-isolation -e .; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -Eq \"numpy/NPY_C_CONTIGUOUS\\.pxd' not found|numpy/NPY_F_CONTIGUOUS\\.pxd' not found\" \"$pip_log\"; then",
        '    upgrade_python_packages "$pip_log" '"'"'numpy<2'"'"' '"'"'Cython<3'"'"' || true',
        '    if run_repo_pip_install "none" "$pip_log" --no-build-isolation -e .; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -q \"No module named 'distutils.msvccompiler'\" \"$pip_log\"; then",
        '    echo "Info: distutils.msvccompiler is missing; pinning setuptools<70 and retrying without build isolation." | tee -a "$pip_log"',
        "    export SETUPTOOLS_USE_DISTUTILS=local",
        '    upgrade_python_packages "$pip_log" '"'"'setuptools<70'"'"' || true',
        '    if try_repo_install_with_mode_no_build_isolation "none" "$pip_log" || try_repo_install_with_mode_no_build_isolation "strict" "$pip_log" || try_repo_install_with_mode_for_repo "none" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -Eq \"Program 'swig' not found or not executable|Program swig found: NO|No such file or directory: 'swig'|swig: not found\" \"$pip_log\"; then",
        '    if ensure_apt_packages "$pip_log" swig; then',
        '      if try_repo_install_with_mode_for_repo "strict" "$pip_log" || try_repo_install_with_mode_for_repo "none" "$pip_log"; then',
        "        return 0",
        "      fi",
        "    fi",
        "  fi",
        "",
        "  if grep -q \"pybind11-config\" \"$pip_log\"; then",
        '    echo "Info: pybind11-config is missing; installing pybind11 and retrying." | tee -a "$pip_log"',
        '    upgrade_python_packages "$pip_log" pybind11 || true',
        '    if try_repo_install_with_mode_no_build_isolation "none" "$pip_log" || try_repo_install_with_mode_for_repo "none" "$pip_log" || try_repo_install_with_mode_for_repo "strict" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        "  if grep -Eq \"setuptools\\.build_meta\\.build_editable|module 'setuptools\\.build_meta' has no attribute\" \"$pip_log\"; then",
        '    echo "Info: editable build backend is too old; upgrading pip/setuptools/wheel and retrying." | tee -a "$pip_log"',
        '    upgrade_python_packages "$pip_log" pip setuptools wheel || true',
        '    if try_repo_install_with_mode_no_build_isolation "none" "$pip_log" || try_repo_install_with_mode_for_repo "none" "$pip_log" || try_repo_install_with_mode_for_repo "strict" "$pip_log"; then',
        "      return 0",
        "    fi",
        "  fi",
        "",
        '  if grep -q "Multiple top-level packages discovered in a flat-layout" "$pip_log"; then',
        '    echo "Info: skipping repository package install (flat-layout package discovery)."',
        "    return 0",
        "  fi",
        "",
        '  if grep -Eq "is not installable|neither '"'"'setup.py'"'"' nor '"'"'pyproject.toml'"'"' found" "$pip_log"; then',
        '    echo "Info: repository root is not pip-installable; skipping package install."',
        "    return 0",
        "  fi",
        "",
        "  if grep -Eq \"Cython-generated file .* not found|doesn't match any files|CompileError:|error: command '/usr/bin/gcc' failed with exit code 1|Python version >= [0-9.]+ required|requires at least Python [0-9.]+\" \"$pip_log\"; then",
        '    echo "Info: skipping repository package install after build-time compatibility failure."',
        "    return 0",
        "  fi",
        "",
        f'  if [[ "{spec.repo.lower()}" == "scipy/scipy" ]]; then',
        '    install_scipy_build_prerequisites "$pip_log" || true',
        "  fi",
        "",
        '  if try_repo_install_with_mode_for_repo "strict" "$pip_log" \\',
        '    || try_repo_install_with_mode_for_repo "none" "$pip_log"; then',
        "    return 0",
        "  fi",
        "",
        '  echo "Warning: repository package install failed; continuing"',
        "}",
    ]
    )

    if pip_install_line:
        lines.extend(["", pip_install_line])

    lines.extend(
        [
            "",
            "install_talib_deb_for_freqtrade",
            "",
            "install_required_requirements requirements.txt",
            "install_optional_requirements requirements-dev.txt",
            "install_optional_requirements requirements_dev.txt",
            "install_optional_requirements dev-requirements.txt",
            "install_optional_requirements requirements/test.txt",
            "install_optional_requirements requirements/tests.txt",
            "",
            'if compgen -G "requirements/*.txt" > /dev/null; then',
            '  for req_file in requirements/*.txt; do',
            '    case "$req_file" in',
            "      requirements/test.txt|requirements/tests.txt)",
            "        continue",
            "        ;;",
            "    esac",
            '    install_optional_requirements "$req_file"',
            "  done",
            "fi",
            "",
            "install_repo_package",
        ]
    )

    return "\n".join(lines) + "\n"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _dedupe(items: tuple[str, ...]) -> tuple[str, ...]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _has_retryable_buildkit_frontend_error(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    return _BUILDKIT_FRONTEND_GRPC_ERROR in combined


def _write_build_log(
    log_path: Path,
    command: list[str],
    stdout: str,
    stderr: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"COMMAND: {' '.join(command)}",
        "",
        "===== STDOUT =====",
        stdout,
        "",
        "===== STDERR =====",
        stderr,
    ]
    log_path.write_text("\n".join(body))


def _get_repo_seed_lock(repo: str) -> threading.Lock:
    with _REPO_SEED_LOCKS_GUARD:
        lock = _REPO_SEED_LOCKS.get(repo)
        if lock is None:
            lock = threading.Lock()
            _REPO_SEED_LOCKS[repo] = lock
        return lock


def _prepare_cached_repo_seed(
    spec: PREnvironmentSpec,
    *,
    seed_dir: Path,
    cached_repos_dir: Path | None,
) -> bool:
    if seed_dir.exists():
        shutil.rmtree(seed_dir)

    if cached_repos_dir is None:
        seed_dir.mkdir(parents=True, exist_ok=True)
        return False

    cached_repo = cached_repos_dir / spec.repo.replace("/", "__")
    if not (cached_repo / ".git").exists():
        seed_dir.mkdir(parents=True, exist_ok=True)
        return False

    lock = _get_repo_seed_lock(spec.repo)
    with lock:
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    str(cached_repo),
                    str(seed_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Failed to seed context from cached repo %s: %s",
                cached_repo,
                exc.stderr.strip() if exc.stderr else exc,
            )
            if seed_dir.exists():
                shutil.rmtree(seed_dir)
            seed_dir.mkdir(parents=True, exist_ok=True)
            return False
