"""Docker-based execution environment builders for PR-specific repos."""

from .container_runtime import (
    ContainerCommandResult,
    ContainerLogsResult,
    DockerContainerSession,
)
from .docker_env import BuildResult, build_image, prepare_build_context, spec_from_instance
from .python_version import PythonVersionResolution, resolve_python_version
from .specs import (
    IMPLEMENTED_LANGUAGES,
    KNOWN_LANGUAGES,
    PREnvironmentSpec,
    RepoEnvironmentSpec,
    infer_apt_packages_for_repo,
)

__all__ = [
    "BuildResult",
    "ContainerCommandResult",
    "ContainerLogsResult",
    "DockerContainerSession",
    "IMPLEMENTED_LANGUAGES",
    "KNOWN_LANGUAGES",
    "PythonVersionResolution",
    "PREnvironmentSpec",
    "RepoEnvironmentSpec",
    "infer_apt_packages_for_repo",
    "build_image",
    "prepare_build_context",
    "resolve_python_version",
    "spec_from_instance",
]
