"""Programmatic runtime utilities for Docker-backed execution environments."""

from __future__ import annotations

import logging
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContainerCommandResult:
    """Result of executing a command against a container."""

    container_id: str
    container_name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class ContainerLogsResult:
    """Result of reading logs for a container."""

    container_id: str
    container_name: str
    command: tuple[str, ...]
    stdout: str
    stderr: str


class DockerContainerSession:
    """Utility wrapper around a long-running docker container session.

    Typical use:
        with DockerContainerSession(image="reviewbench/foo:tag") as session:
            session.run_command(["python", "-V"])
            session.run_tests(pytest_args=["-q"])
            session.copy_to("local.txt", "/workspace/local.txt")
            logs = session.read_logs(tail=200)
    """

    def __init__(
        self,
        image: str,
        *,
        name: str | None = None,
        workdir: str = "/workspace",
        env: Mapping[str, str] | None = None,
        volumes: Sequence[str] | None = None,
        start_command: Sequence[str] = ("sleep", "infinity"),
        remove_on_exit: bool = True,
        auto_start: bool = False,
    ) -> None:
        self.image = image
        self._name = name or f"reviewbench-{uuid.uuid4().hex[:10]}"
        self.workdir = workdir
        self.env = dict(env or {})
        self.volumes = list(volumes or [])
        self.start_command = tuple(start_command)
        self.remove_on_exit = remove_on_exit

        self._container_id: str | None = None
        if auto_start:
            self.start()

    @property
    def container_name(self) -> str:
        return self._name

    @property
    def container_id(self) -> str:
        if self._container_id is None:
            raise RuntimeError("Container has not been started yet")
        return self._container_id

    def start(self) -> str:
        """Start the long-running container if it is not already started."""
        if self._container_id is not None:
            return self._container_id

        _ensure_docker_available()
        cmd = [
            "docker",
            "run",
            "--detach",
            "--network", "host",
            "--name",
            self._name,
            "--workdir",
            self.workdir,
        ]
        for key, value in sorted(self.env.items()):
            cmd.extend(["--env", f"{key}={value}"])
        for vol in self.volumes:
            cmd.extend(["-v", vol])
        cmd.append(self.image)
        cmd.extend(self.start_command)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container '{self._name}' from '{self.image}'. "
                f"stderr:\n{result.stderr[-3000:]}"
            )

        self._container_id = result.stdout.strip()
        logger.info("Started container %s (%s)", self._name, self._container_id[:12])
        return self._container_id

    def stop(self, *, timeout: int = 10) -> None:
        """Stop the running container."""
        if self._container_id is None:
            return

        cmd = ["docker", "stop", "--time", str(timeout), self._name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to stop container %s: %s", self._name, result.stderr.strip())
        self._container_id = None

    def remove(self, *, force: bool = True) -> None:
        """Remove the container."""
        cmd = ["docker", "rm"]
        if force:
            cmd.append("--force")
        cmd.append(self._name)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "Failed to remove container %s: %s", self._name, result.stderr.strip()
            )
        self._container_id = None

    def run_command(
        self,
        command: Sequence[str] | str,
        *,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 300,
        check: bool = False,
    ) -> ContainerCommandResult:
        """Execute a command in the running container using `docker exec`."""
        normalized_command = _normalize_exec_command(command)

        self.start()
        cmd = ["docker", "exec"]
        if workdir:
            cmd.extend(["--workdir", workdir])
        for key, value in sorted((env or {}).items()):
            cmd.extend(["--env", f"{key}={value}"])
        cmd.append(self._name)
        cmd.extend(normalized_command)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            exec_result = ContainerCommandResult(
                container_id=self.container_id,
                container_name=self._name,
                command=tuple(normalized_command),
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            exec_result = ContainerCommandResult(
                container_id=self.container_id,
                container_name=self._name,
                command=tuple(normalized_command),
                returncode=-1,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\nCommand timed out after {timeout}s",
                timed_out=True,
            )

        if check and exec_result.returncode != 0:
            raise RuntimeError(
                f"Command failed in container {self._name}: {' '.join(normalized_command)}\n"
                f"stdout:\n{exec_result.stdout[-3000:]}\n"
                f"stderr:\n{exec_result.stderr[-3000:]}"
            )
        return exec_result

    def run_tests(
        self,
        *,
        test_target: str = ".",
        pytest_args: Sequence[str] = (),
        timeout: int = 600,
        check: bool = False,
    ) -> ContainerCommandResult:
        """Run pytest in the container."""
        command = ["python", "-m", "pytest", test_target, *pytest_args]
        return self.run_command(command, timeout=timeout, check=check)

    def copy_to(self, local_path: str | Path, container_path: str) -> None:
        """Copy files or directories from host into the container."""
        self.start()
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Local path does not exist: {src}")

        cmd = ["docker", "cp", str(src), f"{self._name}:{container_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed copying '{src}' to container '{self._name}:{container_path}'. "
                f"stderr:\n{result.stderr[-3000:]}"
            )

    def copy_from(self, container_path: str, local_path: str | Path) -> None:
        """Copy files or directories from the container back to host."""
        self.start()
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["docker", "cp", f"{self._name}:{container_path}", str(dst)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed copying '{self._name}:{container_path}' to '{dst}'. "
                f"stderr:\n{result.stderr[-3000:]}"
            )

    def read_logs(
        self,
        *,
        tail: int | None = None,
        since: str | None = None,
        timestamps: bool = False,
    ) -> ContainerLogsResult:
        """Read logs from the container's primary process."""
        self.start()
        cmd = ["docker", "logs"]
        if timestamps:
            cmd.append("--timestamps")
        if since:
            cmd.extend(["--since", since])
        if tail is not None:
            cmd.extend(["--tail", str(tail)])
        cmd.append(self._name)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to read logs for container '{self._name}'. "
                f"stderr:\n{result.stderr[-3000:]}"
            )

        return ContainerLogsResult(
            container_id=self.container_id,
            container_name=self._name,
            command=tuple(cmd),
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def __enter__(self) -> DockerContainerSession:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.remove_on_exit:
            self.remove(force=True)
        else:
            self.stop()


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
        raise RuntimeError("docker daemon is not reachable. Make sure Docker is running.")


def get_docker_image_name(instance_id: str) -> str:
    """Derive Docker image name from instance ID.

    e.g. 'ansible__ansible-20646@f695114' => 'reviewbench/ansible__ansible-20646'
    """
    slug = instance_id.split("@")[0].lower()
    return f"reviewbench/{slug}"


def docker_image_exists(image: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def _normalize_exec_command(command: Sequence[str] | str) -> list[str]:
    if isinstance(command, str):
        shell_cmd = command.strip()
        if not shell_cmd:
            raise ValueError("command string must not be empty")
        return ["bash", "-lc", shell_cmd]

    normalized = [str(part) for part in command]
    if not normalized:
        raise ValueError("command must not be empty")
    return normalized
