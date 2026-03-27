"""Repo cloning and environment management.

Handles cloning GitHub repos, checking out commits, creating virtual
environments with uv, and running commands within repos.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "repos"


def create_instance_workdir(
    source_repo: Path, instance_id: str, workdir_root: Path
) -> Path:
    """Create a lightweight per-instance working directory using git clone --shared.

    The --shared flag makes the clone reuse the source repo's git object store
    via alternates, so all commits (including fetched PR refs) are accessible
    by SHA without re-fetching. Only working tree files consume extra disk.

    Args:
        source_repo: Path to the cached source repo (already cloned and fetched).
        instance_id: Instance identifier (used to derive the workdir name).
        workdir_root: Parent directory for all instance workdirs.

    Returns:
        Path to the created workdir.
    """
    source_repo = Path(source_repo)
    workdir_root = Path(workdir_root)
    safe_id = instance_id.replace("/", "__")
    workdir = workdir_root / safe_id

    # Remove existing workdir if present (for retries)
    if workdir.exists():
        shutil.rmtree(workdir)

    workdir_root.mkdir(parents=True, exist_ok=True)
    logger.info("Creating instance workdir: %s", workdir)
    subprocess.run(
        ["git", "clone", "--shared", "--quiet", str(source_repo), str(workdir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return workdir


def clone_repo(repo: str, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    """Clone a GitHub repo if not already cached.

    Args:
        repo: GitHub repo in 'owner/name' format (e.g. 'tobymao/sqlglot').
        cache_dir: Directory to cache clones.

    Returns:
        Path to the cloned repo directory.
    """
    cache_dir = Path(cache_dir)
    repo_dir = cache_dir / repo.replace("/", "__")

    if repo_dir.exists() and (repo_dir / ".git").exists():
        logger.info("Using cached repo: %s", repo_dir)
        # Fetch latest to ensure all commits are available
        _run_git(repo_dir, ["fetch", "--all", "--quiet"])
        return repo_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    logger.info("Cloning %s → %s", url, repo_dir)
    subprocess.run(
        ["git", "clone", "--quiet", url, str(repo_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return repo_dir


def fetch_pr_commits(repo_path: str | Path, pull_number: int) -> None:
    """Fetch commits for a specific PR from GitHub.

    PR commits are often not reachable from any branch and need to be
    fetched explicitly via the pull/<n>/head ref.
    """
    repo_path = Path(repo_path)
    logger.info("Fetching PR #%d refs", pull_number)
    # Fetch both the PR head and the merge commit ref
    for ref in [f"pull/{pull_number}/head", f"pull/{pull_number}/merge"]:
        try:
            _run_git(repo_path, ["fetch", "origin", ref])
        except subprocess.CalledProcessError:
            logger.debug("Could not fetch %s (may not exist)", ref)


def checkout_commit(repo_path: str | Path, commit: str) -> None:
    """Checkout a specific commit in the repo.

    Cleans the working tree first to avoid conflicts.
    """
    repo_path = Path(repo_path)
    _run_git(repo_path, ["checkout", "--force", commit])
    _run_git(repo_path, ["clean", "-fd", "--quiet"])


def get_file_at_commit(
    repo_path: str | Path, commit: str, filepath: str
) -> str:
    """Get file contents at a specific commit without checking out.

    Uses `git show commit:filepath` to read file contents directly.

    Returns:
        File contents as a string, or empty string if file doesn't exist.
    """
    repo_path = Path(repo_path)
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{filepath}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        logger.warning(
            "File %s not found at commit %s", filepath, commit[:12]
        )
        return ""


def setup_venv(repo_path: str | Path) -> Path:
    """Create a Python virtual environment for a repo using uv.

    Creates venv at repo_path/.venv and installs the repo in editable mode.

    Returns:
        Path to the venv directory.
    """
    repo_path = Path(repo_path)
    venv_path = repo_path / ".venv"

    if not venv_path.exists():
        logger.info("Creating venv at %s", venv_path)
        subprocess.run(
            ["uv", "venv", str(venv_path)],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )

    # Install pytest first (needed for test execution)
    logger.info("Installing pytest in venv")
    pip_install(repo_path, venv_path, ["pytest"])

    # Install the repo in editable mode
    logger.info("Installing repo in editable mode")
    pip_install(repo_path, venv_path, ["-e", "."])

    return venv_path


def setup_node_env(repo_path: str | Path) -> None:
    """Install Node.js dependencies and Jest for a JS/TS repo.

    Runs ``npm install`` (if package.json exists) and ensures Jest is
    available for test execution.
    """
    repo_path = Path(repo_path)
    package_json = repo_path / "package.json"

    if package_json.exists():
        logger.info("Running npm install in %s", repo_path)
        subprocess.run(
            ["npm", "install", "--ignore-scripts"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

    # Ensure jest is available (install locally if not already present)
    node_modules_jest = repo_path / "node_modules" / ".bin" / "jest"
    if not node_modules_jest.exists():
        logger.info("Installing jest in %s", repo_path)
        subprocess.run(
            ["npm", "install", "--save-dev", "jest"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )


def pip_install(
    repo_path: str | Path,
    venv_path: str | Path,
    packages: list[str],
    no_deps: bool = False,
) -> tuple[int, str, str]:
    """Install packages into the repo's venv using uv pip.

    Returns:
        Tuple of (returncode, stdout, stderr).
    """
    repo_path = Path(repo_path)
    venv_path = Path(venv_path)

    cmd = ["uv", "pip", "install", "--python", str(venv_path / "bin" / "python")]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)

    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("pip install failed: %s", result.stderr[:500])
    return result.returncode, result.stdout, result.stderr


def run_in_repo(
    repo_path: str | Path,
    command: list[str],
    venv_path: str | Path | None = None,
    timeout: int = 120,
) -> tuple[int, str, str]:
    """Run a command in the repo directory, optionally with venv activated.

    Args:
        repo_path: Path to the repository.
        command: Command and arguments to run.
        venv_path: If provided, use the Python from this venv.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (returncode, stdout, stderr).
    """
    repo_path = Path(repo_path)
    env = os.environ.copy()

    if venv_path:
        venv_path = Path(venv_path)
        venv_bin = str(venv_path / "bin")
        env["VIRTUAL_ENV"] = str(venv_path)
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        # If the command starts with python/pytest, use the venv's python
        if command[0] in ("python", "python3"):
            command = [str(venv_path / "bin" / "python")] + command[1:]
        elif command[0] == "pytest":
            command = [str(venv_path / "bin" / "python"), "-m", "pytest"] + command[1:]

    try:
        result = subprocess.run(
            command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out after %ds: %s", timeout, " ".join(command))
        return -1, "", f"Command timed out after {timeout}s"


def _run_git(repo_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Run a git command in the given repo directory."""
    return subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
