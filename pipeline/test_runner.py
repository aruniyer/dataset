"""Test execution on before/after code versions.

Runs generated tests against the pre-review and post-review code to verify
the expected fail-then-pass pattern.  Supports multiple languages by
dispatching to the appropriate test runner (pytest, jest, go test, etc.).
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from execution.container_runtime import DockerContainerSession

from . import repo_manager
from .test_generator import DEFAULT_LANGUAGE

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of running a test on both before and after versions."""
    comment_index: int
    test_file: str
    before_passed: bool       # should be False (test should FAIL before)
    after_passed: bool        # should be True (test should PASS after)
    before_output: str
    after_output: str
    success: bool             # True if before_passed=False AND after_passed=True


def _write_test_file(test_code: str, output_dir: Path, index: int) -> Path:
    """Write test code to a file in the output directory."""
    test_path = (output_dir / f"test_review_comment_{index}.py").resolve()
    test_path.write_text(test_code)
    return test_path


def _run_single_test(
    repo_path: Path,
    venv_path: Path | None,
    test_file: Path,
    timeout: int = 60,
    language: str = DEFAULT_LANGUAGE,
) -> tuple[bool, str]:
    """Run a single test file and return (passed, output).

    Dispatches to the appropriate test runner based on language.

    Returns:
        Tuple of (passed: bool, combined_output: str).
    """
    if language == "python":
        returncode, stdout, stderr = repo_manager.run_in_repo(
            repo_path,
            ["pytest", str(test_file), "-x", "-v", "--tb=short", "--no-header"],
            venv_path=venv_path,
            timeout=timeout,
        )
    elif language in ("javascript", "typescript"):
        returncode, stdout, stderr = repo_manager.run_in_repo(
            repo_path,
            ["npx", "jest", "--no-coverage", str(test_file)],
            timeout=timeout,
        )
    elif language == "go":
        # go test expects the test file to be in a package directory
        test_dir = str(test_file.parent)
        returncode, stdout, stderr = repo_manager.run_in_repo(
            repo_path,
            ["go", "test", "-v", "-run", ".", test_dir],
            timeout=timeout,
        )
    else:
        logger.warning(
            "No test runner configured for language '%s', skipping execution",
            language,
        )
        return False, f"Execution not supported for language: {language}"

    combined = stdout + "\n" + stderr
    passed = returncode == 0
    return passed, combined.strip()


def run_test_on_version(
    repo_path: Path,
    venv_path: Path | None,
    test_file: Path,
    commit: str,
    reinstall: bool = True,
    timeout: int = 60,
    language: str = DEFAULT_LANGUAGE,
) -> tuple[bool, str]:
    """Checkout a commit, optionally reinstall, and run a test.

    Args:
        repo_path: Path to the cloned repository.
        venv_path: Path to the virtual environment (Python) or None.
        test_file: Path to the test file to run.
        commit: Git commit hash to checkout.
        reinstall: Whether to reinstall the package after checkout.
        timeout: Test execution timeout in seconds.
        language: Programming language of the test.

    Returns:
        Tuple of (passed: bool, output: str).
    """
    logger.info("Checking out %s", commit[:12])
    repo_manager.checkout_commit(repo_path, commit)

    if reinstall:
        if language == "python" and venv_path:
            logger.info("Installing repo at %s", commit[:12])
            repo_manager.pip_install(repo_path, venv_path, ["-e", "."], no_deps=True)
        elif language in ("javascript", "typescript"):
            logger.info("Installing npm packages at %s", commit[:12])
            repo_manager.run_in_repo(repo_path, ["npm", "install", "--ignore-scripts"], timeout=120)

    return _run_single_test(repo_path, venv_path, test_file, timeout=timeout, language=language)


def run_test_pair(
    repo_path: Path,
    venv_path: Path | None,
    test_file: Path,
    head_commit: str,
    merged_commit: str,
    comment_index: int = 0,
    timeout: int = 60,
    language: str = DEFAULT_LANGUAGE,
) -> TestResult:
    """Run a single test file against both before/after commits.

    Checks out each commit, reinstalls dependencies, runs the test,
    and returns a TestResult.

    Args:
        repo_path: Path to the cloned repository.
        venv_path: Path to the virtual environment (Python) or None.
        test_file: Path to the test file to run.
        head_commit: The 'before' commit (pre-review).
        merged_commit: The 'after' commit (post-review).
        comment_index: Index of the comment this test corresponds to.
        timeout: Test execution timeout in seconds.
        language: Programming language of the test.

    Returns:
        TestResult with before/after pass status and outputs.
    """
    # Run on "before" version
    logger.info("  [pair] Testing BEFORE (head: %s)", head_commit[:12])
    before_passed, before_output = run_test_on_version(
        repo_path, venv_path, test_file, head_commit,
        reinstall=True, timeout=timeout, language=language,
    )
    before_status = "PASS" if before_passed else "FAIL"
    logger.info("  [pair] before: %s", before_status)

    # Run on "after" version
    logger.info("  [pair] Testing AFTER (merged: %s)", merged_commit[:12])
    after_passed, after_output = run_test_on_version(
        repo_path, venv_path, test_file, merged_commit,
        reinstall=True, timeout=timeout, language=language,
    )
    after_status = "PASS" if after_passed else "FAIL"
    logger.info("  [pair] after: %s", after_status)

    success = (not before_passed) and after_passed

    return TestResult(
        comment_index=comment_index,
        test_file=str(test_file),
        before_passed=before_passed,
        after_passed=after_passed,
        before_output=before_output,
        after_output=after_output,
        success=success,
    )


def run_tests_for_instance(
    instance: dict,
    repo_path: Path,
    venv_path: Path | None,
    test_codes: list[tuple[int, str]],
    output_dir: Path | None = None,
    timeout: int = 60,
    language: str = DEFAULT_LANGUAGE,
) -> list[TestResult]:
    """Run all generated tests for an instance on both before/after versions.

    Args:
        instance: The dataset instance dict.
        repo_path: Path to the cloned repository.
        venv_path: Path to the virtual environment (Python) or None.
        test_codes: List of (comment_index, test_code_string) tuples.
        output_dir: Directory to write test files. Uses temp dir if None.
        timeout: Per-test timeout in seconds.
        language: Programming language of the tests.

    Returns:
        List of TestResult objects.
    """
    head_commit = instance["commit_to_review"]["head_commit"]
    merged_commit = instance["merged_commit"]

    # Write test files
    if output_dir is None:
        tmp = tempfile.mkdtemp(prefix="review_tests_")
        output_dir = Path(tmp)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_files = []
    for idx, code in test_codes:
        tf = _write_test_file(code, output_dir, idx)
        test_files.append((idx, tf))

    results = []

    # Phase 1: Run all tests on "before" version (head_commit)
    logger.info("=== Testing BEFORE version (head_commit: %s) ===", head_commit[:12])
    repo_manager.checkout_commit(repo_path, head_commit)
    if language == "python" and venv_path:
        repo_manager.pip_install(repo_path, venv_path, ["-e", "."], no_deps=True)
    elif language in ("javascript", "typescript"):
        repo_manager.run_in_repo(repo_path, ["npm", "install", "--ignore-scripts"], timeout=120)

    before_results = {}
    for idx, tf in test_files:
        passed, output = _run_single_test(
            repo_path, venv_path, tf, timeout=timeout, language=language,
        )
        before_results[idx] = (passed, output)
        status = "PASS" if passed else "FAIL"
        logger.info("  [before] test_%d: %s", idx, status)

    # Phase 2: Run all tests on "after" version (merged_commit)
    logger.info("=== Testing AFTER version (merged_commit: %s) ===", merged_commit[:12])
    repo_manager.checkout_commit(repo_path, merged_commit)
    if language == "python" and venv_path:
        repo_manager.pip_install(repo_path, venv_path, ["-e", "."], no_deps=True)
    elif language in ("javascript", "typescript"):
        repo_manager.run_in_repo(repo_path, ["npm", "install", "--ignore-scripts"], timeout=120)

    for idx, tf in test_files:
        passed, output = _run_single_test(
            repo_path, venv_path, tf, timeout=timeout, language=language,
        )
        status = "PASS" if passed else "FAIL"
        logger.info("  [after]  test_%d: %s", idx, status)

        before_passed, before_output = before_results[idx]
        success = (not before_passed) and passed

        results.append(TestResult(
            comment_index=idx,
            test_file=str(tf),
            before_passed=before_passed,
            after_passed=passed,
            before_output=before_output,
            after_output=output,
            success=success,
        ))

    return results


def _docker_run_on_version(
    session: DockerContainerSession,
    test_file: Path,
    container_test_path: str,
    commit: str,
    language: str = DEFAULT_LANGUAGE,
    timeout: int = 60,
) -> tuple[bool, str]:
    """Checkout a commit inside a Docker container, reinstall, copy test, and run it.

    Returns:
        Tuple of (passed: bool, combined_output: str).
    """
    # Checkout and clean
    checkout_result = session.run_command(
        f"git checkout --force {commit} && git clean -fd --quiet",
        timeout=120,
    )
    if checkout_result.returncode != 0:
        return False, f"git checkout failed:\n{checkout_result.stdout}\n{checkout_result.stderr}"

    # Reinstall package
    if language == "python":
        session.run_command("pip install -e . --no-deps --quiet", timeout=120)
    elif language in ("javascript", "typescript"):
        session.run_command("npm install --ignore-scripts", timeout=120)

    # Copy test file into container (after git clean which removes untracked files).
    # Retry up to 3 times to handle transient Docker "RWLayer unexpectedly nil" errors.
    for _copy_attempt in range(3):
        try:
            session.copy_to(test_file, container_test_path)
            break
        except RuntimeError as exc:
            if "RWLayer" in str(exc) and _copy_attempt < 2:
                logger.warning("docker cp failed (attempt %d/3): %s — retrying", _copy_attempt + 1, exc)
                time.sleep(2)
            else:
                raise

    # Run the test — use -c /dev/null to ignore repo pytest config (avoids
    # conftest interference from setup.cfg/pyproject.toml plugins and options)
    if language == "python":
        run_cmd = (
            f"python -m pytest {container_test_path} -x -v --tb=short --no-header "
            f"-c /dev/null -p no:cacheprovider"
        )
    elif language in ("javascript", "typescript"):
        # Check if npx is available
        check = session.run_command("which npx", timeout=10)
        if check.returncode != 0:
            return False, "npx/node.js is not installed in this Docker image. Cannot run JS/TS tests."
        run_cmd = f"npx jest --no-coverage {container_test_path}"
    elif language == "go":
        test_dir = str(Path(container_test_path).parent)
        run_cmd = f"go test -v -run . {test_dir}"
    else:
        return False, f"Execution not supported for language: {language}"

    result = session.run_command(run_cmd, timeout=timeout)
    combined = result.stdout + "\n" + result.stderr
    passed = result.returncode == 0
    return passed, combined.strip()


def run_test_pair_docker(
    session: DockerContainerSession,
    test_file: Path,
    head_commit: str,
    merged_commit: str,
    comment_index: int = 0,
    language: str = DEFAULT_LANGUAGE,
    timeout: int = 60,
) -> TestResult:
    """Run a test against both before/after commits inside a Docker container.

    Args:
        session: An active DockerContainerSession.
        test_file: Local path to the test file (will be copied into container).
        head_commit: The 'before' commit (pre-review).
        merged_commit: The 'after' commit (post-review).
        comment_index: Index of the comment this test corresponds to.
        language: Programming language of the test.
        timeout: Per-test timeout in seconds.

    Returns:
        TestResult with before/after pass status and outputs.
    """
    container_test_path = f"/workspace/{test_file.name}"

    # Run on "before" version
    logger.info("  [docker-pair] Testing BEFORE (head: %s)", head_commit[:12])
    before_passed, before_output = _docker_run_on_version(
        session, test_file, container_test_path, head_commit,
        language=language, timeout=timeout,
    )
    logger.info("  [docker-pair] before: %s", "PASS" if before_passed else "FAIL")

    # Run on "after" version
    logger.info("  [docker-pair] Testing AFTER (merged: %s)", merged_commit[:12])
    after_passed, after_output = _docker_run_on_version(
        session, test_file, container_test_path, merged_commit,
        language=language, timeout=timeout,
    )
    logger.info("  [docker-pair] after: %s", "PASS" if after_passed else "FAIL")

    success = (not before_passed) and after_passed

    return TestResult(
        comment_index=comment_index,
        test_file=str(test_file),
        before_passed=before_passed,
        after_passed=after_passed,
        before_output=before_output,
        after_output=after_output,
        success=success,
    )


def check_fail_pass(before_result: bool, after_result: bool) -> bool:
    """Verify the expected fail-then-pass pattern.

    Returns True if the test failed before the review and passed after.
    """
    return (not before_result) and after_result
