from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INSTANCE_IDS_FILE = REPO_ROOT / "results_preprocessed" / "instance-ids.txt"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "assets"
REPORT_PATH = REPORTS_DIR / "swe_care_images_test_report.json"
IGNORES_FILE = Path(__file__).resolve().parent.parent / "assets" / "swe_care_ignores.json"
STAGE_IMAGE_EXISTS = "test_images_exist"
STAGE_PYTEST_EXISTS = "test_images_pytest_exists"
STAGE_BASIC_IMPORT = "test_images_basic_import"
MAX_STAGE_WORKERS = 8
_IMAGE_REPO_SELECTOR_RE = re.compile(r"^(?P<org_repo>[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+)-\d+$")
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")

def _import_command(module_name: str) -> tuple[str, ...]:
    return (
        "python",
        "-c",
        f"import {module_name}; print(getattr({module_name}, '__version__', 'import_ok'))",
    )


PYTEST_EXISTENCE_COMMAND = _import_command("pytest")


COMMAND_MODULES: dict[str, str] = {
    "all-hands-ai__openhands": "openhands",
    "ansible__ansible": "ansible",
    "apache__airflow": "airflow",
    "avaiga__taipy": "taipy",
    "bridgecrewio__checkov": "checkov",
    "certbot__certbot": "certbot",
    "ccxt__ccxt": "ccxt",
    "chia-network__chia-blockchain": "chia",
    "conan-io__conan-6052": "conans",
    "conan-io__conan": "conan",
    "cvat-ai__cvat": "cvat",
    "dask__dask": "dask",
    "dbt-labs__dbt-core": "dbt",
    "deepset-ai__haystack": "haystack",
    "dmlc__dgl": "dgl",
    "geldata__gel": "edb",
    "frappe__frappe": "frappe",
    "freqtrade__freqtrade": "freqtrade",
    "getmoto__moto": "moto",
    "getsentry__sentry": "sentry",
    "goauthentik__authentik": "authentik",
    "gradio-app__gradio": "gradio",
    "home-assistant__core": "homeassistant",
    "hpcaitech__colossalai": "colossalai",
    "huggingface__accelerate": "accelerate",
    "huggingface__datasets": "datasets",
    "huggingface__diffusers": "diffusers",
    "hummingbot__hummingbot": "hummingbot",
    "intel__ipex-llm": "ipex_llm",
    "iterative__dvc": "dvc",
    "jina-ai__serve": "jina",
    "keephq__keep": "keep",
    "langflow-ai__langflow": "langflow",
    "letta-ai__letta-58": "memgpt",
    "letta-ai__letta": "letta",
    "lightning-ai__pytorch-lightning": "pytorch_lightning",
    "localstack__localstack": "localstack",
    "manimcommunity__manim": "manim",
    "marimo-team__marimo": "marimo",
    "matplotlib__matplotlib": "matplotlib",
    "microsoft__autogen-4044": "autogen_core",
    "microsoft__autogen-5843": "autogen_core",
    "microsoft__autogen-6529": "autogen_core",
    "microsoft__autogen": "autogen",
    "mlflow__mlflow": "mlflow",
    "modin-project__modin": "modin",
    "netbox-community__netbox": "netbox",
    "networkx__networkx": "networkx",
    "numba__numba": "numba",
    "numpy__numpy": "numpy",
    "onnx__onnx": "onnx",
    "openmined__pysyft": "syft",
    "optuna__optuna": "optuna",
    "pandas-dev__pandas": "pandas",
    "prefecthq__prefect": "prefect",
    "psf__black": "black",
    "pwndbg__pwndbg": "pwndbg",
    "pydantic__pydantic": "pydantic",
    "pyg-team__pytorch_geometric": "torch_geometric",
    "python-poetry__poetry": "poetry",
    "python-telegram-bot__python-telegram-bot": "telegram",
    "python__mypy": "mypy",
    "pytorch__vision": "torchvision",
    "qutebrowser__qutebrowser": "qutebrowser",
    "rasahq__rasa": "rasa",
    "ray-project__ray": "ray",
    "reflex-dev__reflex": "reflex",
    "run-llama__llama_index": "llama_index",
    "saleor__saleor": "saleor",
    "saltstack__salt": "salt",
    "scikit-learn__scikit-learn": "sklearn",
    "scipy__scipy": "scipy",
    "scrapy__scrapy": "scrapy",
    "skypilot-org__skypilot": "sky",
    "spyder-ide__spyder": "spyder",
    "sqlfluff__sqlfluff": "sqlfluff",
    "sympy__sympy": "sympy",
    "tobymao__sqlglot": "sqlglot",
    "vega__altair": "altair",
    "voxel51__fiftyone": "fiftyone",
    "xorbitsai__inference": "xinference",
    "zulip__zulip": "zerver",
}

COMMANDS: dict[str, tuple[str, ...]] = {
    prefix: _import_command(module_name)
    for prefix, module_name in COMMAND_MODULES.items()
}
COMMANDS["dmlc__dgl"] = ("DGLBACKEND=pytorch", *COMMANDS["dmlc__dgl"])
COMMANDS["numba__numba-4282"] = ("NUMBA_DISABLE_JIT=1", *COMMANDS["numba__numba"])
COMMANDS["posthog__posthog"] = (
    "SECRET_KEY=test",
    "DATABASE_URL=sqlite:////tmp/posthog.db",
    "REDIS_URL=redis://localhost:6379",
    *_import_command("posthog"),
)
COMMANDS["langflow-ai__langflow"] = (
    "python",
    "-c",
    "import importlib.metadata as m; d=m.distribution('langflow'); "
    "u=d.read_text('direct_url.json') or ''; "
    "print(m.version('langflow')); "
    "print('editable' if '\"editable\": true' in u else 'not_editable')",
)
COMMANDS["rasahq__rasa-7797"] = (
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python",
    *COMMANDS["rasahq__rasa"]
)
COMMANDS["voxel51__fiftyone"] = (
    "FIFTYONE_DISABLE_SERVICES=1",
    *COMMANDS["voxel51__fiftyone"]
)


def _tail(text: str | None, *, limit: int = 600) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text.strip()
    return text[-limit:].strip()


def _command_for_instance(instance_id: str) -> tuple[str, ...] | None:
    for prefix, command in COMMANDS.items():
        if prefix in instance_id:
            return command
    return None


def _write_report(payload: dict[str, object]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _append_failure_with_output(
    failures: list[str],
    *,
    instance_id: str,
    message: str,
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> None:
    failures.append(f"{instance_id}: {message}")
    if stdout_tail:
        failures.append(f"{instance_id}: stdout tail: {stdout_tail}")
    if stderr_tail:
        failures.append(f"{instance_id}: stderr tail: {stderr_tail}")


@dataclass(frozen=True)
class DatasetIgnores:
    instance_selectors: frozenset[str]
    repo_selectors: frozenset[str]

    @property
    def count(self) -> int:
        return len(self.instance_selectors) + len(self.repo_selectors)


@dataclass(frozen=True)
class ImageCheckStatus:
    exists: bool
    detail: str = ""


@dataclass(frozen=True)
class CommandRunResult:
    instance_id: str
    status: str
    command: tuple[str, ...]
    detail: str = ""
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class InstanceReportRow:
    instance_id: str
    failed_on_test: str | None = None
    skipped_after_stage: str | None = None
    stages: list[dict[str, object]] = field(default_factory=list)

    def status(self) -> str:
        if self.failed_on_test is not None:
            return "failed"
        if self.skipped_after_stage is not None:
            return "skipped"
        if self.stages:
            return "ok"
        return "pending"


class InstanceReportCollector:
    def __init__(self, instance_ids: list[str]) -> None:
        self._instance_ids = instance_ids
        self._rows: dict[str, InstanceReportRow] = {
            instance_id: InstanceReportRow(instance_id=instance_id)
            for instance_id in instance_ids
        }
        self._stage_summary: dict[str, dict[str, int]] = {}
        self._failure_messages: list[str] = []

    def blocker_for_instance(self, instance_id: str) -> str | None:
        return self._rows[instance_id].skipped_after_stage

    def add_failure_messages(self, failures: list[str]) -> None:
        self._failure_messages.extend(failures)

    def failure_messages(self) -> list[str]:
        return list(self._failure_messages)

    def record_stage(
        self,
        instance_id: str,
        stage: str,
        outcome: str,
        *,
        blocking: bool = False,
        **details: object,
    ) -> None:
        if outcome not in {"passed", "failed", "skipped"}:
            raise ValueError(f"Unsupported outcome: {outcome}")

        row = self._rows[instance_id]
        entry: dict[str, object] = {"test": stage, "status": outcome}
        for key, value in details.items():
            if value is not None:
                entry[key] = value
        row.stages.append(entry)

        stage_summary = self._stage_summary.setdefault(
            stage,
            {"passed": 0, "failed": 0, "skipped": 0},
        )
        stage_summary[outcome] += 1

        if outcome == "failed":
            if row.failed_on_test is None:
                row.failed_on_test = stage
            if row.skipped_after_stage is None:
                row.skipped_after_stage = stage
            return

        if outcome == "skipped" and blocking and row.skipped_after_stage is None:
            row.skipped_after_stage = stage

    def build_payload(self) -> dict[str, object]:
        status_summary = {"ok": 0, "failed": 0, "skipped": 0, "pending": 0}
        blocked_after_stage: dict[str, int] = {}
        results: list[dict[str, object]] = []

        for instance_id in self._instance_ids:
            row = self._rows[instance_id]
            status = row.status()
            status_summary[status] += 1
            if row.skipped_after_stage:
                blocked_after_stage[row.skipped_after_stage] = (
                    blocked_after_stage.get(row.skipped_after_stage, 0) + 1
                )
            results.append(
                {
                    "instance_id": instance_id,
                    "status": status,
                    "failed_on_test": row.failed_on_test,
                    "skipped_after_stage": row.skipped_after_stage,
                    "stages": row.stages,
                }
            )

        return {
            "total_instances": len(self._instance_ids),
            "total_failures": status_summary["failed"],
            "summary": {
                "instance_status": status_summary,
                "blocked_after_stage": blocked_after_stage,
                "stage_status": self._stage_summary,
            },
            "results": results,
        }


def _normalize_repo_selector(selector: str) -> str:
    value = selector.strip()
    if not value:
        raise ValueError("repo selector cannot be empty")

    if "/" in value:
        owner, repo = value.split("/", 1)
        if owner and repo:
            return f"{owner}/{repo}"
        raise ValueError(f"invalid repo selector: {selector!r}")

    if value.count("__") == 1:
        owner, repo = value.split("__", 1)
        if owner and repo:
            return f"{owner}/{repo}"

    raise ValueError(f"invalid repo selector: {selector!r}")


def _parse_ignore_entries(raw: object, *, key: str, selector_type: str) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, dict):
        entries: list[object] = list(raw.keys())
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"{key} must be a list or object")

    parsed: set[str] = set()
    for idx, entry in enumerate(entries):
        if isinstance(entry, str):
            selector = entry.strip()
        elif isinstance(entry, dict):
            if selector_type == "instance":
                raw_selector = entry.get("instance_id", entry.get("id", entry.get("instance")))
            else:
                raw_selector = entry.get("repo", entry.get("repository"))
            if not isinstance(raw_selector, str):
                raise ValueError(
                    f"{key}[{idx}] must include a string {selector_type} selector"
                )
            selector = raw_selector.strip()
        else:
            raise ValueError(f"{key}[{idx}] must be a string or object")

        if not selector:
            raise ValueError(f"{key}[{idx}] selector cannot be empty")
        if selector_type == "instance":
            parsed.add(selector)
        else:
            parsed.add(_normalize_repo_selector(selector))
    return parsed


def _load_dataset_ignores(path: Path) -> DatasetIgnores:
    if not path.exists():
        return DatasetIgnores(instance_selectors=frozenset(), repo_selectors=frozenset())

    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return DatasetIgnores(instance_selectors=frozenset(), repo_selectors=frozenset())

    raw = json.loads(raw_text)
    if not isinstance(raw, dict):
        raise ValueError("SWE-CARE ignore file must be a JSON object")

    instance_keys = ("instance_ignores", "instance_ids", "instances")
    repo_keys = ("repo_ignores", "repos", "repository_ignores")
    allowed_keys = {*instance_keys, *repo_keys}
    unknown_keys = sorted(key for key in raw.keys() if key not in allowed_keys)
    if unknown_keys:
        raise ValueError(f"unknown top-level keys: {', '.join(unknown_keys)}")

    instance_selectors: set[str] = set()
    repo_selectors: set[str] = set()
    for key in instance_keys:
        instance_selectors.update(
            _parse_ignore_entries(raw.get(key), key=key, selector_type="instance")
        )
    for key in repo_keys:
        repo_selectors.update(_parse_ignore_entries(raw.get(key), key=key, selector_type="repo"))
    return DatasetIgnores(
        instance_selectors=frozenset(instance_selectors),
        repo_selectors=frozenset(repo_selectors),
    )


def _repo_selector_for_image_instance_id(instance_id: str) -> str | None:
    base = instance_id.rsplit("/", 1)[-1]
    match = _IMAGE_REPO_SELECTOR_RE.fullmatch(base)
    if match is None:
        return None
    org_repo = match.group("org_repo")
    if "__" not in org_repo:
        return None
    org, repo = org_repo.split("__", 1)
    return f"{org}/{repo}"


def _instance_ignore_candidates(raw_line: str, image_instance_id: str) -> set[str]:
    candidates = {
        raw_line,
        raw_line.rsplit("/", 1)[-1],
        image_instance_id,
        image_instance_id.rsplit("/", 1)[-1],
    }
    if "@" not in raw_line:
        return candidates

    _, commit = raw_line.split("@", 1)
    commit = commit.strip()
    if not commit:
        return candidates

    base = image_instance_id.rsplit("/", 1)[-1]
    match = _IMAGE_REPO_SELECTOR_RE.fullmatch(base)
    if match is None:
        return candidates
    candidates.add(f"{match.group('org_repo')}__{commit}")
    return candidates


def _split_command_env_assignments(command: tuple[str, ...]) -> tuple[list[str], tuple[str, ...]]:
    env_assignments: list[str] = []
    index = 0
    for token in command:
        if _ENV_ASSIGNMENT_RE.fullmatch(token):
            env_assignments.append(token)
            index += 1
            continue
        break
    return env_assignments, command[index:]


def _run_container_command(
    docker_bin: str,
    instance_id: str,
    command: tuple[str, ...],
) -> CommandRunResult:
    env_assignments, container_command = _split_command_env_assignments(command)
    if not container_command:
        return CommandRunResult(
            instance_id=instance_id,
            status="failed",
            command=command,
            detail="invalid_command",
            stderr_tail="missing container command after environment assignments",
        )

    docker_run_command = [docker_bin, "run", "--rm"]
    for assignment in env_assignments:
        docker_run_command.extend(["--env", assignment])
    docker_run_command.extend([instance_id, *container_command])

    try:
        run_result = subprocess.run(
            docker_run_command,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandRunResult(
            instance_id=instance_id,
            status="failed",
            command=command,
            detail="command_timed_out",
            stdout_tail=_tail(exc.stdout),
            stderr_tail=_tail(exc.stderr),
        )
    except Exception as exc:  # noqa: BLE001
        return CommandRunResult(
            instance_id=instance_id,
            status="failed",
            command=command,
            detail="command_exception",
            stderr_tail=_tail(str(exc)),
        )

    if run_result.returncode != 0:
        return CommandRunResult(
            instance_id=instance_id,
            status="failed",
            command=command,
            detail="command_failed",
            exit_code=run_result.returncode,
            stdout_tail=_tail(run_result.stdout),
            stderr_tail=_tail(run_result.stderr),
        )
    return CommandRunResult(
        instance_id=instance_id,
        status="passed",
        command=command,
    )


@pytest.fixture(scope="session")
def docker_bin() -> str:
    docker = shutil.which("docker")
    if docker is None:
        pytest.fail("docker CLI not found in PATH")

    version = subprocess.run(
        [docker, "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
    )
    if version.returncode != 0:
        detail = _tail(version.stderr) or _tail(version.stdout) or "unknown error"
        pytest.fail(f"docker daemon is not reachable: {detail}")
    return docker


@pytest.fixture(scope="session")
def instance_ids() -> list[str]:
    if not INSTANCE_IDS_FILE.exists():
        pytest.fail(f"Missing instance id file: {INSTANCE_IDS_FILE}")

    try:
        ignores = _load_dataset_ignores(IGNORES_FILE)
    except (json.JSONDecodeError, ValueError) as exc:
        pytest.fail(f"Invalid SWE-CARE ignore file {IGNORES_FILE}: {exc}")

    seen: set[str] = set()
    ids: list[str] = []
    ignored_by_instance = 0
    ignored_by_repo = 0
    with INSTANCE_IDS_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            instance_id = stripped.split("@", 1)[0].strip()
            if instance_id in seen:
                continue

            ignore_candidates = _instance_ignore_candidates(stripped, instance_id)
            if ignores.instance_selectors.intersection(ignore_candidates):
                ignored_by_instance += 1
                continue
            repo_selector = _repo_selector_for_image_instance_id(instance_id)
            if repo_selector is not None and repo_selector in ignores.repo_selectors:
                ignored_by_repo += 1
                continue

            seen.add(instance_id)
            ids.append(instance_id)

    if ignores.count > 0 and (ignored_by_instance > 0 or ignored_by_repo > 0):
        print(
            f"Applied SWE-CARE ignores from {IGNORES_FILE}: "
            f"ignored {ignored_by_instance + ignored_by_repo} instance(s) "
            f"(instance selectors={ignored_by_instance}, repo selectors={ignored_by_repo})"
        )

    return ids


@pytest.fixture(scope="session")
def report_collector(instance_ids: list[str]):
    collector = InstanceReportCollector(instance_ids)
    yield collector
    _write_report(collector.build_payload())


@pytest.fixture(scope="session")
def image_check_results(
    docker_bin: str,
    instance_ids: list[str],
    report_collector: InstanceReportCollector,
) -> dict[str, ImageCheckStatus]:
    results: dict[str, ImageCheckStatus] = {}

    for instance_id in tqdm(instance_ids, desc=STAGE_IMAGE_EXISTS):
        inspect_result = subprocess.run(
            [docker_bin, "image", "inspect", instance_id],
            capture_output=True,
            text=True,
        )
        if inspect_result.returncode != 0:
            detail = _tail(inspect_result.stderr) or _tail(inspect_result.stdout)
            if not detail:
                detail = "docker image inspect failed"
            results[instance_id] = ImageCheckStatus(exists=False, detail=detail)
            report_collector.record_stage(
                instance_id,
                STAGE_IMAGE_EXISTS,
                "failed",
                detail=detail,
                blocking=True,
            )
            continue

        results[instance_id] = ImageCheckStatus(exists=True)
        report_collector.record_stage(instance_id, STAGE_IMAGE_EXISTS, "passed")

    return results


def test_images_exist(image_check_results: dict[str, ImageCheckStatus]) -> None:
    # Stage placeholder: per-instance outcomes are recorded in image_check_results/report_collector.
    # Final failure assertion is consolidated in test_swe_care_images_report.
    pass


@pytest.fixture(scope="session")
def pytest_check_results(
    docker_bin: str,
    instance_ids: list[str],
    report_collector: InstanceReportCollector,
    image_check_results: dict[str, ImageCheckStatus],
) -> dict[str, CommandRunResult]:
    if len(image_check_results) != len(instance_ids):
        pytest.fail(f"{STAGE_IMAGE_EXISTS} did not record results for all instances")

    failures: list[str] = []
    results: dict[str, CommandRunResult] = {}
    runnable: list[str] = []

    for instance_id in tqdm(instance_ids, desc=f"{STAGE_PYTEST_EXISTS} precheck"):
        blocker = report_collector.blocker_for_instance(instance_id)
        if blocker is not None:
            results[instance_id] = CommandRunResult(
                instance_id=instance_id,
                status="skipped",
                command=PYTEST_EXISTENCE_COMMAND,
                detail=f"Skipped after {blocker}",
            )
            report_collector.record_stage(
                instance_id,
                STAGE_PYTEST_EXISTS,
                "skipped",
                detail=f"Skipped after {blocker}",
                blocked_by_stage=blocker,
            )
            continue

        runnable.append(instance_id)

    if runnable:
        max_workers = min(MAX_STAGE_WORKERS, len(runnable))
        future_to_instance: dict[object, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for instance_id in runnable:
                future = executor.submit(
                    _run_container_command,
                    docker_bin,
                    instance_id,
                    PYTEST_EXISTENCE_COMMAND,
                )
                future_to_instance[future] = instance_id

            for future in tqdm(
                as_completed(future_to_instance),
                total=len(future_to_instance),
                desc=f"{STAGE_PYTEST_EXISTS} run",
            ):
                result = future.result()
                results[result.instance_id] = result
                if result.status == "passed":
                    report_collector.record_stage(
                        result.instance_id,
                        STAGE_PYTEST_EXISTS,
                        "passed",
                        command=list(result.command),
                    )
                    continue

                report_collector.record_stage(
                    result.instance_id,
                    STAGE_PYTEST_EXISTS,
                    "failed",
                    command=list(result.command),
                    detail=result.detail,
                    exit_code=result.exit_code,
                    stdout_tail=result.stdout_tail,
                    stderr_tail=result.stderr_tail,
                    blocking=True,
                )
                _append_failure_with_output(
                    failures,
                    instance_id=result.instance_id,
                    message=f"pytest check failed ({' '.join(result.command)})",
                    stdout_tail=result.stdout_tail,
                    stderr_tail=result.stderr_tail,
                )
    else:
        max_workers = 0

    if failures:
        report_collector.add_failure_messages(failures)
        return results

    # keep this value visible in report/debug streams when needed
    if max_workers:
        print(f"{STAGE_PYTEST_EXISTS}: executed {len(runnable)} commands with {max_workers} workers")

    return results


def test_images_pytest_exists(pytest_check_results: dict[str, CommandRunResult]) -> None:
    # Stage placeholder: per-instance outcomes are recorded in pytest_check_results/report_collector.
    # Final failure assertion is consolidated in test_swe_care_images_report.
    pass


def test_images_basic_import(
    docker_bin: str,
    instance_ids: list[str],
    report_collector: InstanceReportCollector,
    image_check_results: dict[str, ImageCheckStatus],
    pytest_check_results: dict[str, CommandRunResult],
) -> None:
    if len(pytest_check_results) != len(instance_ids):
        pytest.fail(f"{STAGE_PYTEST_EXISTS} did not record results for all instances")

    failures: list[str] = []
    runnable: list[tuple[str, tuple[str, ...]]] = []

    for instance_id in tqdm(instance_ids, desc=f"{STAGE_BASIC_IMPORT} precheck"):
        blocker = report_collector.blocker_for_instance(instance_id)
        if blocker is not None:
            report_collector.record_stage(
                instance_id,
                STAGE_BASIC_IMPORT,
                "skipped",
                detail=f"Skipped after {blocker}",
                blocked_by_stage=blocker,
            )
            continue

        command = _command_for_instance(instance_id)
        if command is None:
            report_collector.record_stage(
                instance_id,
                STAGE_BASIC_IMPORT,
                "skipped",
                detail="No command configured in COMMANDS",
                blocking=True,
            )
            continue

        image_status = image_check_results[instance_id]
        if not image_status.exists:
            report_collector.record_stage(
                instance_id,
                STAGE_BASIC_IMPORT,
                "skipped",
                detail=f"Skipped after {STAGE_IMAGE_EXISTS}",
                blocked_by_stage=STAGE_IMAGE_EXISTS,
            )
            continue

        runnable.append((instance_id, command))

    if runnable:
        max_workers = min(MAX_STAGE_WORKERS, len(runnable))
        future_to_instance: dict[object, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for instance_id, command in runnable:
                future = executor.submit(
                    _run_container_command,
                    docker_bin,
                    instance_id,
                    command,
                )
                future_to_instance[future] = instance_id

            for future in tqdm(
                as_completed(future_to_instance),
                total=len(future_to_instance),
                desc=f"{STAGE_BASIC_IMPORT} run",
            ):
                result = future.result()
                if result.status == "passed":
                    report_collector.record_stage(
                        result.instance_id,
                        STAGE_BASIC_IMPORT,
                        "passed",
                        command=list(result.command),
                    )
                    continue

                if result.detail == "command_timed_out":
                    message = f"command timed out ({' '.join(result.command)})"
                elif result.detail == "command_failed":
                    message = f"command failed ({' '.join(result.command)})"
                elif result.detail == "command_exception":
                    message = f"command runner exception ({' '.join(result.command)})"
                else:
                    message = f"command error ({' '.join(result.command)})"

                report_collector.record_stage(
                    result.instance_id,
                    STAGE_BASIC_IMPORT,
                    "failed",
                    command=list(result.command),
                    detail=result.detail,
                    exit_code=result.exit_code,
                    stdout_tail=result.stdout_tail,
                    stderr_tail=result.stderr_tail,
                    blocking=True,
                )
                _append_failure_with_output(
                    failures,
                    instance_id=result.instance_id,
                    message=message,
                    stdout_tail=result.stdout_tail,
                    stderr_tail=result.stderr_tail,
                )
    else:
        max_workers = 0

    if failures:
        report_collector.add_failure_messages(failures)
        return

    # keep this value visible in report/debug streams when needed
    if max_workers:
        print(f"{STAGE_BASIC_IMPORT}: executed {len(runnable)} commands with {max_workers} workers")


def test_swe_care_images_report(
    report_collector: InstanceReportCollector,
    image_check_results: dict[str, ImageCheckStatus],
) -> None:
    failures = report_collector.failure_messages()
    for instance_id, status in image_check_results.items():
        if status.exists:
            continue
        detail = status.detail or "docker image inspect failed"
        failures.append(f"{instance_id}: docker image missing ({detail})")

    if failures:
        pytest.fail("\n".join(failures))
