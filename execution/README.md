# Execution Environments

Build execution environments as Docker images for SWE-CARE instances.

Language support is inferred per dataset row (or language override) and currently includes:
- Python (full dependency install path)
- JavaScript/TypeScript (install/check Node runtime)

If both Python and JS/TS are present, Python setup runs and Node is installed.
If the row is JS/TS-only, Python package setup is skipped.

## CLI usage

```bash
# First 10 instances from test split
python -m execution.build_swe_care --split test --limit 10 --max-workers 4

# Only one repo from the dataset
python -m execution.build_swe_care --split test --repo pandas-dev/pandas --max-workers 4

# Only specific instances (repeat --instance and/or comma-separate values)
python -m execution.build_swe_care --split test --max-workers 4 \
  --instance pandas-dev__pandas__22a6bff \
  --instance scipy__scipy__57e11f0,cupy__cupy__158ac6a
```

Default SWE-CARE image tags:

- `reviewbench/{org}__{repo}-{pull_number}:latest`

After build:

```bash
docker run --rm -it <image-tag>
```

Inside the container, the repo is available at `/workspace` and already on the configured commit.

To force a specific Python version instead of auto-inference:

```bash
python -m execution.build_swe_care --split test --repo pandas-dev/pandas --python-version 3.10
```

When `--python-version auto` is used, the builder retries other supported
Python versions on dependency compatibility errors.

Use `--build-timeout-sec` to cap a single Docker build duration
(default: `7200` seconds, set `0` to disable).

To route dependency downloads through local proxy/index services:

- `--apt-proxy` (default: `http://host.docker.internal:3142`)
- `--pip-index-url` (default: `http://host.docker.internal:3141/root/pypi/+simple`)
- `--pip-extra-index-url` (optional secondary index)
- `--pip-trusted-host` (default: `host.docker.internal`)

`--repos-dir` is used for Python version inference and as the local cached-repo
source for Docker build seeding.

## Proxy/index setup (apt + pip)

If you keep the default CLI values, run local services on the host with:

- apt proxy at `http://host.docker.internal:3142`
- pip index at `http://host.docker.internal:3141/root/pypi/+simple`

### 1) apt-cacher-ng (apt proxy)

Use the Quickstart command from:

- https://github.com/sameersbn/docker-apt-cacher-ng

This project’s default `--apt-proxy` value already points to that service on port `3142`.

### 2) devpi-server (pip index)

```bash
uv add devpi-server
devpi-init
devpi-server --host 0.0.0.0
```

Then keep (or set) the pip flags to:

- `--pip-index-url http://host.docker.internal:3141/root/pypi/+simple`
- `--pip-trusted-host host.docker.internal`

Example:

```bash
python -m execution.build_swe_care --split test --repo pandas-dev/pandas \
  --pip-index-url http://host.docker.internal:3141/root/pypi/+simple \
  --pip-trusted-host host.docker.internal
```

## Programmatic Container Utilities

Use `execution.container_runtime.DockerContainerSession` to run commands/tests,
copy files, and read logs from containers.

```python
from execution.container_runtime import DockerContainerSession

with DockerContainerSession(image="<image-tag>") as session:
    cmd = session.run_command(["python", "-V"], check=True)
    test = session.run_tests(test_target="tests/test_local.py", pytest_args=["-q"], timeout=600)

    session.copy_to("tests/test_local.py", "/workspace/tests/test_local.py")
    session.copy_from("/workspace/.pytest_cache", "artifacts/pytest_cache")

    logs = session.read_logs(tail=200)
    print(cmd.stdout.strip())
    print(test.returncode)
    print(logs.stdout[-1000:])
```

## SWE-CARE Image Builder

Build images for SWE-CARE `test` split with multithreading. The script writes:

- run log: `<output_dir>/<split>_<timestamp>/run.log`
- per-build logs: `<output_dir>/<split>_<timestamp>/build_logs/*.log`
- detailed results: `<output_dir>/<split>_<timestamp>/results.json`
- summary: `<output_dir>/<split>_<timestamp>/summary.json`

Result statuses include:

- `ok`: image built and basic checks passed
- `check_failed`: image built but post-build checks failed
- `soft_error`: image built, but build log shows suppressed dependency/setup failures
  (for example, `Warning: repository package install failed; continuing` or
  `Info: skipping repository package install after build-time compatibility failure.`)
- `error`: docker build failed
- `skipped_existing_image`: image tag already exists and `--force-rebuild` was not set
- `skipped_non_python`: dataset row skipped because language is non-Python (default behavior)
- `skipped_invalid_repo`: dataset row has invalid `repo`
- `skipped_invalid_commit`: dataset row has invalid commit
- `skipped_invalid_pull_number`: dataset row has invalid PR number
- `skipped_short_commit_collision`: duplicate short commit id collision within one repo

SWE-CARE image tags are:

- `reviewbench/{org}__{repo}-{pull_number}:latest`

```bash
# First 10 instances
python -m execution.build_swe_care --split test --limit 10 --max-workers 4

# Full test split
python -m execution.build_swe_care --split test --max-workers 4

# Only one repo from the dataset
python -m execution.build_swe_care --split test --repo pandas-dev/pandas --max-workers 4

# Only specific instances (org__repo__hash format; repeat --instance or use commas)
python -m execution.build_swe_care --split test --max-workers 4 \
  --instance pandas-dev__pandas__22a6bff \
  --instance scipy__scipy__57e11f0,cupy__cupy__158ac6a

# Full test split with 2-hour per-build timeout (default behavior)
python -m execution.build_swe_care --split test --max-workers 4 --build-timeout-sec 7200
```

### Patch SWE-CARE Language Rows

SWE-CARE `language` rows can be monkey-patched using a local JSON file.
By default, `execution/build_swe_care.py` reads:

- `execution/swe_care_language_overrides.json`

The file can be empty (or `{}`), which means no overrides.

Run explicitly with a custom file if needed:

```bash
python -m execution.build_swe_care \
  --split test \
  --language-overrides-file execution/swe_care_language_overrides.example.json
```

Supported override file shapes:

- Backward-compatible object mapping `instance_id -> language_string`
- Object with:
  - `instance_overrides` (or `language_overrides` / `instance_id_overrides` / `overrides`)
  - `repo_overrides` (or `repository_overrides`)
- List of rules:
  - `{"instance_id": "...", "language": "..."}`
  - `{"repo": "owner/name", "language": "..."}`

If both match one row, instance-level overrides take precedence over repo-level overrides.

Language strings are parsed robustly (case-insensitive, extra spaces allowed):

- `Python`
- `Python, Javascript`
- `py, js`
- `py, ts`

### Manual Build Overrides

You can provide repo/commit-specific overrides under:

- `execution/assets/swe_care_build_overrides/{org}__{repo}/{commit_prefix}/`

The commit directory is matched by prefix (longest match wins).

Supported override files:

- `.python-version`: overrides Python auto-inference for that task
- `Dockerfile`: used directly instead of rendered Dockerfile
- `setup_repo.sh`: used directly instead of rendered setup script
- `install_deps.sh`: used directly instead of rendered install script
- `post_install.sh`: runs after `install_deps.sh` succeeds

Example:

- `execution/assets/swe_care_build_overrides/pandas-dev__pandas/8e4adbb/.python-version`
- `execution/assets/swe_care_build_overrides/pandas-dev__pandas/8e4adbb/install_deps.sh`

`build_swe_care` uses this override root by default:

- `--build-overrides-root execution/assets/swe_care_build_overrides`
