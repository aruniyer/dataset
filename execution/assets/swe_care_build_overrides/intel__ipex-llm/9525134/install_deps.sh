#!/usr/bin/env bash
set -euxo pipefail

PY_BIN="${PY_BIN:-python}"
REPO_DIR="${REPO_DIR:-/workspace}"
INSTANCE="intel__ipex-llm-5097"
NANO_DIR="$REPO_DIR/python/nano"
BRIDGE_ROOT="$REPO_DIR/compat_ipex_llm"
BRIDGE_PKG="$BRIDGE_ROOT/ipex_llm"

if [[ ! -f "$NANO_DIR/setup.py" ]]; then
  echo "Expected BigDL Nano source at $NANO_DIR" >&2
  exit 1
fi

"$PY_BIN" -m pip install --upgrade "pip<25.3" "setuptools<82" "wheel<0.46"
"$PY_BIN" -m pip install cloudpickle "protobuf<4" intel-openmp
"$PY_BIN" -m pip install -e "$NANO_DIR" --no-deps

SITE_PKGS="$("$PY_BIN" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

mkdir -p "$BRIDGE_PKG"
cat > "$BRIDGE_PKG/__init__.py" <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bigdl-nano")
except PackageNotFoundError:
    __version__ = "0.0.0"
PY
printf "%s\n" "$BRIDGE_ROOT" > "$SITE_PKGS/zz_${INSTANCE}_src.pth"

"$PY_BIN" - <<'PY'
import ipex_llm
print("IMPORT_OK", getattr(ipex_llm, "__version__", "import_ok"))
PY
