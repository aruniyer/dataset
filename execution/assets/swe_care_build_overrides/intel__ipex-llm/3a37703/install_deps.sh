#!/usr/bin/env bash
set -euxo pipefail

PY_BIN="${PY_BIN:-python}"
REPO_DIR="${REPO_DIR:-/workspace}"
INSTANCE="intel__ipex-llm-774"
LEGACY_SRC="$REPO_DIR/pyspark"
BRIDGE_ROOT="$REPO_DIR/compat_ipex_llm"
BRIDGE_PKG="$BRIDGE_ROOT/ipex_llm"

"$PY_BIN" -m pip install --upgrade "pip<25.3" "setuptools<82" "wheel<0.46"
"$PY_BIN" -m pip install numpy six cloudpickle

SITE_PKGS="$("$PY_BIN" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

if [[ -d "$LEGACY_SRC" ]]; then
  printf "%s\n" "$LEGACY_SRC" > "$SITE_PKGS/zz_${INSTANCE}_legacy_src.pth"
fi

mkdir -p "$BRIDGE_PKG"
cat > "$BRIDGE_PKG/__init__.py" <<'PY'
from pathlib import Path

__version__ = "0.0.0"
_repo_root = Path(__file__).resolve().parents[2]
_version_file = _repo_root / "python" / "version.txt"
if _version_file.is_file():
    try:
        __version__ = _version_file.read_text(encoding="utf-8").strip()
    except OSError:
        pass
PY
printf "%s\n" "$BRIDGE_ROOT" > "$SITE_PKGS/zz_${INSTANCE}_src.pth"

"$PY_BIN" - <<'PY'
import ipex_llm
print("IMPORT_OK", getattr(ipex_llm, "__version__", "import_ok"))
PY
