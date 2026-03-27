#!/usr/bin/env bash
set -euxo pipefail
PY_BIN="python"
REPO_DIR="/workspace"
PKG_DIR="/workspace/numba"
INSTANCE="numba__numba-9144"
SITE_PKGS="$($PY_BIN - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

$PY_BIN -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
$PY_BIN -m pip install "numba==0.58.1" "llvmlite==0.41.1" "numpy<2"
WHEEL_NUMBA="$SITE_PKGS/numba"
if [ ! -d "$WHEEL_NUMBA" ]; then
  echo "Installed numba wheel path not found: $WHEEL_NUMBA" >&2
  exit 1
fi

export WHEEL_NUMBA PKG_DIR
$PY_BIN - <<'PY'
import os
import shutil
from pathlib import Path
wheel = Path(os.environ["WHEEL_NUMBA"])
pkg = Path(os.environ["PKG_DIR"])
count = 0
for p in wheel.rglob("*"):
    if p.is_file() and p.suffix in {".so", ".pyd", ".dylib"}:
        rel = p.relative_to(wheel)
        out = pkg / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        count += 1
if count == 0:
    raise SystemExit("No compiled numba artifacts copied from wheel")
print(f"Copied {count} compiled artifacts into {pkg}")
PY

$PY_BIN -m pip uninstall -y numba
printf "%s\n" "$REPO_DIR" > "$SITE_PKGS/zz_${INSTANCE}_src.pth"
