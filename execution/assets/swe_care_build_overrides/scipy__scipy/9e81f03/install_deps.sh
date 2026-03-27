#!/usr/bin/env bash
set -euxo pipefail
PY_BIN="python"
REPO_DIR="/workspace"
PKG_DIR="/workspace/scipy"
INSTANCE="scipy__scipy-9488"
SCIPY_WHEEL_VERSION="1.3.3"

if "$PY_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info < (3, 8) else 1)
PY
then
  "$PY_BIN" -m pip install --upgrade "pip<24" "setuptools<69" wheel
else
  "$PY_BIN" -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
fi

"$PY_BIN" -m pip install "scipy==$SCIPY_WHEEL_VERSION"
SITE_PKGS="$($PY_BIN - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
WHEEL_SCIPY="$SITE_PKGS/scipy"
if [ ! -d "$WHEEL_SCIPY" ]; then
  echo "Installed scipy wheel path not found: $WHEEL_SCIPY" >&2
  exit 1
fi

export WHEEL_SCIPY PKG_DIR
"$PY_BIN" - <<'PY'
import os
import shutil
from pathlib import Path
wheel = Path(os.environ["WHEEL_SCIPY"])
pkg = Path(os.environ["PKG_DIR"])
copy_names = {"__config__.py", "_distributor_init.py", "version.py"}
count = 0
for p in wheel.rglob("*"):
    if not p.is_file():
        continue
    if p.suffix in {".so", ".pyd", ".dylib"} or p.name in copy_names:
        rel = p.relative_to(wheel)
        out = pkg / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        count += 1
if count == 0:
    raise SystemExit("No SciPy binary/config artifacts copied from wheel")
if not (pkg / "__config__.py").exists():
    raise SystemExit(f"Missing expected file: {pkg / '__config__.py'}")
print(f"Copied {count} artifacts into {pkg}")
PY

"$PY_BIN" -m pip uninstall -y scipy
printf "%s\n" "$REPO_DIR" > "$SITE_PKGS/zz_${INSTANCE}_src.pth"
