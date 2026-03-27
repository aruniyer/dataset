#!/usr/bin/env bash
set -euxo pipefail
PY_BIN="python"
SITE_PKGS="$($PY_BIN - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
$PY_BIN -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
$PY_BIN -m pip install --no-warn-conflicts "numpy==1.23.5" "pandas==1.5.3"
printf "%s\n" "/workspace" > "$SITE_PKGS/zz_hummingbot__hummingbot-4889_src.pth"
