#!/usr/bin/env bash
set -euxo pipefail

if [[ -x "/usr/local/bin/python" ]]; then
  PY_BIN="/usr/local/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
else
  PY_BIN="$(command -v python3)"
fi

"$PY_BIN" -m pip install --upgrade "pip<24.1" "setuptools<66" "wheel<0.46"
"$PY_BIN" -m pip install -e /workspace/python

"$PY_BIN" - <<'PY'
import ccxt
print("IMPORT_OK", getattr(ccxt, "__version__", "import_ok"))
PY
