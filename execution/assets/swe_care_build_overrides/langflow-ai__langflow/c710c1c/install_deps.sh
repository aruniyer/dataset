#!/usr/bin/env bash
set -euxo pipefail

if [[ -x "/usr/local/bin/python" ]]; then
  PY_BIN="/usr/local/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python)"
else
  PY_BIN="$(command -v python3)"
fi

"$PY_BIN" -m pip install --upgrade "pip<25.3" "setuptools<82" "wheel<0.46"

if "$PY_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  export PIP_IGNORE_REQUIRES_PYTHON=1
fi

"$PY_BIN" -m pip install \
  "pydantic<2" \
  "fastapi<0.101" \
  "langchain==0.0.274" \
  "loguru<0.8" \
  "pyyaml" \
  "appdirs"

cd /workspace
"$PY_BIN" -m pip install -e . --no-deps

"$PY_BIN" - <<'PY'
import importlib.metadata as metadata

dist = metadata.distribution("langflow")
direct_url = dist.read_text("direct_url.json") or ""
print("IMPORT_OK", metadata.version("langflow"), "editable=" + str('"editable": true' in direct_url))
PY
