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

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake protobuf-compiler libprotobuf-dev
rm -rf /var/lib/apt/lists/*

cd /workspace
CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON -DProtobuf_USE_STATIC_LIBS=OFF" \
  "$PY_BIN" -m pip install -e . --no-build-isolation --no-deps
"$PY_BIN" -m pip install "protobuf<4" "numpy<2" six "typing-extensions>=3.6.2.1"

"$PY_BIN" - <<'PY'
import onnx
print("IMPORT_OK", getattr(onnx, "__version__", "import_ok"))
PY
