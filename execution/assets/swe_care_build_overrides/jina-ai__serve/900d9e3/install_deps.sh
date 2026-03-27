PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install "setuptools<70"
# This commit pins an opentelemetry prerelease range that is no longer resolvable.
"$PY_BIN" -m pip install -e . --no-deps
"$PY_BIN" -m pip install --no-warn-conflicts \
    "docarray<=0.21.0" \
    "protobuf<4" \
    "grpcio<2" \
    "grpcio-tools<2" \
    "grpcio-health-checking<2" \
    "grpcio-reflection<2" \
    "pyyaml" "requests" "aiohttp" "fastapi" "uvicorn" "websockets" "pydantic<2"