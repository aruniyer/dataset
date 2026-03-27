PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install "setuptools<70"
"$PY_BIN" -m pip install -e .
"$PY_BIN" -m pip install --no-warn-conflicts "protobuf<4" "grpcio<2" "grpcio-tools<2"
"$PY_BIN" -m pip install --no-warn-conflicts "docarray<=0.21.0"