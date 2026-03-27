PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install -e .
"$PY_BIN" -m pip install --no-warn-conflicts "psutil"