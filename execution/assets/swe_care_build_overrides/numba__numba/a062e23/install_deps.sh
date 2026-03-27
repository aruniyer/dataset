PY_BIN="python"
"$PY_BIN" -m pip install --upgrade "pip<24" "setuptools<69" wheel
"$PY_BIN" -m pip uninstall -y numba >/dev/null 2>&1 || true
"$PY_BIN" -m pip install "llvmlite==0.29.0" "numpy==1.17.5"
"$PY_BIN" -m pip install -e . --no-deps