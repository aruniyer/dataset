PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install -e . --no-deps
"$PY_BIN" -m pip install --no-warn-conflicts Click Werkzeug PyPika redis traceback_with_variables bleach-allowlist psutil requests semantic_version