PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install "pip==23.2.1"
SENTRY_LIGHT_BUILD=1 "$PY_BIN" -m pip install -e . --no-deps