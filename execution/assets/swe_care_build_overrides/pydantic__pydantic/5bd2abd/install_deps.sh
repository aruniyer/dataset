PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install -e "$SRC_DIR" --no-deps
"$PY_BIN" -m pip install --no-warn-conflicts "pydantic-core==0.13.0" "typing_extensions>=4.2" "annotated-types>=0.4.0"