PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install -e .
"$PY_BIN" -m pip install "huggingface_hub<1.0" "setuptools==70.0.0"