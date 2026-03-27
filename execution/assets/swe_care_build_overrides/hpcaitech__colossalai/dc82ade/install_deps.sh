PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==1.13.1"
"$PY_BIN" -m pip install "numpy<2"
CUDA_EXT=0 "$PY_BIN" -m pip install -e .