PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2"
"$PY_BIN" -m pip install "numpy<2"
"$PY_BIN" -m pip install -e .
"$PY_BIN" -m pip install \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html \
  "torch-scatter" "torch-sparse"