PY_BIN="python"
SRC_DIR="/workspace"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install -f https://download.pytorch.org/whl/torch_stable.html "torch==1.7.1+cpu"
"$PY_BIN" -m pip install "setuptools==69.5.1"
"$PY_BIN" -m pip install "numpy<2" "pillow<10" "ninja" "typing_extensions<5"
# Avoid compiling C/C++ extensions for this benchmark setup.
sed -i 's/return ext_modules/return []/' "$SRC_DIR/setup.py"
BUILD_CUSTOM_OPS=0 FORCE_CUDA=0 "$PY_BIN" -m pip install -e "$SRC_DIR" --no-build-isolation