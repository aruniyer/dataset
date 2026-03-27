#!/usr/bin/env bash
set -euxo pipefail
PY_BIN="python"
SRC_DIR="/workspace"
$PY_BIN -m pip install --upgrade "pip<24.1" "setuptools<60" wheel
$PY_BIN -m pip install -f https://download.pytorch.org/whl/torch_stable.html "torch==1.10.2+cpu"
$PY_BIN -m pip install "numpy<1.20" "pillow<9" "ninja" "typing_extensions<4.5"
BUILD_CUSTOM_OPS=0 FORCE_CUDA=0 $PY_BIN -m pip install -e "$SRC_DIR" --no-build-isolation
