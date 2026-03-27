#!/usr/bin/env bash
set -euxo pipefail
python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
python -m pip install -e .
python -m pip install --no-warn-conflicts \
  "numpy<2" "pandas<2" "protobuf<4" "pydantic<2" "scikit-learn<1.2" \
  "huggingface_hub==0.10.1"
python -m pip install --no-warn-conflicts -f https://download.pytorch.org/whl/torch_stable.html \
  "torch==1.13.1+cpu"
