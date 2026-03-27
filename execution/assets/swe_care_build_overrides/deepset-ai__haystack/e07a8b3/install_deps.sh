#!/usr/bin/env bash
set -euxo pipefail
python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
python -m pip install -e .
python -m pip install --no-warn-conflicts \
  "numpy==1.23.5" "pandas<2" "protobuf<4" "pydantic<2" \
  "huggingface_hub==0.10.1" "transformers==4.21.2"
python -m pip install --no-warn-conflicts -f https://download.pytorch.org/whl/torch_stable.html \
  "torch==1.13.1+cpu" "torchvision==0.14.1+cpu"
