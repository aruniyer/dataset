#!/usr/bin/env bash
set -euxo pipefail
python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
python -m pip install -e . --no-deps
python -m pip install --no-warn-conflicts \
  "numpy<2" "packaging" "filelock" "importlib_metadata" "requests" "Pillow" \
  "regex!=2019.12.17" "typing_extensions" "huggingface_hub==0.16.4"
python -m pip uninstall -y accelerate transformers torch torchvision || true
