#!/usr/bin/env bash
set -euxo pipefail
python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
python -m pip install -e .
python -m pip install --no-warn-conflicts "protobuf<3.21" "numpy<2"
