#!/usr/bin/env bash
set -euxo pipefail
python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
FIFTYONE_DISABLE_SERVICES=1 python -m pip install -e .
python -m pip install --no-warn-conflicts "pymongo<4" "mongoengine<0.25" "motor<3"
