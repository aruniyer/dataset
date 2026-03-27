#!/usr/bin/env bash
set -euxo pipefail
python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
python -m pip install --no-warn-conflicts "numpy<2" "meson>=1.10" "meson-python>=0.15" pybind11 ninja "setuptools_scm[toml]>=7"
rm -rf "./build/cp311"
python -m pip install -e . --no-build-isolation --config-settings=setup-args=-Dsystem-freetype=true --config-settings=setup-args=-Dsystem-qhull=true
python -m pip install --no-warn-conflicts "numpy<2"
