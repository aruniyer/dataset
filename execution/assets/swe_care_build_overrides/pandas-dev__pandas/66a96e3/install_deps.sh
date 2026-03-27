#!/usr/bin/env bash
set -euxo pipefail
python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
python -m pip install --no-warn-conflicts "Cython>=0.29.13,<3" "python-dateutil>=2.6.1" "pytz>=2017.2" "numpy==1.23.5"
rm -rf build
find pandas -name "*.so" -delete
CFLAGS="-DNUMPY_IMPORT_ARRAY_RETVAL=NULL" python setup.py build_ext --inplace --force -j 4
python setup.py develop
