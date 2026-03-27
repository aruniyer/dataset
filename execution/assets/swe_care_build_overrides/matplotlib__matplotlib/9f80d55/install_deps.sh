pip install --upgrade pip setuptools wheel
pip install "numpy<2" "meson>=1.10" "meson-python>=0.15" pybind11 ninja
pip install --upgrade "meson>=1.10" "meson-python>=0.15" pybind11
rm -rf "./build/cp311"
pip install -e . --no-build-isolation --config-settings=setup-args=-Dsystem-freetype=true --config-settings=setup-args=-Dsystem-qhull=true
pip install "numpy<2"