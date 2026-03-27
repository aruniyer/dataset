pip install --upgrade pip setuptools wheel
pip install "numpy<2" "meson>=1.10" "meson-python>=0.15" pybind11 ninja
cat > "/root/mplsetup.cfg" <<'CFG'
[libs]
system_freetype = true
system_qhull = true
CFG
MPLSETUPCFG="/root/mplsetup.cfg" pip install -e .
pip install "numpy<2"