PY_BIN="python"
SRC_DIR="/workspace"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
if [ ! -f "$SRC_DIR/setup.py" ] && [ ! -f "$SRC_DIR/pyproject.toml" ]; then
  cat > "$SRC_DIR/setup.py" <<'PYSETUP'
from setuptools import setup, find_packages

setup(
    name="memgpt",
    version="0.0.0",
    packages=find_packages(),
)
PYSETUP
fi
"$PY_BIN" -m pip install -r "$SRC_DIR/requirements.txt"
"$PY_BIN" -m pip install -e "$SRC_DIR" --no-deps