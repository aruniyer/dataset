PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install "dgl==1.1.3"
SITE_PKGS="$($PY_BIN - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
LIB_FROM_WHEEL="$SITE_PKGS/dgl/libdgl.so"
if [ ! -f "$LIB_FROM_WHEEL" ]; then
  echo "libdgl.so not found at $LIB_FROM_WHEEL" >&2
  exit 1
fi
cp -f "$LIB_FROM_WHEEL" "./python/dgl/libdgl.so"
"$PY_BIN" -m pip install -e "./python" --no-build-isolation
"$PY_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2"
"$PY_BIN" -m pip install "numpy<2"