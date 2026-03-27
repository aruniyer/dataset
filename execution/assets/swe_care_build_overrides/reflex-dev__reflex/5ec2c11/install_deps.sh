PY_BIN="python"
SRC_DIR="/workspace"
INSTANCE="reflex-dev__reflex-4406"
site_packages() {
  "$PY_BIN" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
}
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
if ! "$PY_BIN" -m pip install -e "$SRC_DIR"; then
  "$PY_BIN" -m pip install "$SRC_DIR"
  "$PY_BIN" -m pip uninstall -y reflex
  SITE_PKGS="$(site_packages)"
  printf "%s\n" "$SRC_DIR" > "$SITE_PKGS/zz_${INSTANCE}_src.pth"
fi
"$PY_BIN" -m pip install --no-warn-conflicts "pydantic==2.10.6" "sqlmodel==0.0.22"