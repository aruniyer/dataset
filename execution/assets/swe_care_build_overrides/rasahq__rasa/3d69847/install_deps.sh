SRC_DIR="/workspace"
INSTANCE="rasahq__rasa-7797"
PY_BIN="python"
site_packages() {
  "$PY_BIN" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
}
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install "rasa==2.2.4"
"$PY_BIN" -m pip install --no-warn-conflicts "protobuf<3.21"
"$PY_BIN" -m pip uninstall -y rasa
SITE_PKGS="$(site_packages)"
printf "%s\n" "$SRC_DIR" > "$SITE_PKGS/zz_${INSTANCE}_src.pth"