PY_BIN="python"
"$PY_BIN" -m pip install --upgrade pip setuptools wheel
"$PY_BIN" -m pip install "pip==23.2.1"
NO_WEB_UI=1 "$PY_BIN" -m pip install -e . --no-deps
"$PY_BIN" -m pip install --no-warn-conflicts "xoscar>=0.3.0" "sse-starlette>=1.6.5" "orjson" "pydantic==1.10.12" "openai==1.30.5" "requests" "Pillow" "huggingface_hub<1.0" "transformers" "nvidia-ml-py"
"$PY_BIN" -m pip install --no-warn-conflicts --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1"
"$PY_BIN" -m pip install --no-warn-conflicts "numpy<2"