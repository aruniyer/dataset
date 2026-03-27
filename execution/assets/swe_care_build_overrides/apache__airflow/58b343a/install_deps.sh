#!/usr/bin/env bash
set -euxo pipefail
python -m pip install --upgrade "pip<24.1" "setuptools<70" wheel
python - <<'PY'
import subprocess
import sys
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="google-re2-stub-"))
(tmp / "setup.py").write_text(
    "from setuptools import setup\n"
    "setup(name='google-re2', version='1.0.0', py_modules=['re2'])\n"
)
(tmp / "re2.py").write_text(
    "import re as _re\n"
    "from re import *  # noqa: F401,F403\n"
    "def _normalize(pattern):\n"
    "    if isinstance(pattern, str):\n"
    "        return pattern.replace(r'\\z', r'\\Z')\n"
    "    return pattern\n"
    "def compile(pattern, flags=0):\n"
    "    return _re.compile(_normalize(pattern), flags)\n"
    "def match(pattern, string, flags=0):\n"
    "    return _re.match(_normalize(pattern), string, flags)\n"
    "def search(pattern, string, flags=0):\n"
    "    return _re.search(_normalize(pattern), string, flags)\n"
    "def fullmatch(pattern, string, flags=0):\n"
    "    return _re.fullmatch(_normalize(pattern), string, flags)\n"
    "def findall(pattern, string, flags=0):\n"
    "    return _re.findall(_normalize(pattern), string, flags)\n"
    "def finditer(pattern, string, flags=0):\n"
    "    return _re.finditer(_normalize(pattern), string, flags)\n"
    "def split(pattern, string, maxsplit=0, flags=0):\n"
    "    return _re.split(_normalize(pattern), string, maxsplit, flags)\n"
    "def sub(pattern, repl, string, count=0, flags=0):\n"
    "    return _re.sub(_normalize(pattern), repl, string, count, flags)\n"
    "def subn(pattern, repl, string, count=0, flags=0):\n"
    "    return _re.subn(_normalize(pattern), repl, string, count, flags)\n"
)
subprocess.check_call([sys.executable, "-m", "pip", "install", str(tmp)])
print(f"Installed google-re2 stub from {tmp}")
PY
python -m pip install -e .
python -m pip install --no-warn-conflicts "PyYAML<7" "MarkupSafe<2.1" "Jinja2<3.1" "pendulum<3"
