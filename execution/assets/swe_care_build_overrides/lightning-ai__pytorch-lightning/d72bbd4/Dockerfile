# syntax=docker/dockerfile:1.4
FROM python:3.8-slim

ARG REVIEWBENCH_APT_PROXY=http://host.docker.internal:3142
ARG REVIEWBENCH_PIP_INDEX_URL=http://host.docker.internal:3141/root/pypi/+simple
ARG REVIEWBENCH_PIP_EXTRA_INDEX_URL=
ARG REVIEWBENCH_PIP_TRUSTED_HOST=host.docker.internal

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    REVIEWBENCH_PIP_INDEX_URL=${REVIEWBENCH_PIP_INDEX_URL} \
    REVIEWBENCH_PIP_EXTRA_INDEX_URL=${REVIEWBENCH_PIP_EXTRA_INDEX_URL} \
    REVIEWBENCH_PIP_TRUSTED_HOST=${REVIEWBENCH_PIP_TRUSTED_HOST} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN APT_PROXY_FILE=/etc/apt/apt.conf.d/99reviewbench-proxy && \
    APT_UPDATE_LOG=/tmp/reviewbench-apt-update.log && \
    if [ -n "$REVIEWBENCH_APT_PROXY" ]; then \
      echo "Acquire::http::Proxy \"$REVIEWBENCH_APT_PROXY\";" > "$APT_PROXY_FILE" && \
      echo "Acquire::https::Proxy \"$REVIEWBENCH_APT_PROXY\";" >> "$APT_PROXY_FILE" ; \
    fi && \
    apt_status=0; \
    apt-get update > "$APT_UPDATE_LOG" 2>&1 || apt_status=$?; \
    cat "$APT_UPDATE_LOG" && \
    if [ "$apt_status" -ne 0 ] || ( [ -n "$REVIEWBENCH_APT_PROXY" ] && grep -Eiq "Could not resolve 'host\\.docker\\.internal'|Failed to fetch .*host\\.docker\\.internal|Temporary failure in name resolution|Name or service not known|Connection refused|ProxyError|timed out|Could not connect" "$APT_UPDATE_LOG" ); then \
      if [ -n "$REVIEWBENCH_APT_PROXY" ]; then \
        echo "Warning: apt proxy unavailable ($REVIEWBENCH_APT_PROXY); retrying direct." >&2 && \
        rm -f "$APT_PROXY_FILE" && \
        apt-get update; \
      else \
        exit "$apt_status"; \
      fi; \
    fi && \
    rm -f "$APT_UPDATE_LOG" && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    libffi-dev \
    libssl-dev \
    python3-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY claude /usr/local/bin/claude
COPY setup_repo.sh /tmp/setup_repo.sh
COPY install_deps.sh /tmp/install_deps.sh
COPY post_install.sh /tmp/post_install.sh

# dummy uv passthrough wrapper
RUN cat > /usr/local/bin/uv <<'EOF' && chmod +x /usr/local/bin/uv
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--version" ]]; then
  echo "uv 0.0.0-dummy"
  exit 0
fi

if [[ "${1:-}" != "pip" ]]; then
  echo "Dummy uv only supports: uv pip ..." >&2
  exit 2
fi
shift

python_bin=""
args=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --python)
      if [[ "$#" -lt 2 ]]; then
        echo "Missing value for --python" >&2
        exit 2
      fi
      python_bin="$2"
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${UV_CONSTRAINT:-}" && -z "${PIP_CONSTRAINT:-}" ]]; then
  export PIP_CONSTRAINT="$UV_CONSTRAINT"
fi

if [[ -n "$python_bin" ]]; then
  exec "$python_bin" -m pip "${args[@]}"
fi
exec pip "${args[@]}"
EOF

RUN --mount=type=bind,source=repo_seed,target=/tmp/repo_seed,readonly \
    chmod +x /usr/local/bin/uv /usr/local/bin/claude /tmp/setup_repo.sh /tmp/install_deps.sh /tmp/post_install.sh && \
    /tmp/setup_repo.sh && \
    /tmp/install_deps.sh && \
    /tmp/post_install.sh

WORKDIR /workspace
CMD ["/bin/bash"]
