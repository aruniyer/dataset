#!/usr/bin/env bash
# Run batch test generation on the SWE-CARE test split.
#
# Usage:
#   ./run_testgen_full.sh                                           # Full test split
#   ./run_testgen_full.sh --repos-dir /data/repos --output-dir /data/results
#   ./run_testgen_full.sh --instances-file test_gen_improvement/dev_instances.txt
#   ./run_testgen_full.sh --no-use-docker                           # Disable Docker
#   WORKERS=8 MODEL=gpt-5.2 ./run_testgen_full.sh

set -euo pipefail

REPOS_DIR="${REPOS_DIR:-repos}"
OUTPUT_DIR="${OUTPUT_DIR:-results_testgen}"
WORKERS="${WORKERS:-4}"
MODEL="${MODEL:-gpt-5.2}"
INSTANCES_FILE="${INSTANCES_FILE:-}"
NO_USE_DOCKER="${NO_USE_DOCKER:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repos-dir)       REPOS_DIR="$2";       shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --workers)         WORKERS="$2";         shift 2 ;;
        --model)           MODEL="$2";           shift 2 ;;
        --instances-file)  INSTANCES_FILE="$2";  shift 2 ;;
        --no-use-docker)   NO_USE_DOCKER=1;      shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")"

EXTRA_ARGS=()
if [[ -n "$INSTANCES_FILE" ]]; then
    EXTRA_ARGS+=(--instances-file "$INSTANCES_FILE")
fi
if [[ -n "$NO_USE_DOCKER" ]]; then
    EXTRA_ARGS+=(--no-use-docker)
fi

exec uv run python run_batch_testgen.py \
    --split test \
    --workers "$WORKERS" \
    --repos-dir "$REPOS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    "${EXTRA_ARGS[@]}"
