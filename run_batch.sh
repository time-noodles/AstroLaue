#!/usr/bin/env bash
# AstroLaue 一括バッチ処理スクリプト (Linux)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PY_CMD=""
if [ -f "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PY_CMD="${SCRIPT_DIR}/.venv/bin/python"
elif [ -f "${HOME}/.venvs/astrolaue/bin/python" ]; then
    PY_CMD="${HOME}/.venvs/astrolaue/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PY_CMD="python"
fi

if [ -z "${PY_CMD}" ]; then
    echo "❌ エラー: Python が見つかりませんでした。"
    exit 1
fi

TARGET_DIR="share"
if [ $# -ge 1 ]; then
    TARGET_DIR="$1"
fi

echo "========================================================"
echo "  AstroLaue: 一括バッチ復元を実行します"
echo "  対象ディレクトリ: ${TARGET_DIR}"
echo "========================================================"

exec "${PY_CMD}" "${SCRIPT_DIR}/main.py" \
    --batch-dir "${TARGET_DIR}" \
    --output "${SCRIPT_DIR}/results_batch/" \
    --export-transparent \
    --validate-physics
