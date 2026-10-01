#!/usr/bin/env bash
# AstroLaue 常駐ホットキー監視モードスクリプト (Linux)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PY_CMD=""
if [ -f "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PY_CMD="${SCRIPT_DIR}/.venv/bin/python"
elif [ -f "${HOME}/miniconda3/envs/py312/bin/python" ]; then
    PY_CMD="${HOME}/miniconda3/envs/py312/bin/python"
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

echo "========================================================"
echo "  AstroLaue: 常駐ホットキー監視モード (Resident Watcher)"
echo "  Python: ${PY_CMD}"
echo "========================================================"
echo "  キー操作: [Enter] でキャプチャ＆復元実行, [Q] で終了"
echo "========================================================"
echo ""

exec "${PY_CMD}" "${SCRIPT_DIR}/main.py" \
    --watch \
    --output "${SCRIPT_DIR}/results_live/" \
    --show-diagnostic \
    --export-transparent "$@"
