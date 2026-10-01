#!/usr/bin/env bash
# AstroLaue ライブ画面キャプチャ＆復元スクリプト (Linux)

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

TITLE_ARG=""
if [ $# -ge 1 ]; then
    TITLE_ARG="--window-title $1"
fi

echo "========================================================"
echo "  AstroLaue: 画面キャプチャ＆復元 (特定アプリに非依存)"
echo "========================================================"

exec "${PY_CMD}" "${SCRIPT_DIR}/main.py" \
    --live \
    ${TITLE_ARG} \
    --output "${SCRIPT_DIR}/results_live/" \
    --show-diagnostic \
    --export-transparent "$@"
