#!/usr/bin/env bash
# AstroLaue 簡単起動スクリプト (Linux)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Python 実行環境の自動探索
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

# 引数がある場合 (画像パス直接指定)
if [ $# -ge 1 ]; then
    exec "${PY_CMD}" "${SCRIPT_DIR}/launcher.py" "$@"
fi

# 引数がない場合は対話ランチャー
exec "${PY_CMD}" "${SCRIPT_DIR}/launcher.py"
