#!/usr/bin/env bash
# AstroLaue Git Wrapper (Google Drive / rclone マウント環境用)
# rclone の --exclude **/.git/** に影響されず、ローカルSSDの ~/.git_repos/AstroLaue.git を通じて
# 通常の git コマンドを実行します。
# 例: ./git.sh status, ./git.sh diff, ./git.sh log 等

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_DIR_PATH="${HOME}/.git_repos/AstroLaue.git"

GIT_DIR="${GIT_DIR_PATH}" GIT_WORK_TREE="${PROJECT_DIR}" exec git "$@"
