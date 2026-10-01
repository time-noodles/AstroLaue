"""OS クリップボード操作ユーティリティ.

Windows および Linux 環境において、復元画像 (PNG) をクリップボードへ自動コピーします。
Windows では ctypes による Win32 API (CF_DIB および PNG クリップボード形式) を直接利用し、
追加パッケージ不要で Office, Slack, ペイント等の各種アプリへの貼り付けに対応します。
"""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def copy_image_to_clipboard(image_path: str | Path) -> bool:
    """指定された画像ファイルを OS のクリップボードにコピーします.

    Args:
        image_path: 画像ファイルパス (PNG等)

    Returns:
        bool: コピーに成功した場合 True, 失敗した場合 False
    """
    path = Path(image_path)
    if not path.is_file():
        logger.warning("クリップボードコピー対象のファイルが存在しません: %s", path)
        return False

    os_name = platform.system()

    if os_name == "Windows":
        return _copy_to_windows_clipboard(path)
    elif os_name == "Linux":
        return _copy_to_linux_clipboard(path)
    elif os_name == "Darwin":
        return _copy_to_macos_clipboard(path)
    else:
        logger.info("未対応のOSです: %s", os_name)
        return False


def _copy_to_windows_clipboard(path: Path) -> bool:
    """Windows Win32 API を用いて画像をクリップボードに転送します."""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        # 定数定義
        CF_DIB = 8
        GMEM_MOVEABLE = 0x0002

        # 1. 生の PNG バイト列を読み込み
        png_bytes = path.read_bytes()

        # 2. DIB データの作成 (OpenCV で BMP エンコードし、先頭 14 バイトの BITMAPFILEHEADER を除外)
        # 画像を BGRA または BGR で読み込み
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("画像の読み込みに失敗しました: %s", path)
            return False

        # BMP 形式に変換
        # アルファチャンネルがある場合、白背景に合成した BGR 画像で DIB を作成すると
        # DIB 対応のペイントや古いアプリでも黒潰れせず自然に貼り付け可能
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            bgr = img[:, :, 0:3].astype(np.float32)
            white_bg = np.ones_like(bgr) * 255.0
            composite = np.clip(bgr * alpha + white_bg * (1.0 - alpha), 0, 255).astype(np.uint8)
            success, bmp_buf = cv2.imencode(".bmp", composite)
        else:
            success, bmp_buf = cv2.imencode(".bmp", img)

        dib_bytes = bmp_buf[14:].tobytes() if success else None

        # クリップボードを開く
        # hWnd=0 (現在のタスクに関連付け)
        if not user32.OpenClipboard(None):
            logger.warning("クリップボードのオープンに失敗しました。")
            return False

        try:
            user32.EmptyClipboard()

            # A. CF_DIB を登録 (Office, ペイント, 一般アプリ向け)
            if dib_bytes:
                h_dib = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(dib_bytes))
                if h_dib:
                    p_dib = kernel32.GlobalLock(h_dib)
                    if p_dib:
                        ctypes.memmove(p_dib, dib_bytes, len(dib_bytes))
                        kernel32.GlobalUnlock(h_dib)
                        user32.SetClipboardData(CF_DIB, h_dib)

            # B. "PNG" カスタムフォーマットを登録 (Slack, Teams, Webブラウザ, 高度アプリ向け透過PNG)
            png_format = user32.RegisterClipboardFormatW("PNG")
            if png_format:
                h_png = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(png_bytes))
                if h_png:
                    p_png = kernel32.GlobalLock(h_png)
                    if p_png:
                        ctypes.memmove(p_png, png_bytes, len(png_bytes))
                        kernel32.GlobalUnlock(h_png)
                        user32.SetClipboardData(png_format, h_png)

            logger.info("画像をクリップボードに正常にコピーしました: %s", path.name)
            return True

        finally:
            user32.CloseClipboard()

    except Exception as e:
        logger.warning("Windowsクリップボードへのコピーに失敗しました: %s", e)
        return False


def _copy_to_linux_clipboard(path: Path) -> bool:
    """Linux 環境で xclip または wl-copy を使って画像をクリップボードに格納します."""
    try:
        # wl-copy (Wayland)
        res = subprocess.run(
            ["which", "wl-copy"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if res.returncode == 0:
            subprocess.run(
                ["wl-copy", "--type", "image/png"],
                input=path.read_bytes(),
                check=False,
            )
            logger.info("wl-copy を使用してクリップボードにコピーしました: %s", path.name)
            return True

        # xclip (X11)
        res = subprocess.run(
            ["which", "xclip"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if res.returncode == 0:
            subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-i", str(path)],
                check=False,
            )
            logger.info("xclip を使用してクリップボードにコピーしました: %s", path.name)
            return True

        logger.debug("Linuxクリップボードツール (wl-copy, xclip) が見つかりませんでした。")
        return False
    except Exception as e:
        logger.debug("Linuxクリップボードコピー処理例外: %s", e)
        return False


def _copy_to_macos_clipboard(path: Path) -> bool:
    """macOS pbcopy を使って画像をクリップボードに格納します."""
    try:
        script = f'set the clipboard to (read (POSIX file "{path.resolve()}") as «class PNGf»)'
        subprocess.run(["osascript", "-e", script], check=False)
        return True
    except Exception:
        return False
