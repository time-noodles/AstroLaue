"""モジュール 1: 画面キャプチャ & 関心領域(ROI)自動抽出.

静止画ファイルからの読み込み、およびデスクトップ画面/特定ウィンドウからの
ライブキャプチャをサポートし、スクリーンショット全体から回折パターン領域を
高精度に自動検出・切り出します。

Windows環境では ctypes による標準Win32 APIを駆使し、
複数のウィンドウが立ち上がっている状態でも対象ウィンドウを自動特定・最前面化して
確実にキャプチャします。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(source_path: str | Path) -> np.ndarray:
    """指定された画像パスから画像を読み込み、BGRまたはグレースケールとして返します.

    日本語パス等にも対応するため、np.fromfile + cv2.imdecode を使用します。

    Args:
        source_path: 画像ファイルのパス

    Returns:
        読み込まれた画像配列 (BGR形式またはグレースケール)

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: 画像のデコードに失敗した場合
    """
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {path}")

    # 日本語パス対応の読み込み
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"画像のデコードに失敗しました: {path}")

    # アルファチャンネルがある場合はRGBに合成または除去
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def _find_windows_win32(title_candidates: List[str]) -> List[Dict[str, Any]]:
    """Windows API (ctypes) を用いてタイトルに部分一致する可視ウィンドウを探索します (Windows専用)."""
    if sys.platform != "win32":
        return []

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    results: List[Dict[str, Any]] = []

    def enum_windows_proc(hwnd: int, lParam: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value
                for cand in title_candidates:
                    if cand.lower() in title.lower():
                        rect = wintypes.RECT()
                        user32.GetWindowRect(hwnd, ctypes.byref(rect))
                        w = rect.right - rect.left
                        h = rect.bottom - rect.top
                        if w > 80 and h > 80:  # 小さすぎるアイコン等は除外
                            results.append({
                                "hwnd": hwnd,
                                "title": title,
                                "matched_pattern": cand,
                                "left": rect.left,
                                "top": rect.top,
                                "width": w,
                                "height": h,
                            })
                            break
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_windows_proc), 0)
    return results


def _activate_window_win32(hwnd: int) -> None:
    """指定されたウィンドウを最前面に表示してアクティブ化します (Windows専用)."""
    if sys.platform != "win32":
        return

    import ctypes
    user32 = ctypes.windll.user32

    SW_RESTORE = 9
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    # ウィンドウの再描画と最前面遷移を待機
    time.sleep(0.2)


def _capture_window_printwindow_win32(hwnd: int, w: int, h: int) -> Optional[np.ndarray]:
    """PrintWindow API により背面に隠れていてもウィンドウの描画内容を直接取得します (Windows専用)."""
    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        return None

    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    gdi32.SelectObject(mem_dc, bitmap)

    # PW_RENDERFULLCONTENT = 2 (Windows 8.1以降の高精度描画)
    PW_RENDERFULLCONTENT = 2
    success = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
    if not success:
        # フォールバック
        success = user32.PrintWindow(hwnd, mem_dc, 0)

    frame = None
    if success:
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h  # 上から下へ
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0

        buffer = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(mem_dc, bitmap, 0, h, buffer, ctypes.byref(bmi), 0)
        img_arr = np.frombuffer(buffer, dtype=np.uint8).reshape((h, w, 4))
        frame = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)

    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)

    return frame


def _get_foreground_window_win32() -> Optional[Dict[str, Any]]:
    """現在アクティブな最前面ウィンドウの情報を取得します (Windows専用)."""
    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None

    length = user32.GetWindowTextLengthW(hwnd)
    title = ""
    if length > 0:
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value

    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top

    # コンソール自身（cmd, powershell, terminal, python等）か判定
    console_keywords = ["cmd.exe", "powershell", "terminal", "astrolaue", "python"]
    is_console = any(k in title.lower() for k in console_keywords)

    return {
        "hwnd": hwnd,
        "title": title,
        "is_console": is_console,
        "left": rect.left,
        "top": rect.top,
        "width": w,
        "height": h,
    }


def capture_window(
    window_title_pattern: Optional[str] = None,
    monitor_index: int = 1,
    target_mode: str = "auto",
) -> np.ndarray:
    """デスクトップ画面または最前面ウィンドウから画像を取得します.

    特定のアプリケーションに固定されず、
    - ユーザーが手前に表示・整理したウィンドウ（測定ソフト、画像ビューア、ブラウザ等）
    - またはデスクトップ画面全体
    から画像を直接取得し、ROI自動抽出に引き渡します。

    Args:
        window_title_pattern: 特定のタイトルを明示的に検索・指定する場合 (省略時はアプリ名不問)
        monitor_index: mssで使用するモニタ番号 (1: プライマリモニタ, 0: 全モニタ)
        target_mode: 'auto' (アクティブウィンドウ優先、コンソールなら画面全体),
                     'screen' (画面全体/指定モニタ),
                     'active' (アクティブウィンドウ)

    Returns:
        キャプチャされたBGR画像配列
    """
    try:
        import mss
    except ImportError as e:
        raise ImportError("ライブキャプチャには 'mss' パッケージが必要です: pip install mss") from e

    target_box = None

    # 1. 特定のウィンドウタイトルが明示指定された場合 (オプション)
    if window_title_pattern and sys.platform == "win32":
        try:
            win_list = _find_windows_win32([window_title_pattern])
            if win_list:
                best_win = win_list[0]
                target_hwnd = best_win["hwnd"]
                target_box = {
                    "left": best_win["left"],
                    "top": best_win["top"],
                    "width": best_win["width"],
                    "height": best_win["height"],
                }
                logger.info(
                    "指定タイトルウィンドウを検出: '%s' (HWND=0x%x, %dx%d)",
                    best_win["title"], target_hwnd, best_win["width"], best_win["height"],
                )
                _activate_window_win32(target_hwnd)
                direct_frame = _capture_window_printwindow_win32(
                    target_hwnd, best_win["width"], best_win["height"]
                )
                if direct_frame is not None and direct_frame.mean() > 5.0:
                    logger.info("PrintWindow APIにより直接キャプチャ成功")
                    return direct_frame
        except Exception as ex:
            logger.warning("タイトル指定ウィンドウ特定でエラー: %s", ex)

    # 2. アプリ不問モード: アクティブウィンドウまたは画面全体
    if target_box is None and target_mode in ("auto", "active") and sys.platform == "win32":
        try:
            fg = _get_foreground_window_win32()
            if fg and not fg["is_console"] and fg["width"] > 150 and fg["height"] > 150:
                logger.info(
                    "手前のアクティブウィンドウをキャプチャ対象とします: '%s' (%dx%d at %d,%d)",
                    fg["title"], fg["width"], fg["height"], fg["left"], fg["top"],
                )
                direct_frame = _capture_window_printwindow_win32(
                    fg["hwnd"], fg["width"], fg["height"]
                )
                if direct_frame is not None and direct_frame.mean() > 5.0:
                    logger.info("PrintWindow APIによりアクティブウィンドウから直接キャプチャ成功")
                    return direct_frame
                target_box = {
                    "left": fg["left"],
                    "top": fg["top"],
                    "width": fg["width"],
                    "height": fg["height"],
                }
            elif fg and fg["is_console"]:
                logger.info("コンソールウィンドウがアクティブなため、デスクトップ画面全体をキャプチャします")
        except Exception as ex:
            logger.debug("アクティブウィンドウ情報取得エラー: %s", ex)

    # 3. mss による領域または画面全体のキャプチャ
    with mss.mss() as sct:
        if target_box is not None:
            logger.info("ウィンドウ矩形領域をキャプチャします: %s", target_box)
            raw = sct.grab(target_box)
        else:
            # 画面全体 (指定モニタまたはプライマリモニタ)
            idx = monitor_index if 0 <= monitor_index < len(sct.monitors) else 1
            mon = sct.monitors[idx]
            logger.info("デスクトップ画面をキャプチャします (モニタ %d: %dx%d)", idx, mon["width"], mon["height"])
            raw = sct.grab(mon)

        frame = np.array(raw)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame


def extract_diffraction_roi(
    image: np.ndarray,
    min_white_radius: float = 6.0,
    max_white_radius: float = 80.0,
    white_thresh: int = 235,
    outer_radius_ratio: float = 8.5,
    margin_ratio: float = 1.15,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """スクリーンショット画像から回折パターン領域（円盤＋中央白丸）を自動検出して切り出します.

    複数ウィンドウが存在する場合でも、
    「高輝度白丸」とその周囲の「暗い回折円盤コントラスト」を評価して
    目的の回折像を確実に同定します。

    Args:
        image: 入力BGRまたはグレースケール画像
        min_white_radius: 中心白丸の最小半径ピクセル
        max_white_radius: 中心白丸の最大半径ピクセル
        white_thresh: 白丸検出の二値化閾値 (0-255)
        outer_radius_ratio: 白丸半径に対する外周回折円盤の典型的な半径倍率 (約8.0〜9.0)
        margin_ratio: 外周円盤に対する切り出しマージン倍率

    Returns:
        cropped: 切り出された正方形画像 (BGRまたはGray)
        bbox: 切り出し矩形 (x, y, w, h)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # 既に切り出された正方形回折像かチェック
    aspect = w / max(h, 1)
    if 0.9 <= aspect <= 1.1 and w < 1200:
        center_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        if np.any(center_roi >= white_thresh):
            logger.info("入力画像は既に切り出された回折像と判定されました (%dx%d)", w, h)
            return image.copy(), (0, 0, w, h)

    # 白丸領域の二値化 (暗い円盤内の白丸を拾うため RETR_LIST)
    _, mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_cand = None
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0 or area <= 0:
            continue

        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
        (cx, cy), rad = cv2.minEnclosingCircle(c)

        if not (min_white_radius <= rad <= max_white_radius):
            continue

        # 真円度基準 (ビームストッパー白丸は真円度が高い)
        if circularity < 0.55:
            continue

        cx_i, cy_i = int(round(cx)), int(round(cy))
        r_inner = int(round(1.4 * rad))
        r_outer = int(round(3.8 * rad))

        # 画像端のマージンチェック
        if cx_i - r_outer < 0 or cx_i + r_outer >= w or cy_i - r_outer < 0 or cy_i + r_outer >= h:
            continue

        # 周囲の環状領域（回折円盤）のサンプリング
        sample_y, sample_x = np.ogrid[cy_i - r_outer : cy_i + r_outer, cx_i - r_outer : cx_i + r_outer]
        dists = np.hypot(sample_x - cx_i, sample_y - cy_i)
        ring_mask = (dists >= r_inner) & (dists <= r_outer)
        ring_pixels = gray[cy_i - r_outer : cy_i + r_outer, cx_i - r_outer : cx_i + r_outer][ring_mask]

        if len(ring_pixels) == 0:
            continue

        ring_mean = float(np.mean(ring_pixels))
        # 回折パターンは暗い (通常 20〜70)
        contrast = 255.0 - ring_mean

        # 暗い円盤であるほどスコアが高くなる
        score = circularity * 3.0 + (contrast / 30.0)
        # もし周囲が極端に明るい (ring_mean > 150) 場合は通常GUIのボタンや白UIなので除外
        if ring_mean > 160:
            continue

        if score > best_score:
            best_score = score
            best_cand = (cx, cy, rad)

    if best_cand is None:
        # フォールバック: Otsu二値化
        logger.warning("通常二値化で中心白丸が見つからず、Otsu二値化で再探索します")
        _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_otsu, _ = cv2.findContours(mask_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_otsu:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0 or area <= 0:
                continue
            circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
            (cx, cy), rad = cv2.minEnclosingCircle(c)
            if min_white_radius <= rad <= max_white_radius and circularity >= 0.5:
                best_cand = (cx, cy, rad)
                break

    if best_cand is None:
        logger.warning("中心白丸が自動検出できませんでした。画像全体を返します。")
        return image.copy(), (0, 0, w, h)

    cx, cy, r_white = best_cand
    logger.info("回折像中心白丸を検出: center=(%.1f, %.1f), r=%.1f", cx, cy, r_white)

    # 外径推定 (約8.0〜9.0倍) とマージン
    outer_radius = r_white * outer_radius_ratio
    crop_radius = int(round(outer_radius * margin_ratio))

    x1 = max(0, int(round(cx - crop_radius)))
    y1 = max(0, int(round(cy - crop_radius)))
    x2 = min(w, int(round(cx + crop_radius)))
    y2 = min(h, int(round(cy + crop_radius)))

    # 正方形化
    box_w = x2 - x1
    box_h = y2 - y1
    side = min(box_w, box_h)
    x2 = x1 + side
    y2 = y1 + side

    cropped = image[y1:y2, x1:x2].copy()
    bbox = (x1, y1, side, side)
    logger.info("回折像ROIを切り出しました: bbox=%s (size=%dx%d)", bbox, side, side)

    return cropped, bbox
