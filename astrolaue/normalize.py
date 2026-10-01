"""モジュール 2: 中心検出 & 幾何規格化.

回折パターンの中心にある基準白色円（ダイレクトビーム／ビームストッパー中心）を
サブピクセル精度で検出し、キャンバス中心への幾何補正（アフィン変換）および
蛍光X線・非弾性散乱等の背景光除去を行います。
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_center_and_radius(
    image: np.ndarray,
    white_thresh: int = 240,
    search_roi_ratio: float = 0.5,
) -> Tuple[Tuple[float, float], float, float]:
    """中央の基準白色円の中心座標 (xc, yc) と半径 r0、および外周半径 R_outer を高精度検出します.

    画像モーメント (cv2.moments) および輪郭フィッティングにより、
    サブピクセル精度で中心座標を算出します。

    Args:
        image: 入力BGRまたはグレースケール画像
        white_thresh: 白丸検出の二値化閾値 (0-255)
        search_roi_ratio: 画像中央から探索する領域の割合 (0.5なら中央50%の領域)

    Returns:
        center: サブピクセル中心座標 (xc, yc)
        r0: 中心白丸の半径 (ピクセル)
        r_outer: 回折像外周円盤の半径 (ピクセル)

    Raises:
        ValueError: 中心白丸が検出できなかった場合
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    center_img_x = w / 2.0
    center_img_y = h / 2.0

    # 中央探索領域のマスク
    rx1 = int(round(w * (1.0 - search_roi_ratio) / 2.0))
    rx2 = int(round(w * (1.0 + search_roi_ratio) / 2.0))
    ry1 = int(round(h * (1.0 - search_roi_ratio) / 2.0))
    ry2 = int(round(h * (1.0 + search_roi_ratio) / 2.0))

    sub_gray = gray[ry1:ry2, rx1:rx2]

    # 適応的二値化または高輝度閾値処理
    # 中心白丸は極めて高輝度 (通常 > 240)
    _, sub_mask = cv2.threshold(sub_gray, white_thresh, 255, cv2.THRESH_BINARY)

    # 孤立点ノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(sub_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_cand = None
    min_dist_to_center = float("inf")

    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.6:
            continue

        # モーメントによるサブピクセル重心
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        sub_cx = M["m10"] / M["m00"]
        sub_cy = M["m01"] / M["m00"]

        # 元座標へ変換
        global_cx = sub_cx + rx1
        global_cy = sub_cy + ry1

        (circle_cx, circle_cy), rad = cv2.minEnclosingCircle(c)

        dist = np.hypot(global_cx - center_img_x, global_cy - center_img_y)
        if dist < min_dist_to_center:
            min_dist_to_center = dist
            best_cand = (global_cx, global_cy, float(rad))

    if best_cand is None:
        # フォールバック: Otsu二値化
        _, sub_mask_otsu = cv2.threshold(sub_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(sub_mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            sub_cx = M["m10"] / M["m00"]
            sub_cy = M["m01"] / M["m00"]
            global_cx = sub_cx + rx1
            global_cy = sub_cy + ry1
            _, rad = cv2.minEnclosingCircle(c)
            dist = np.hypot(global_cx - center_img_x, global_cy - center_img_y)
            if dist < min_dist_to_center:
                min_dist_to_center = dist
                best_cand = (global_cx, global_cy, float(rad))

    if best_cand is None:
        raise ValueError("中心白色円を検出できませんでした。入力画像を確認してください。")

    xc, yc, r0 = best_cand
    logger.info("中心白色円を検出: xc=%.3f, yc=%.3f, r0=%.2f", xc, yc, r0)

    # 外周円盤半径 (R_outer) の推定 (動径方向の輝度プロファイル急変位置)
    max_search_r = min(xc, yc, w - xc, h - yc)
    if max_search_r > r0 * 3:
        radii = np.arange(int(round(r0 * 2)), int(round(max_search_r)))
        # 16方向のサンプリングで平均プロファイルを計測
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        r_grid, th_grid = np.meshgrid(radii, angles)
        sample_x = xc + r_grid * np.cos(th_grid)
        sample_y = yc + r_grid * np.sin(th_grid)
        sampled = cv2.remap(
            gray.astype(np.float32),
            sample_x.astype(np.float32),
            sample_y.astype(np.float32),
            cv2.INTER_LINEAR,
        )
        radial_mean = np.mean(sampled, axis=0)
        # 外周境界（暗い円盤から明るいGUI背景へ急上昇する点）の勾配ピーク
        diff = np.diff(radial_mean)
        if len(diff) > 0 and np.max(diff) > 5.0:
            outer_idx = np.argmax(diff)
            r_outer = float(radii[outer_idx])
        else:
            r_outer = float(r0 * 8.5)
    else:
        r_outer = float(r0 * 8.5)

    logger.info("外周円盤半径を推定: r_outer=%.2f", r_outer)
    return (xc, yc), r0, r_outer


def normalize_geometry(
    image: np.ndarray,
    center: Tuple[float, float],
    output_size: int = 1024,
    scale: float = 1.0,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """検出した中心 (xc, yc) が出力キャンバス中心 (N/2, N/2) に一致するようアフィン幾何変換します.

    Args:
        image: 入力画像 (H, W) または (H, W, C)
        center: 検出された中心座標 (xc, yc)
        output_size: 出力正方形キャンバスサイズ (N x N ピクセル)
        scale: スケーリング倍率 (1.0で等倍)

    Returns:
        normalized: 幾何規格化された正方形画像 (output_size x output_size)
        new_center: 変換後の中心座標 (output_size/2, output_size/2)
    """
    xc, yc = center
    target_c = output_size / 2.0

    # 変換行列:
    # 1. (xc, yc) を原点へ平行移動
    # 2. scale を適用
    # 3. (target_c, target_c) へ平行移動
    # x' = scale * (x - xc) + target_c = scale * x + (target_c - scale * xc)
    # y' = scale * (y - yc) + target_c = scale * y + (target_c - scale * yc)
    tx = target_c - scale * xc
    ty = target_c - scale * yc

    M = np.array([
        [scale, 0.0, tx],
        [0.0, scale, ty],
    ], dtype=np.float32)

    normalized = cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    new_center = (target_c, target_c)
    logger.info(
        "幾何規格化完了: 元中心=(%.2f, %.2f) -> 新中心=(%.2f, %.2f), 出力サイズ=%dx%d",
        xc, yc, target_c, target_c, output_size, output_size,
    )
    return normalized, new_center


def subtract_background(
    image: np.ndarray,
    method: str = "tophat",
    kernel_size: int = 51,
) -> np.ndarray:
    """モルフォロジー演算 (Top-Hat) またはローパス残差により背景散乱・蛍光X線を除去します.

    走査ブラー長（約15-25px）に対応した大口径カーネル（デフォルト 51px）により、
    外周部に伸びる円弧状スポットの削ぎ落としを防ぎ、スポット状の回折斑点をシャープに残して
    広域の滑らかなバックグラウンドを平坦化します。

    Args:
        image: グレースケール画像 (float32 [0, 1] または uint8 [0, 255])
        method: 除去アルゴリズム ('tophat' または 'gaussian_residual')
        kernel_size: フィルタサイズ (奇数、デフォルト: 51)

    Returns:
        平坦化・背景減算された画像 (float32 [0.0, 1.0])
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # float32 正規化
    if gray.dtype != np.float32:
        gray_f = gray.astype(np.float32) / 255.0
    else:
        gray_f = gray.copy()
        if gray_f.max() > 1.0:
            gray_f /= 255.0

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if method == "tophat":
        # 円形構造要素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # uint8 に変換して OpenCV Top-Hat 演算 (高速かつ高精度)
        gray_u8 = (np.clip(gray_f, 0.0, 1.0) * 255.0).astype(np.uint8)
        tophat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel)
        result = tophat.astype(np.float32) / 255.0
    elif method == "gaussian_residual":
        sigma = k / 3.0
        blurred = cv2.GaussianBlur(gray_f, (k, k), sigma)
        result = np.maximum(0.0, gray_f - blurred)
    elif method in ("photutils", "sextractor"):
        try:
            from photutils.background import Background2D, MedianBackground
            from astropy.stats import SigmaClip

            box_s = max(k, 16)
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MedianBackground()
            bkg = Background2D(
                gray_f,
                box_size=(box_s, box_s),
                filter_size=(3, 3),
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator,
            )
            result = np.maximum(0.0, gray_f - bkg.background.astype(np.float32))
        except Exception as e:
            logger.warning("photutils Background2D 実行エラーのため tophat にフォールバックします: %s", e)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            gray_u8 = (np.clip(gray_f, 0.0, 1.0) * 255.0).astype(np.uint8)
            tophat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel)
            result = tophat.astype(np.float32) / 255.0
    else:
        raise ValueError(f"未知の背景減算手法です: {method}")

    # 背景分散 sigma_MAD に基づく適応的ノイズゲート
    # 微弱ピークを固定値引き算で消去せず、微小ノイズ揺らぎのみを滑らかに抑制
    pos_vals = result[result > 0.001]
    if len(pos_vals) > 100:
        med = float(np.median(pos_vals))
        mad = float(np.median(np.abs(pos_vals - med)))
        sigma_mad = 1.4826 * mad
        # 背景レベルの微小残差ゲート閾値 (微弱ピークの足元を切断しないよう控えめに設定)
        gate_thresh = max(0.5 * sigma_mad, 0.002)
        mask_low = (result > 0.0) & (result < gate_thresh)
        result[mask_low] = result[mask_low] * (result[mask_low] / gate_thresh)

        max_v = float(result.max())
        if max_v > 0.0:
            result = np.clip(result / max_v, 0.0, 1.0)

    return result
