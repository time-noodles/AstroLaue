"""モジュール 3: 極座標変換.

直交座標系 (x, y) と極座標系 (r, theta) の高精度相互変換（バイキュービック補間）
および角度方向の境界アーティファクトを抑える循環パディング (Circular Padding) を提供します。
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def cartesian_to_polar(
    image: np.ndarray,
    center: Tuple[float, float],
    r_range: Tuple[float, float],
    n_r: int = 512,
    n_theta: int = 1440,
    pad_angle_pix: int = 0,
) -> np.ndarray:
    """直交座標系 (x, y) の画像を極座標系 (r, theta) へ展開します.

    展開後の画像形状:
        - 縦軸 (行): 半径 r in [r_min, r_max] (0: r_min, n_r-1: r_max)
        - 横軸 (列): 方位角 theta in [0, 2*pi) (0: 0 rad, n_theta-1: 2*pi - dtheta)
        ※ pad_angle_pix > 0 の場合、横軸の両端に角度循環パディングが付与されます。

    Args:
        image: 入力直交画像 (H, W) または (H, W, C)
        center: 中心座標 (xc, yc)
        r_range: (r_min, r_max) 半径範囲。中央のビームストッパー飽和を除外。
        n_r: 動径方向のサンプリング点数 (行数)
        n_theta: 方位角方向のサンプリング点数 (列数)
        pad_angle_pix: 角度方向に付与する循環パディング幅 (ピクセル)

    Returns:
        polar_image: 極座標展開画像。形状 (n_r, n_theta + 2 * pad_angle_pix)
    """
    xc, yc = center
    r_min, r_max = r_range

    r_vals = np.linspace(r_min, r_max, n_r, endpoint=True, dtype=np.float32)
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float32)

    # パディング考慮
    if pad_angle_pix > 0:
        d_theta = 2.0 * np.pi / n_theta
        left_theta = np.linspace(-pad_angle_pix * d_theta, -d_theta, pad_angle_pix, dtype=np.float32)
        right_theta = np.linspace(2.0 * np.pi, 2.0 * np.pi + (pad_angle_pix - 1) * d_theta, pad_angle_pix, dtype=np.float32)
        theta_all = np.concatenate([left_theta, theta_vals, right_theta])
    else:
        theta_all = theta_vals

    R, Theta = np.meshgrid(r_vals, theta_all, indexing="ij")
    map_x = xc + R * np.cos(Theta)
    map_y = yc + R * np.sin(Theta)

    polar_img = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )

    logger.info(
        "極座標展開完了: r=[%.1f, %.1f] (%d点), theta=[0, 2pi) (%d点), 出力shape=%s",
        r_min, r_max, n_r, n_theta, polar_img.shape,
    )
    return polar_img


def polar_to_cartesian(
    polar_image: np.ndarray,
    output_shape: Tuple[int, int],
    center: Tuple[float, float],
    r_range: Tuple[float, float],
    pad_angle_pix: int = 0,
) -> np.ndarray:
    """極座標系 (r, theta) の画像を直交座標系 (x, y) へ逆変換します.

    内部で角度軸 (theta) に循環パディング (Circular Wrap Padding) を施すことで、
    theta=0 (360度) 境界での補間ゼロ値混入や不連続線・モアレアーティファクトを完全に解消します。
    また、バイキュービック補間によるアンダーシュート（負値）を排除してノイズ増加を防ぎます。

    Args:
        polar_image: 極座標画像。形状 (n_r, n_theta + 2 * pad_angle_pix)
        output_shape: 出力画像の解像度 (H, W)
        center: 出力画像上の中心座標 (xc, yc)
        r_range: (r_min, r_max) 半径範囲
        pad_angle_pix: 極座標画像に含まれる角度方向の循環パディング幅

    Returns:
        cartesian_image: 逆変換された直交座標画像 (H, W)
    """
    h_out, w_out = output_shape
    xc, yc = center
    r_min, r_max = r_range

    # 外部パディングを除去してクリーンなコア画像を取得
    if pad_angle_pix > 0:
        polar_core = polar_image[:, pad_angle_pix:-pad_angle_pix].copy()
    else:
        polar_core = polar_image.copy()

    # 非負化
    polar_core = np.maximum(polar_core, 0.0)
    n_r, n_theta = polar_core.shape[:2]

    # 境界補間ゼロ混入防止のための内部循環パディング (16 ピクセル)
    # これにより theta=0 や theta=2*pi 近傍でも真の周期連続性が保たれ、境界アーティファクトが消滅する
    wrap_pad = 16
    padded_polar = np.pad(polar_core, ((0, 0), (wrap_pad, wrap_pad)), mode="wrap")

    # 出力直交グリッド (X, Y)
    y_indices, x_indices = np.indices((h_out, w_out), dtype=np.float32)
    dx = x_indices - xc
    dy = y_indices - yc

    r_grid = np.hypot(dx, dy)
    theta_grid = np.arctan2(dy, dx)
    # theta を [0, 2*pi) の範囲へ正規化
    theta_grid = np.mod(theta_grid, 2.0 * np.pi)

    # 極座標画像のインデックスへのマッピング
    # r in [r_min, r_max] -> [0, n_r - 1]
    # theta in [0, 2*pi) -> [wrap_pad, wrap_pad + n_theta)
    r_idx_map = (r_grid - r_min) / max(r_max - r_min, 1e-4) * (n_r - 1)
    theta_idx_map = wrap_pad + (theta_grid / (2.0 * np.pi)) * n_theta

    # マスク: r_min <= r <= r_max の外側は無効
    valid_mask = (r_grid >= r_min) & (r_grid <= r_max)

    # 動径方向インデックスの安全クリップ
    map_x = theta_idx_map.astype(np.float32)
    map_y = np.clip(r_idx_map, 0, n_r - 1).astype(np.float32)

    cartesian_img = cv2.remap(
        padded_polar,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # バイキュービック補間の負のローブ (アンダーシュート) によるノイズを除去
    cartesian_img = np.maximum(cartesian_img, 0.0)

    # 有効範囲外をゼロマスク
    if len(cartesian_img.shape) == 3:
        cartesian_img[~valid_mask] = 0
    else:
        cartesian_img[~valid_mask] = 0.0

    logger.info("直交座標逆変換完了: 出力shape=%s, 有効領域比率=%.2f%%", output_shape, 100.0 * np.mean(valid_mask))
    return cartesian_img


def apply_circular_padding(polar_image: np.ndarray, pad_width: int) -> np.ndarray:
    """極座標画像の角度軸 (axis=1) に循環パディングを施します.

    Args:
        polar_image: 形状 (n_r, n_theta) の極座標画像
        pad_width: パディング幅 (ピクセル)

    Returns:
        形状 (n_r, n_theta + 2 * pad_width) のパディング済み画像
    """
    if pad_width <= 0:
        return polar_image
    return np.pad(polar_image, ((0, 0), (pad_width, pad_width)), mode="wrap")


def remove_circular_padding(padded_polar: np.ndarray, pad_width: int) -> np.ndarray:
    """循環パディングを除去して元の形状に戻します.

    Args:
        padded_polar: パディング済み極座標画像
        pad_width: パディング幅 (ピクセル)

    Returns:
        元の極座標画像
    """
    if pad_width <= 0:
        return padded_polar
    return padded_polar[:, pad_width:-pad_width]
