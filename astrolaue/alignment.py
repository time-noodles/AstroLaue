"""モジュール: 背面反射ラウエX線回折における結晶主軸自動推定 & 2軸ゴニオメーター補正角算出.

背面反射（Back-reflection）配置のラウエX線回折パターンから、
結晶の空間群や格子定数が完全に未知の状態であっても、同一晶帯（Zone）のスポット群が
グノモン投影面上で直線上に並ぶ幾何学的性質を利用して主要な晶帯軸（Zone Axis、例: (001)面法線等）を
自動同定し、検出器中心（X線入射中心）へ一致させるための2軸ゴニオメーター回転角度
(Δθx, Δθy) をサブ度（< 0.1°）の高精度かつリアルタイム（約 15〜35 ms）で推定します。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Literal, Union

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 共通基盤モジュールからの再利用（コード重複・スパゲティ化防止）
_pkg_root = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from astrolaue.capture import load_image
from astrolaue.normalize import detect_center_and_radius, subtract_background

logger = logging.getLogger(__name__)


@dataclass
class LaueAlignmentConfig:
    """背面反射ラウエ結晶方位自動アライメント設定パラメータ."""

    # 実験幾何・検出器パラメータ
    camera_length_mm: float = 27.0
    """試料―検出器間距離 L [mm] (通常 27〜100 mm、sLaue 実機幾何: 27.0 mm)."""

    pixel_size_mm: float = 0.05
    """検出器ピクセルサイズ [mm/pix] (例: 50 um = 0.05 mm)."""

    beam_center_xy: Tuple[float, float] = (0.0, 0.0)
    """検出器中心（入射X線軸）のピクセル座標 (xc, yc)。(0, 0) かつ auto_detect_center=True 時は自動同定."""

    auto_detect_center: bool = True
    """中心白色ビームストッパーの自動同定を試みるか (デフォルト: True)."""

    # ピーク検出パラメータ (画像入力時)
    min_sigma: float = 1.5
    """DoG/ブロブ検出の最小スケール (ピクセル単位)."""

    max_sigma: float = 8.0
    """DoG/ブロブ検出の最大スケール (ピクセル単位)."""

    threshold: float = 0.02
    """スポット検出の相対輝度閾値 (0.0〜1.0)."""

    tophat_kernel_size: int = 15
    """モルフォロジー背景除去のカーネルサイズ (奇数ピクセル)."""

    max_detected_spots: int = 150
    """検出する最大スポット数 (強度上位順)."""

    # 晶帯（直線）フィッティングパラメータ (グノモン空間)
    dist_thresh: float = 0.015
    """グノモン平面上での直線インライア距離閾値 (グノモン単位)."""

    min_inliers: int = 4
    """晶帯直線とみなす最小インライアスポット数."""

    r_decay: float = 0.35
    """原点からの距離減衰パラメータ (グノモン単位)."""

    max_zones: int = 5
    """抽出する主要晶帯直線の最大本数 K (デフォルト: 5)."""

    nms_angle_thresh_deg: float = 4.0
    """NMS (Non-Maximum Suppression) での直線法線角差閾値 [deg]."""

    nms_dist_thresh: float = 0.025
    """NMS での直線原点距離差閾値 (グノモン単位)."""

    # 主軸（交点）探索パラメータ
    min_intersection_angle_deg: float = 15.0
    """直線ペアの最小交差角 [deg] (平行に近いペアを除外)."""

    search_cone_half_angle_deg: float = 25.0
    """探索半角 (中心からの許容結晶傾き角) [deg]."""

    cluster_eps: float = 0.025
    """近接交点のクラスタリング許容距離 (グノモン単位)."""

    # ゴニオメーターステージパラメータ
    stage_order: Literal["XY", "YX"] = "XY"
    """ゴニオメーターの軸構成 ('XY': 下X上Y, 'YX': 下Y上X)."""

    sign_x: int = 1
    """X軸回転の符号係数 (+1 または -1)."""

    sign_y: int = 1
    """Y軸回転の符号係数 (+1 または -1、数学的標準: 1, sLaue実機: -1)."""


@dataclass
class ZoneLine:
    """グノモン空間で検出された晶帯直線 a*u + b*v + c = 0 (a^2 + b^2 = 1)."""

    line_id: int
    """晶帯識別番号 (1-indexed)."""

    a: float
    """直線方程式の係数 a (cos phi)."""

    b: float
    """直線方程式の係数 b (sin phi)."""

    c: float
    """直線方程式の係数 c (-d)."""

    score: float
    """晶帯直線の信頼度スコア (強度・距離重み付き和)."""

    num_inliers: int
    """直線上にあるインライアスポット数."""

    inlier_indices: List[int]
    """インライアスポットのインデックスリスト."""

    zone_axis_norm: Tuple[float, float, float]
    """対応する晶帯軸の3次元単位方向ベクトル (a, b, c)."""


@dataclass
class AxisCandidate:
    """推定された主軸（交点）候補."""

    rank: int
    """候補順位 (1: 最良候補, 2: 第2候補, 3: 第3候補)."""

    u: float
    """グノモン投影面座標 u."""

    v: float
    """グノモン投影面座標 v."""

    normal_vector: Tuple[float, float, float]
    """3次元単位面法線ベクトル (nx, ny, nz)."""

    delta_theta_x_deg: float
    """X軸まわりゴニオメーター必要回転補正角 Δθx [deg]."""

    delta_theta_y_deg: float
    """Y軸まわりゴニオメーター必要回転補正角 Δθy [deg]."""

    total_tilt_deg: float
    """中心軸（Z軸）からの全傾き角 [deg]."""

    score: float
    """主軸候補の総合信頼度スコア."""

    contributing_zones: List[int]
    """交差を構成する晶帯 line_id リスト."""

    detector_x_mm: float
    """検出器平面上の対応位置 X [mm]."""

    detector_y_mm: float
    """検出器平面上の対応位置 Y [mm]."""


@dataclass
class LaueAlignmentResult:
    """背面反射ラウエ自動アライメントの総合結果."""

    spots_xy_mm: np.ndarray
    """検出器スポット物理座標 (N, 2) [mm]."""

    intensities: np.ndarray
    """各スポットの強度 (N,)."""

    normals: np.ndarray
    """結晶面単位法線ベクトル (N, 3)."""

    gnomonic_uv: np.ndarray
    """グノモン投影面座標 (N, 2)."""

    zones: List[ZoneLine]
    """検出された主要晶帯直線リスト."""

    candidates: List[AxisCandidate]
    """検出された主軸候補リスト (上位最大3件)."""

    best_candidate: AxisCandidate
    """最良推定主軸 (第1候補)."""

    camera_length_mm: float
    """使用したカメラ長 L [mm]."""

    config: LaueAlignmentConfig
    """実行設定."""

    elapsed_ms: float = 0.0
    """総計算処理時間 [ms]."""

    @property
    def delta_theta_x_deg(self) -> float:
        """最良候補のX軸まわりゴニオメーター回転補正角 [deg]."""
        return self.best_candidate.delta_theta_x_deg

    @property
    def delta_theta_y_deg(self) -> float:
        """最良候補のY軸まわりゴニオメーター回転補正角 [deg]."""
        return self.best_candidate.delta_theta_y_deg

    @property
    def total_tilt_deg(self) -> float:
        """最良候補の中心軸からの全傾き角 [deg]."""
        return self.best_candidate.total_tilt_deg


# ============================================================================
# Step 1: スポット抽出（画像入力対応 & 物理座標変換）
# ============================================================================

def extract_laue_spots_from_image(
    image_input: Union[np.ndarray, str, Path],
    config: Optional[LaueAlignmentConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ラウエ回折画像から背景除去・ピーク検出を行い、検出器物理座標 (X_i, Y_i) [mm] と強度 I_i を抽出します.

    Args:
        image_input: 入力画像配列 (2D float32/uint8/uint16) または画像ファイルパス
        config: アライメント設定。省略時はデフォルト値を使用。

    Returns:
        spots_xy_mm: 検出器中心 (0, 0) を原点とするスポット物理座標配列 (N, 2) [mm]
        intensities: 各スポットのピーク強度配列 (N,) [正規化任意単位]
    """
    cfg = config or LaueAlignmentConfig()

    # 1. 画像読み込み (日本語パス対応・共通関数 load_image を再利用)
    if isinstance(image_input, (str, Path)):
        img = load_image(image_input)
    else:
        img = image_input.copy()

    # グレースケール化
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    xc, yc = cfg.beam_center_xy
    r0 = 0.0

    # 中心白丸（ダイレクトビーム穴・ビームストッパー）の自動検出
    if (xc == 0.0 and yc == 0.0) and cfg.auto_detect_center:
        try:
            (det_xc, det_yc), det_r0, _ = detect_center_and_radius(gray)
            xc, yc = det_xc, det_yc
            r0 = det_r0
            logger.info("中心ビームストッパーを自動同定しました: center=(%.1f, %.1f), r0=%.1f px", xc, yc, r0)
        except Exception:
            xc, yc = w / 2.0, h / 2.0
    elif xc == 0.0 and yc == 0.0:
        xc, yc = w / 2.0, h / 2.0

    # 2. 背景除去 (共通モジュール normalize.subtract_background を再利用)
    tophat = subtract_background(gray, method="tophat", kernel_size=cfg.tophat_kernel_size)

    # ビームストッパー円内（飽和アーティファクト）のマスク除外
    if r0 > 0.0:
        y_idx, x_idx = np.indices((h, w), dtype=np.float32)
        dist_from_c = np.hypot(x_idx - xc, y_idx - yc)
        tophat[dist_from_c <= r0 * 1.05] = 0.0

    # 3. 局所最大値探索 + サブピクセル重心計算
    thresh_val = float(np.clip(cfg.threshold, 0.005, 0.5))
    _, mask = cv2.threshold(tophat, thresh_val, 1.0, cv2.THRESH_BINARY)
    mask_u8 = (mask * 255.0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spot_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1.0:
            continue

        # モーメントによるサブピクセル重心計算
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx_sub = M["m10"] / M["m00"]
        cy_sub = M["m01"] / M["m00"]

        # ピーク強度
        xi = int(np.clip(round(cx_sub), 0, w - 1))
        yi = int(np.clip(round(cy_sub), 0, h - 1))
        peak_i = float(tophat[yi, xi])

        spot_candidates.append((cx_sub, cy_sub, peak_i))

    if not spot_candidates:
        logger.warning("スポットが検出されませんでした。閾値を調整してください。")
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    # 強度順にソートして上位を抽出
    spot_candidates.sort(key=lambda item: item[2], reverse=True)
    selected = spot_candidates[: cfg.max_detected_spots]

    # 物理座標系 (X, Y) [mm] への変換
    # X軸: 水平右方向 (+X = (x - xc) * pixel_size)
    # Y軸: 垂直上方向 (+Y = -(y - yc) * pixel_size, 画像座標系Y反転)
    pix_coords = np.array([[s[0], s[1]] for s in selected], dtype=np.float64)
    intensities = np.array([s[2] for s in selected], dtype=np.float64)

    X_mm = (pix_coords[:, 0] - xc) * cfg.pixel_size_mm
    Y_mm = -(pix_coords[:, 1] - yc) * cfg.pixel_size_mm
    spots_xy_mm = np.column_stack([X_mm, Y_mm])

    logger.info("画像から %d 個のラウエスポットを抽出しました (検出器中心=(%.1f, %.1f) px)", len(spots_xy_mm), xc, yc)
    return spots_xy_mm, intensities


# ============================================================================
# Step 2: 検出器座標から結晶面法線ベクトル n_i への幾何学的変換
# ============================================================================

def detector_to_normals(
    spots_xy_mm: np.ndarray,
    camera_length_mm: float,
) -> np.ndarray:
    """検出器上のスポット座標 (X_i, Y_i) [mm] を背面反射配置における結晶面単位法線ベクトル n_i に変換します.

    背面反射座標系:
        - 試料から検出器中心へ向かう方向を +Z
        - 入射X線単位ベクトル: s0 = (0, 0, -1)^T
        - 検出器平面: Z = L
        - 回折X線単位ベクトル: s_i = (X_i, Y_i, L) / sqrt(X_i^2 + Y_i^2 + L^2)
        - 格子面単位法線ベクトル: n_i = (s_i - s0) / ||s_i - s0||

    Args:
        spots_xy_mm: 検出器中心を原点とするスポット座標 (N, 2) [mm]
        camera_length_mm: 試料―検出器間距離 L [mm]

    Returns:
        normals: 各スポットに対応する結晶面単位法線ベクトル (N, 3) (nz > 0)
    """
    if len(spots_xy_mm) == 0:
        return np.empty((0, 3), dtype=np.float64)

    pts = np.asarray(spots_xy_mm, dtype=np.float64)
    X = pts[:, 0]
    Y = pts[:, 1]
    L = float(camera_length_mm)

    # 1. 回折単位ベクトル s_i = (sx, sy, sz)
    dist = np.sqrt(X ** 2 + Y ** 2 + L ** 2)
    sx = X / dist
    sy = Y / dist
    sz = L / dist

    # 2. 回折ベクトル Δs = s_i - s0
    # s0 = (0, 0, -1)^T なので s_i - s0 = (sx, sy, sz + 1.0)^T
    dx = sx
    dy = sy
    dz = sz + 1.0

    # 3. 単位法線ベクトル n_i = Δs / ||Δs||
    norm_len = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    nx = dx / norm_len
    ny = dy / norm_len
    nz = dz / norm_len

    normals = np.column_stack([nx, ny, nz])
    return normals


# ============================================================================
# Step 3: グノモン投影 (Gnomonic Projection) と晶帯（直線）検出・主軸推定
# ============================================================================

def normals_to_gnomonic(normals: np.ndarray) -> np.ndarray:
    """結晶面単位法線ベクトル n_i = (nx, ny, nz) をグノモン投影面 (u_i, v_i) へ射影します.

    u = nx / nz, v = ny / nz

    Args:
        normals: 結晶面単位法線ベクトル配列 (N, 3)

    Returns:
        gnomonic_uv: グノモン平面座標配列 (N, 2)
    """
    if len(normals) == 0:
        return np.empty((0, 2), dtype=np.float64)

    norms = np.asarray(normals, dtype=np.float64)
    nz = np.maximum(norms[:, 2], 1e-6)
    u = norms[:, 0] / nz
    v = norms[:, 1] / nz
    return np.column_stack([u, v])


def detect_zone_lines(
    gnomonic_uv: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    config: Optional[LaueAlignmentConfig] = None,
) -> List[ZoneLine]:
    """グノモン平面上の点群から、高速ベクトル化ペア探索 + PCA直交回帰 + NMSにより主要晶帯直線を検出します.

    高速化と高精度を両立するため、全ペアに対する残差計算を行列演算でベクトル化し、
    インライア数上位の候補のみを厳密な PCA (Total Least Squares) 直交回帰でリフィットします。
    （計算時間: 通常 10〜20 ms）

    Args:
        gnomonic_uv: グノモン平面座標 (N, 2)
        intensities: 各スポットの強度 (N,)。Noneの場合は全点1.0
        config: アライメント設定

    Returns:
        zones: 検出・スコアリングされ、NMS統合された主要晶帯直線リスト (上位 max_zones 本)
    """
    cfg = config or LaueAlignmentConfig()
    N_total = len(gnomonic_uv)
    if N_total < max(cfg.min_inliers, 2):
        logger.warning("直線フィッティングに必要なスポット数が不足しています (N=%d)", N_total)
        return []

    # 計算効率とロバスト性を最大化するため、強度上位 N_max 点 (デフォルト 85 点) をペア生成に使用
    N_sub = min(N_total, 85)
    pts_sub = np.asarray(gnomonic_uv[:N_sub], dtype=np.float64)
    pts_all = np.asarray(gnomonic_uv, dtype=np.float64)

    dist_thresh = cfg.dist_thresh
    r_decay = cfg.r_decay

    # 1. 全ペア直線の高速ベクトル化生成 (NumPy triu_indices)
    idx_i, idx_j = np.triu_indices(N_sub, k=1)
    dp = pts_sub[idx_j] - pts_sub[idx_i]
    seg_lens = np.hypot(dp[:, 0], dp[:, 1])
    valid_pairs = seg_lens >= 0.03

    if not np.any(valid_pairs):
        return []

    a0 = -dp[valid_pairs, 1] / seg_lens[valid_pairs]
    b0 = dp[valid_pairs, 0] / seg_lens[valid_pairs]
    p1 = pts_sub[idx_i[valid_pairs]]
    c0 = -(a0 * p1[:, 0] + b0 * p1[:, 1])

    # 2. 行列積による全ペア×全スポット距離のベクトル化計算 (shape: M_pairs x N_sub)
    D = np.abs(a0[:, None] * pts_sub[None, :, 0] + b0[:, None] * pts_sub[None, :, 1] + c0[:, None])
    inlier_counts = np.sum(D <= dist_thresh, axis=1)

    # 最小インライア数を満たすペアを抽出
    candidate_pair_idx = np.where(inlier_counts >= cfg.min_inliers)[0]
    if len(candidate_pair_idx) == 0:
        return []

    # インライア数降順ソートし、上位400候補に絞ってPCA直交回帰
    sorted_cand_idx = candidate_pair_idx[np.argsort(-inlier_counts[candidate_pair_idx])[:400]]

    # 全点に対する重み準備
    if intensities is not None and len(intensities) == N_total and intensities.max() > 0:
        w_all = np.asarray(intensities, dtype=np.float64) / intensities.max()
    else:
        w_all = np.ones(N_total, dtype=np.float64)

    r_all = np.hypot(pts_all[:, 0], pts_all[:, 1])
    w_dist_all = np.exp(-0.5 * (r_all / max(r_decay, 1e-3)) ** 2)

    candidate_lines: List[Dict[str, Any]] = []

    # 3. PCA精密リフィット & スコアリング
    for p_idx in sorted_cand_idx:
        inl_pts = pts_sub[D[p_idx] <= dist_thresh]
        mean_pt = np.mean(inl_pts, axis=0)
        shifted = inl_pts - mean_pt
        cov = shifted.T @ shifted
        # 最小固有値の固有ベクトルが直線の法線ベクトル
        normal = np.linalg.eigh(cov)[1][:, 0]
        a_fit, b_fit = float(normal[0]), float(normal[1])
        c_fit = float(-(a_fit * mean_pt[0] + b_fit * mean_pt[1]))

        # c <= 0 に符号正規化
        if c_fit > 0:
            a_fit, b_fit, c_fit = -a_fit, -b_fit, -c_fit

        # 全点 (N_total 点全体) に対するインライア再評価
        all_dists = np.abs(a_fit * pts_all[:, 0] + b_fit * pts_all[:, 1] + c_fit)
        final_inliers = all_dists <= dist_thresh
        final_idx = np.where(final_inliers)[0].tolist()
        final_count = len(final_idx)

        if final_count < cfg.min_inliers:
            continue

        # スパン確認 (局所密集による偽晶帯を排除)
        dir_vec = np.array([-b_fit, a_fit])
        projs = pts_all[final_inliers] @ dir_vec
        span = float(np.max(projs) - np.min(projs))
        if span < 0.08:
            continue

        # スコアリング: 強度 * 距離減衰 * (1 - 残差/dist_thresh)
        res_weights = 1.0 - (all_dists[final_inliers] / dist_thresh)
        score = float(np.sum(w_all[final_inliers] * w_dist_all[final_inliers] * res_weights))

        candidate_lines.append({
            "a": a_fit,
            "b": b_fit,
            "c": c_fit,
            "score": score,
            "num_inliers": final_count,
            "inlier_indices": final_idx,
        })

    if not candidate_lines:
        return []

    # スコア降順ソート
    candidate_lines.sort(key=lambda item: item["score"], reverse=True)

    # 4. NMS (Non-Maximum Suppression) による類似直線の統合
    selected_zones: List[ZoneLine] = []
    angle_thresh_rad = np.radians(cfg.nms_angle_thresh_deg)

    for cand in candidate_lines:
        a_c = cand["a"]
        b_c = cand["b"]
        c_c = cand["c"]

        is_duplicate = False
        for sz in selected_zones:
            dot_prod = abs(a_c * sz.a + b_c * sz.b)
            dot_prod = min(1.0, max(-1.0, dot_prod))
            angle_diff = np.arccos(dot_prod)
            dist_diff = abs(abs(c_c) - abs(sz.c))

            if angle_diff < angle_thresh_rad and dist_diff < cfg.nms_dist_thresh:
                is_duplicate = True
                break

        if not is_duplicate:
            line_id = len(selected_zones) + 1
            z_len = np.sqrt(a_c ** 2 + b_c ** 2 + c_c ** 2)
            z_axis = (a_c / z_len, b_c / z_len, c_c / z_len)

            selected_zones.append(ZoneLine(
                line_id=line_id,
                a=a_c,
                b=b_c,
                c=c_c,
                score=cand["score"],
                num_inliers=cand["num_inliers"],
                inlier_indices=cand["inlier_indices"],
                zone_axis_norm=z_axis,
            ))

            if len(selected_zones) >= cfg.max_zones:
                break

    logger.info("NMSにより %d 本の主要晶帯直線を抽出しました (最高スコア=%.2f)", len(selected_zones), selected_zones[0].score if selected_zones else 0.0)
    return selected_zones


def find_principal_axis_candidates(
    zones: List[ZoneLine],
    gnomonic_uv: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    config: Optional[LaueAlignmentConfig] = None,
) -> List[AxisCandidate]:
    """検出された晶帯直線の交点群を解析・クラスタリングし、主要結晶面法線（主軸）の候補をランキングします.

    晶帯直線が3本以上集中する交点や、高スコア晶帯が交差するハブを主要低指数面法線
    (例: (001)面) として検出し、ゴニオメーター補正角とともに返します。
    晶帯が不足する場合は、中心近傍の高強度スポットへのフォールバックを行います。

    Args:
        zones: 検出された晶帯直線リスト
        gnomonic_uv: グノモン座標配列 (N, 2)
        intensities: 各スポットの強度 (N,)
        config: アライメント設定

    Returns:
        candidates: 第1候補〜第3候補の主軸候補リスト (空でないリスト)
    """
    cfg = config or LaueAlignmentConfig()
    camera_length = cfg.camera_length_mm
    max_tilt_rad = np.radians(cfg.search_cone_half_angle_deg)
    max_u_v = np.tan(max_tilt_rad)

    intersections: List[Dict[str, Any]] = []
    min_intersect_rad = np.radians(cfg.min_intersection_angle_deg)

    # 1. すべての主要直線ペアの交点を算出
    num_zones = len(zones)
    for i in range(num_zones):
        for j in range(i + 1, num_zones):
            z1 = zones[i]
            z2 = zones[j]

            det = z1.a * z2.b - z1.b * z2.a
            sin_angle = abs(det)
            if sin_angle < np.sin(min_intersect_rad):
                continue

            u_int = (-z1.c * z2.b - (-z2.c) * z1.b) / det
            v_int = (z1.a * (-z2.c) - z2.a * (-z1.c)) / det

            r_int = np.hypot(u_int, v_int)
            if r_int > max_u_v:
                continue

            pair_score = z1.score + z2.score
            intersections.append({
                "u": u_int,
                "v": v_int,
                "score": pair_score,
                "zones": [z1.line_id, z2.line_id],
            })

    # 2. 交点のクラスタリングと統合
    candidates: List[AxisCandidate] = []
    if intersections:
        coords = np.array([[item["u"], item["v"]] for item in intersections], dtype=np.float64)
        used = np.zeros(len(intersections), dtype=bool)
        clusters: List[Dict[str, Any]] = []

        eps = cfg.cluster_eps
        for idx in range(len(intersections)):
            if used[idx]:
                continue
            dists = np.hypot(coords[:, 0] - coords[idx, 0], coords[:, 1] - coords[idx, 1])
            members = np.where((dists <= eps) & (~used))[0]
            used[members] = True

            member_items = [intersections[m] for m in members]
            member_weights = np.array([m["score"] for m in member_items], dtype=np.float64)
            total_w = float(np.sum(member_weights))

            # 加重平均重心
            u_clust = float(np.sum([m["u"] * m["score"] for m in member_items]) / total_w)
            v_clust = float(np.sum([m["v"] * m["score"] for m in member_items]) / total_w)

            all_zones = set()
            for m in member_items:
                all_zones.update(m["zones"])
            contributing = sorted(list(all_zones))

            # 多重交点 (3本以上・4本以上の晶帯が集中する真の結晶対称軸) を強力に優遇
            zone_multi = len(contributing) ** 2.2
            tilt_pen = np.exp(-0.5 * (np.hypot(u_clust, v_clust) / (max_u_v * 0.75)) ** 2)
            clust_score = total_w * zone_multi * tilt_pen

            clusters.append({
                "u": u_clust,
                "v": v_clust,
                "score": clust_score,
                "zones": contributing,
            })

        # スコア降順ソート
        clusters.sort(key=lambda item: item["score"], reverse=True)

        for rank, cl in enumerate(clusters[:3], start=1):
            u_t = cl["u"]
            v_t = cl["v"]
            norm_vec, d_thx, d_thy, total_tilt = calculate_goniometer_angles(u_t, v_t, config=cfg)
            det_xy = gnomonic_to_detector(np.array([[u_t, v_t]]), camera_length_mm=camera_length)[0]

            candidates.append(AxisCandidate(
                rank=rank,
                u=u_t,
                v=v_t,
                normal_vector=norm_vec,
                delta_theta_x_deg=d_thx,
                delta_theta_y_deg=d_thy,
                total_tilt_deg=total_tilt,
                score=cl["score"],
                contributing_zones=cl["zones"],
                detector_x_mm=float(det_xy[0]),
                detector_y_mm=float(det_xy[1]),
            ))

    # 3. フォールバック処理
    if not candidates:
        logger.warning("晶帯直線の明確な交点が検出されませんでした。中心近傍の最高強度スポットにフォールバックします。")
        r_pts = np.hypot(gnomonic_uv[:, 0], gnomonic_uv[:, 1])
        valid_mask = r_pts <= max_u_v
        if np.any(valid_mask):
            valid_idx = np.where(valid_mask)[0]
            if intensities is not None and len(intensities) == len(gnomonic_uv):
                best_sub = valid_idx[np.argmax(intensities[valid_idx])]
            else:
                best_sub = valid_idx[np.argmin(r_pts[valid_idx])]
            u_fb, v_fb = float(gnomonic_uv[best_sub, 0]), float(gnomonic_uv[best_sub, 1])
        else:
            u_fb, v_fb = 0.0, 0.0

        norm_vec, d_thx, d_thy, total_tilt = calculate_goniometer_angles(u_fb, v_fb, config=cfg)
        det_xy = gnomonic_to_detector(np.array([[u_fb, v_fb]]), camera_length_mm=camera_length)[0]

        candidates.append(AxisCandidate(
            rank=1,
            u=u_fb,
            v=v_fb,
            normal_vector=norm_vec,
            delta_theta_x_deg=d_thx,
            delta_theta_y_deg=d_thy,
            total_tilt_deg=total_tilt,
            score=0.1,
            contributing_zones=[],
            detector_x_mm=float(det_xy[0]),
            detector_y_mm=float(det_xy[1]),
        ))

    return candidates


# ============================================================================
# Step 4: 2軸ゴニオメーター回転補正角 (Δθx, Δθy) の逆算
# ============================================================================

def calculate_goniometer_angles(
    u_target: float,
    v_target: float,
    config: Optional[LaueAlignmentConfig] = None,
) -> Tuple[Tuple[float, float, float], float, float, float]:
    """グノモン座標 (u_target, v_target) から3次元単位法線ベクトルと2軸ゴニオ補正角を算出します.

    2軸ゴニオメーターの回転行列 R により、目標結晶面法線 n_target を
    検出器中心方向 (0, 0, 1)^T に一致させる補正角 (Δθx, Δθy) [deg] を求めます。

    ステージ順序 "XY" (下側ステージX軸回転 θx, 上側ステージY軸回転 θy):
        R = Rx(Δθx) * Ry(Δθy)
        Δθy = sign_y * degrees(-arcsin(nx))
        Δθx = sign_x * degrees(atan2(ny, nz))

    ステージ順序 "YX" (下側ステージY軸回転 θy, 上側ステージX軸回転 θx):
        R = Ry(Δθy) * Rx(Δθx)
        Δθx = sign_x * degrees(arcsin(ny))
        Δθy = sign_y * degrees(-atan2(nx, nz))

    Args:
        u_target: 目標結晶面法線のグノモン座標 u
        v_target: 目標結晶面法線のグノモン座標 v
        config: アライメント設定

    Returns:
        normal_vector: 3次元単位面法線ベクトル (nx, ny, nz)
        delta_theta_x_deg: X軸まわり回転補正角 [deg]
        delta_theta_y_deg: Y軸まわり回転補正角 [deg]
        total_tilt_deg: 中心（入射X線方向）からの全傾き角 [deg]
    """
    cfg = config or LaueAlignmentConfig()

    denom = np.sqrt(u_target ** 2 + v_target ** 2 + 1.0)
    nx = float(u_target / denom)
    ny = float(v_target / denom)
    nz = float(1.0 / denom)

    total_tilt_deg = float(np.degrees(np.arccos(np.clip(nz, -1.0, 1.0))))

    sy = cfg.sign_y
    sx = cfg.sign_x

    if cfg.stage_order == "XY":
        delta_theta_y = sy * float(np.degrees(-np.arcsin(np.clip(nx, -1.0, 1.0))))
        delta_theta_x = sx * float(np.degrees(np.arctan2(ny, nz)))
    else:
        delta_theta_x = sx * float(np.degrees(np.arcsin(np.clip(ny, -1.0, 1.0))))
        delta_theta_y = sy * float(np.degrees(-np.arctan2(nx, nz)))

    return (nx, ny, nz), delta_theta_x, delta_theta_y, total_tilt_deg


# ============================================================================
# 逆変換ユーティリティ: グノモン座標 → 検出器平面 (双曲線) への逆変換
# ============================================================================

def gnomonic_to_detector(
    uv: np.ndarray,
    camera_length_mm: float,
) -> np.ndarray:
    """グノモン座標 (u, v) を検出器物理平面 (X, Y) [mm] へ逆写像します.

    回折幾何学における厳密な逆写像公式:
        X = L * 2u / (1 - (u^2 + v^2))
        Y = L * 2v / (1 - (u^2 + v^2))

    Args:
        uv: グノモン座標配列 (N, 2) または (2,)
        camera_length_mm: 試料―検出器間距離 L [mm]

    Returns:
        xy_mm: 検出器物理座標配列 (N, 2) [mm]
    """
    pts = np.asarray(uv, dtype=np.float64)
    is_1d = (pts.ndim == 1)
    if is_1d:
        pts = pts.reshape(1, 2)

    u = pts[:, 0]
    v = pts[:, 1]
    L = float(camera_length_mm)

    u2_v2 = u ** 2 + v ** 2
    denom = 1.0 - u2_v2
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)

    X = L * (2.0 * u) / denom
    Y = L * (2.0 * v) / denom

    res = np.column_stack([X, Y])
    return res[0] if is_1d else res


def zone_line_to_detector_hyperbola(
    zone: ZoneLine,
    camera_length_mm: float,
    num_points: int = 300,
    t_range: Tuple[float, float] = (-0.75, 0.75),
) -> np.ndarray:
    """グノモン平面上の晶帯直線 a*u + b*v + c = 0 を検出器平面上の双曲線軌跡 (X, Y) [mm] に変換します.

    Args:
        zone: 晶帯直線オブジェクト
        camera_length_mm: カメラ長 L [mm]
        num_points: サンプリング点数
        t_range: 直線に沿ったパラメータ t の探索範囲

    Returns:
        curve_xy: 検出器上の双曲線サンプリング点列 (M, 2) [mm]
    """
    a = zone.a
    b = zone.b
    c = zone.c

    t = np.linspace(t_range[0], t_range[1], num_points, dtype=np.float64)
    u_line = -c * a - b * t
    v_line = -c * b + a * t

    valid_mask = (u_line ** 2 + v_line ** 2) < 0.88
    if not np.any(valid_mask):
        return np.empty((0, 2), dtype=np.float64)

    uv_valid = np.column_stack([u_line[valid_mask], v_line[valid_mask]])
    return gnomonic_to_detector(uv_valid, camera_length_mm=camera_length_mm)


# ============================================================================
# パイプライン統括関数
# ============================================================================

def solve_laue_alignment(
    spots_or_image: Union[np.ndarray, str, Path],
    intensities: Optional[np.ndarray] = None,
    config: Optional[LaueAlignmentConfig] = None,
) -> LaueAlignmentResult:
    """背面反射ラウエ画像またはスポット座標から主軸同定・ゴニオ補正角をワンストップで高速推定します.

    Args:
        spots_or_image: 検出器スポット座標配列 (N, 2) [mm]、または画像配列 (2D ndarray)、または画像ファイルパス
        intensities: 各スポットの強度 (N,) [スポット座標入力時のみ]
        config: アライメント設定パラメータ

    Returns:
        result: 推定結果を格納した LaueAlignmentResult オブジェクト
    """
    t_start = time.perf_counter()
    cfg = config or LaueAlignmentConfig()

    # Step 1: スポット抽出
    if isinstance(spots_or_image, (str, Path)) or (isinstance(spots_or_image, np.ndarray) and spots_or_image.ndim >= 2 and spots_or_image.shape[1] > 2):
        spots_xy_mm, detected_intensities = extract_laue_spots_from_image(spots_or_image, config=cfg)
    else:
        spots_xy_mm = np.asarray(spots_or_image, dtype=np.float64)
        if intensities is not None and len(intensities) == len(spots_xy_mm):
            detected_intensities = np.asarray(intensities, dtype=np.float64)
        else:
            detected_intensities = np.ones(len(spots_xy_mm), dtype=np.float64)
        # 主要晶帯探索のため、強度降順に自動ソート
        if len(detected_intensities) > 0:
            sort_order = np.argsort(-detected_intensities)
            spots_xy_mm = spots_xy_mm[sort_order]
            detected_intensities = detected_intensities[sort_order]

    if len(spots_xy_mm) == 0:
        raise ValueError("スポットが1点も検出されませんでした。入力画像または閾値を確認してください。")

    # Step 2: 検出器座標 -> 結晶面法線ベクトル n_i
    normals = detector_to_normals(spots_xy_mm, camera_length_mm=cfg.camera_length_mm)

    # Step 3: グノモン投影 (u, v)
    gnomonic_uv = normals_to_gnomonic(normals)

    # 晶帯直線検出 (高速ベクトル化)
    zones = detect_zone_lines(gnomonic_uv, intensities=detected_intensities, config=cfg)

    # 主軸（交点）候補の抽出
    candidates = find_principal_axis_candidates(zones, gnomonic_uv, intensities=detected_intensities, config=cfg)
    best_candidate = candidates[0]

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    result = LaueAlignmentResult(
        spots_xy_mm=spots_xy_mm,
        intensities=detected_intensities,
        normals=normals,
        gnomonic_uv=gnomonic_uv,
        zones=zones,
        candidates=candidates,
        best_candidate=best_candidate,
        camera_length_mm=cfg.camera_length_mm,
        config=cfg,
        elapsed_ms=elapsed_ms,
    )
    return result


# ============================================================================
# 可視化・診断プロット (2パネル: Detector Plane & Gnomonic Plane)
# ============================================================================

def plot_alignment_diagnostics(
    result: LaueAlignmentResult,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """推定結果を確認するための2パネル診断図（検出器平面 vs グノモン平面）を生成・保存します.

    左パネル (Detector Plane (X, Y) [mm]):
        - 回折スポット群、検出された主要晶帯双曲線、主軸候補点および中心への移動ベクトル
    右パネル (Gnomonic Plane (u, v)):
        - グノモン点群、フィッティング直線群、およびそれらの交点群

    Args:
        result: solve_laue_alignment の結果
        save_path: 画像保存先パス。None の場合は保存しない。
        show: True の場合 plt.show() を呼び出す。

    Returns:
        fig: 作成された matplotlib Figure オブジェクト
    """
    fig, (ax_det, ax_gno) = plt.subplots(1, 2, figsize=(16, 7.8), facecolor="#181818")

    cfg = result.config
    L = result.camera_length_mm
    best = result.best_candidate

    zone_colors = ["#00e5ff", "#ff007f", "#39ff14", "#ffb300", "#b388ff", "#00e676"]

    # ------------------------------------------------------------------------
    # 左パネル: 検出器物理平面 (X, Y) [mm]
    # ------------------------------------------------------------------------
    ax_det.set_facecolor("#101010")

    spots = result.spots_xy_mm
    intens = result.intensities
    if len(intens) > 0 and intens.max() > 0:
        sizes = 18.0 + 80.0 * (intens / intens.max())
    else:
        sizes = 30.0

    ax_det.scatter(
        spots[:, 0], spots[:, 1],
        s=sizes, c="#e0e0e0", alpha=0.75, edgecolors="#ffffff", linewidths=0.5,
        label=f"Laue Spots (N={len(spots)})", zorder=3,
    )

    # 検出器中心 (入射X線軸)
    ax_det.scatter([0.0], [0.0], s=140, c="#ff3333", marker="+", linewidths=2.5, label="Beam Center (0, 0)", zorder=6)
    circle_hole = patches.Circle((0.0, 0.0), radius=1.5, fill=True, color="#ff3333", alpha=0.3, zorder=5)
    ax_det.add_patch(circle_hole)

    # 晶帯双曲線のオーバーレイ
    for idx, z in enumerate(result.zones):
        col = zone_colors[idx % len(zone_colors)]
        hyp = zone_line_to_detector_hyperbola(z, camera_length_mm=L)
        if len(hyp) > 5:
            ax_det.plot(hyp[:, 0], hyp[:, 1], color=col, linewidth=1.8, linestyle="-", alpha=0.85, label=f"Zone #{z.line_id} (N={z.num_inliers})", zorder=4)

    # 主軸候補のプロット
    candidate_markers = ["*", "o", "s"]
    for c in result.candidates:
        m = candidate_markers[(c.rank - 1) % len(candidate_markers)]
        col = "#ffd700" if c.rank == 1 else "#00ffcc"
        s_size = 280 if c.rank == 1 else 130
        lbl = f"Target Axis #{c.rank} (Δθx={c.delta_theta_x_deg:+.2f}°, Δθy={c.delta_theta_y_deg:+.2f}°)" if c.rank == 1 else f"Candidate #{c.rank}"
        ax_det.scatter([c.detector_x_mm], [c.detector_y_mm], s=s_size, marker=m, c=col, edgecolors="#000000", linewidths=1.2, label=lbl, zorder=8)

        if c.rank == 1:
            ax_det.annotate(
                "", xy=(0.0, 0.0), xytext=(c.detector_x_mm, c.detector_y_mm),
                arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.6", color="#ffd700", lw=2.2, linestyle="--"),
                zorder=7,
            )

    max_xy = max(40.0, float(np.max(np.abs(spots))) * 1.15 if len(spots) > 0 else 40.0)
    ax_det.set_xlim(-max_xy, max_xy)
    ax_det.set_ylim(-max_xy, max_xy)
    ax_det.set_aspect("equal")
    ax_det.set_xlabel("Detector X [mm] (Horizontal)", color="#cccccc", fontsize=11)
    ax_det.set_ylabel("Detector Y [mm] (Vertical)", color="#cccccc", fontsize=11)
    ax_det.set_title("Detector Plane (X, Y) [Back-reflection]\nSpots, Hyperbolic Zone Traces & Target Alignment Vector", color="#ffffff", fontsize=12, pad=10)
    ax_det.grid(True, color="#333333", linestyle="--", alpha=0.5)
    ax_det.tick_params(colors="#aaaaaa")
    ax_det.legend(loc="upper right", facecolor="#242424", edgecolor="#444444", labelcolor="#dddddd", fontsize=8.5)

    # ------------------------------------------------------------------------
    # 右パネル: グノモン投影平面 (u, v)
    # ------------------------------------------------------------------------
    ax_gno.set_facecolor("#101010")

    uv = result.gnomonic_uv
    ax_gno.scatter(
        uv[:, 0], uv[:, 1],
        s=sizes * 0.7, c="#aaaaaa", alpha=0.65, edgecolors="none",
        label=f"Projected Normals (N={len(uv)})", zorder=3,
    )

    cone_rad = np.radians(cfg.search_cone_half_angle_deg)
    cone_r = float(np.tan(cone_rad))
    circle_cone = patches.Circle((0.0, 0.0), radius=cone_r, fill=False, edgecolor="#666666", linestyle=":", linewidth=1.5, label=f"Search Cone (±{cfg.search_cone_half_angle_deg:.0f}°)", zorder=2)
    ax_gno.add_patch(circle_cone)
    ax_gno.scatter([0.0], [0.0], s=100, c="#ff3333", marker="+", linewidths=2.0, zorder=5)

    u_grid = np.linspace(-cone_r * 1.3, cone_r * 1.3, 200)
    for idx, z in enumerate(result.zones):
        col = zone_colors[idx % len(zone_colors)]
        inliers_uv = uv[z.inlier_indices]
        ax_gno.scatter(inliers_uv[:, 0], inliers_uv[:, 1], s=45, c=col, alpha=0.9, zorder=4)

        if abs(z.b) > 1e-4:
            v_line = (-z.c - z.a * u_grid) / z.b
            valid_l = np.abs(v_line) <= cone_r * 1.4
            ax_gno.plot(u_grid[valid_l], v_line[valid_l], color=col, linewidth=1.8, linestyle="-", alpha=0.85, zorder=4)
        else:
            u_const = -z.c / z.a
            ax_gno.axvline(u_const, color=col, linewidth=1.8, linestyle="-", alpha=0.85, zorder=4)

    for c in result.candidates:
        m = candidate_markers[(c.rank - 1) % len(candidate_markers)]
        col = "#ffd700" if c.rank == 1 else "#00ffcc"
        s_size = 280 if c.rank == 1 else 130
        ax_gno.scatter([c.u], [c.v], s=s_size, marker=m, c=col, edgecolors="#000000", linewidths=1.2, label=f"Hub #{c.rank} (Score={c.score:.1f})" if c.rank == 1 else f"Hub #{c.rank}", zorder=8)

    ax_gno.set_xlim(-cone_r * 1.35, cone_r * 1.35)
    ax_gno.set_ylim(-cone_r * 1.35, cone_r * 1.35)
    ax_gno.set_aspect("equal")
    ax_gno.set_xlabel("Gnomonic u = nx / nz", color="#cccccc", fontsize=11)
    ax_gno.set_ylabel("Gnomonic v = ny / nz", color="#cccccc", fontsize=11)
    ax_gno.set_title("Gnomonic Plane (u, v)\nLinear Zones, Intersecting Hubs & Principal Axis", color="#ffffff", fontsize=12, pad=10)
    ax_gno.grid(True, color="#333333", linestyle="--", alpha=0.5)
    ax_gno.tick_params(colors="#aaaaaa")
    ax_gno.legend(loc="upper right", facecolor="#242424", edgecolor="#444444", labelcolor="#dddddd", fontsize=8.5)

    info_header = (
        f"AstroLaue Automated Crystal Alignment  |  Camera L={L:.1f} mm  |  Stage Order: {cfg.stage_order}  |  Time: {result.elapsed_ms:.1f} ms\n"
        f"RECOMMENDED CORRECTION:  Δθx = {best.delta_theta_x_deg:+.2f}° ,  Δθy = {best.delta_theta_y_deg:+.2f}°  "
        f"(Total Misorientation ψ = {best.total_tilt_deg:.2f}°)"
    )
    fig.suptitle(info_header, color="#ffffff", fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_p), dpi=160, facecolor=fig.get_facecolor())
        logger.info("アライメント診断図を保存しました: %s", out_p)

    if show:
        plt.show()

    return fig


# ============================================================================
# 合成データ生成関数
# ============================================================================

def generate_synthetic_laue_sample(
    true_tilt_x_deg: float = 6.0,
    true_tilt_y_deg: float = -4.5,
    camera_length_mm: float = 40.0,
    lattice_type: str = "tetragonal",
    c_over_a: float = 1.45,
    max_hkl: int = 5,
    num_fake_spots: int = 12,
    noise_sigma_mm: float = 0.04,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """指定された傾き角 (θx, θy) を持つ未知結晶の合成ラウエスポット群（+ノイズ）を物理シミュレーション生成します."""
    rng = np.random.default_rng(random_seed)

    thx = np.radians(true_tilt_x_deg)
    thy = np.radians(true_tilt_y_deg)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(thx), -np.sin(thx)],
        [0.0, np.sin(thx), np.cos(thx)],
    ], dtype=np.float64)

    Ry = np.array([
        [np.cos(thy), 0.0, np.sin(thy)],
        [0.0, 1.0, 0.0],
        [-np.sin(thy), 0.0, np.cos(thy)],
    ], dtype=np.float64)

    R_sample = Rx @ Ry

    h_range = np.arange(-max_hkl, max_hkl + 1)
    k_range = np.arange(-max_hkl, max_hkl + 1)
    l_range = np.arange(1, max_hkl + 2)

    spots_list = []
    intens_list = []

    for h in h_range:
        for k in k_range:
            for l in l_range:
                if h == 0 and k == 0 and l == 0:
                    continue

                if lattice_type == "cubic":
                    gx, gy, gz = float(h), float(k), float(l)
                elif lattice_type == "tetragonal":
                    gx, gy, gz = float(h), float(k), float(l / c_over_a)
                elif lattice_type == "hexagonal":
                    gx = float(h)
                    gy = float((h + 2 * k) / np.sqrt(3.0))
                    gz = float(l / c_over_a)
                else:
                    gx, gy, gz = float(h), float(k), float(l)

                g_len = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
                if g_len < 1e-4:
                    continue

                nc = np.array([gx / g_len, gy / g_len, gz / g_len], dtype=np.float64)
                n_lab = R_sample @ nc

                if n_lab[2] <= 0.05:
                    continue

                sz = 2.0 * (n_lab[2] ** 2) - 1.0
                if sz <= 0.08:
                    continue

                sx = 2.0 * n_lab[0] * n_lab[2]
                sy = 2.0 * n_lab[1] * n_lab[2]

                X_det = camera_length_mm * (sx / sz)
                Y_det = camera_length_mm * (sy / sz)

                r_det = np.hypot(X_det, Y_det)
                if 2.5 <= r_det <= 42.0:
                    base_i = 1.0 / (g_len ** 1.8)
                    if abs(h) <= 1 and abs(k) <= 1:
                        base_i *= 2.2

                    nx_pos = X_det + rng.normal(0, noise_sigma_mm)
                    ny_pos = Y_det + rng.normal(0, noise_sigma_mm)

                    spots_list.append((nx_pos, ny_pos))
                    intens_list.append(base_i)

    # ランダムな孤立偽スポット混入
    for _ in range(num_fake_spots):
        r_f = rng.uniform(4.0, 40.0)
        th_f = rng.uniform(0.0, 2.0 * np.pi)
        xf = r_f * np.cos(th_f)
        yf = r_f * np.sin(th_f)
        spots_list.append((xf, yf))
        intens_list.append(float(rng.uniform(0.05, 0.35)))

    spots_xy = np.array(spots_list, dtype=np.float64)
    intensities = np.array(intens_list, dtype=np.float64)

    if len(intensities) > 0 and intensities.max() > 0:
        intensities /= intensities.max()

    meta = {
        "true_tilt_x_deg": true_tilt_x_deg,
        "true_tilt_y_deg": true_tilt_y_deg,
        "expected_delta_theta_x_deg": -true_tilt_x_deg,
        "expected_delta_theta_y_deg": -true_tilt_y_deg,
        "camera_length_mm": camera_length_mm,
        "num_spots": len(spots_xy),
        "lattice_type": lattice_type,
    }

    return spots_xy, intensities, meta


# ============================================================================
# CLI インターフェース (単独実行対応)
# ============================================================================

def build_alignment_arg_parser() -> argparse.ArgumentParser:
    """CLI 引数パーサーを構築します."""
    parser = argparse.ArgumentParser(
        prog="astrolaue-align",
        description="AstroLaue: 背面反射ラウエ結晶主軸自動同定 & 2軸ゴニオ補正角推定 CLI",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="入力画像パス (例: share/260930-1.png または share/Feラウエsample.png)。省略時は合成デモを実行。",
    )
    parser.add_argument(
        "--camera-length", "-L",
        type=float,
        default=40.0,
        help="試料―検出器間距離 L [mm] (デフォルト: 40.0)",
    )
    parser.add_argument(
        "--pixel-size", "-p",
        type=float,
        default=0.05,
        help="検出器ピクセルサイズ [mm/pix] (デフォルト: 0.05 mm)",
    )
    parser.add_argument(
        "--stage-order",
        choices=["XY", "YX"],
        default="XY",
        help="ゴニオメーターステージの軸構成 ('XY': 下X上Y, 'YX': 下Y上X) (デフォルト: XY)",
    )
    parser.add_argument(
        "--sign-x",
        type=int,
        choices=[1, -1],
        default=1,
        help="X軸回転の符号係数 (デフォルト: 1)",
    )
    parser.add_argument(
        "--sign-y",
        type=int,
        choices=[1, -1],
        default=1,
        help="Y軸回転の符号係数 (デフォルト: 1)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results/alignment_diagnostic.png",
        help="診断プロットの出力ファイルパス (デフォルト: ./results/alignment_diagnostic.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="診断図の描画・保存をスキップする (高速処理向け)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI メインエントリポイント."""
    parser = build_alignment_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = LaueAlignmentConfig(
        camera_length_mm=args.camera_length,
        pixel_size_mm=args.pixel_size,
        stage_order=args.stage_order,
        sign_x=args.sign_x,
        sign_y=args.sign_y,
    )

    print("=" * 75)
    print(" AstroLaue: 自動結晶主軸方位同定 & 2軸ゴニオメーター補正角推定")
    print("=" * 75)
    print(f"・カメラ長 L: {cfg.camera_length_mm:.1f} mm")
    print(f"・ピクセルサイズ: {cfg.pixel_size_mm:.4f} mm/pix")
    print(f"・ステージ構成: {cfg.stage_order} (sign_x={cfg.sign_x}, sign_y={cfg.sign_y})")

    if args.input:
        in_path = Path(args.input)
        print(f"・解析対象画像: {in_path.resolve()}")
        result = solve_laue_alignment(in_path, config=cfg)
    else:
        print("・入力画像未指定のため、未知正方晶の傾き合成シミュレーションデモを実行します.")
        true_thx, true_thy = 6.0, -4.5
        spots_xy, intensities, meta = generate_synthetic_laue_sample(
            true_tilt_x_deg=true_thx,
            true_tilt_y_deg=true_thy,
            camera_length_mm=cfg.camera_length_mm,
            lattice_type="tetragonal",
            num_fake_spots=15,
            noise_sigma_mm=0.04,
            random_seed=123,
        )
        print(f"  (正解初期傾き: θx = {true_thx:+.2f}°, θy = {true_thy:+.2f}°)")
        result = solve_laue_alignment(spots_xy, intensities=intensities, config=cfg)

    best = result.best_candidate

    print("\n" + "=" * 75)
    print(" 【主軸同定 & ゴニオ補正角 推定結果】")
    print("=" * 75)
    print(f"検出スポット数: {len(result.spots_xy_mm)} 点")
    print(f"同定主要晶帯数: {len(result.zones)} 本")
    for z in result.zones:
        print(f"  - Zone #{z.line_id}: a={z.a:+.3f}, b={z.b:+.3f}, c={z.c:+.3f} | Score={z.score:5.1f} | Inliers={z.num_inliers:2d}")

    print(f"\n推定された主軸 (001) 候補 (上位 {len(result.candidates)} 件):")
    for c in result.candidates:
        print(f"  [第{c.rank}候補] グノモン(u, v)=({c.u:+.4f}, {c.v:+.4f}) | 検出器(X, Y)=({c.detector_x_mm:+.2f}, {c.detector_y_mm:+.2f}) mm")
        print(f"             補正角: Δθx = {c.delta_theta_x_deg:+.3f}°,  Δθy = {c.delta_theta_y_deg:+.3f}° (全傾き={c.total_tilt_deg:.2f}°) | 信頼度={c.score:6.1f}")

    print("-" * 75)
    print(f"★ 推奨ゴニオメーター補正回転角:  Δθx = {best.delta_theta_x_deg:+.3f}°,  Δθy = {best.delta_theta_y_deg:+.3f}°")
    print(f"★ 現在の結晶主軸傾き角 ψ:       {best.total_tilt_deg:.3f}°")
    print(f"★ 解析処理時間:                 {result.elapsed_ms:.1f} ms")
    print("=" * 75)

    if not args.no_plot:
        out_p = Path(args.output)
        plot_alignment_diagnostics(result, save_path=out_p, show=False)
        print(f"アライメント診断図を保存しました: {out_p.resolve()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
