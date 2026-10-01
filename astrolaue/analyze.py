"""モジュール 5: 直交復元 & スポット解析.

復元された画像から回折斑点 (Bragg spots) を自動検出し、
各スポットの重心座標 (x, y)、極座標 (r, theta)、ピーク強度、積分強度、
および復元前後の半値幅 (FWHM) を高精度に算出・比較します。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from scipy import optimize
from skimage.feature import blob_log
from photutils.detection import DAOStarFinder
from astropy.stats import mad_std

logger = logging.getLogger(__name__)


@dataclass
class SpotResult:
    """検出された回折スポットの解析結果."""

    spot_id: int
    x: float
    y: float
    r: float
    theta_rad: float
    theta_deg: float
    peak_intensity: float
    integrated_intensity: float
    fwhm_before: float
    fwhm_after: float
    fwhm_reduction_ratio: float
    sigma: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_spots(
    image: np.ndarray,
    min_sigma: float = 2.0,
    max_sigma: float = 12.0,
    num_sigma: int = 10,
    threshold: Optional[float] = None,
    max_spots: int = 150,
    method: str = "daofind",
) -> List[Tuple[float, float, float]]:
    """天体観測標準 photutils.detection.DAOStarFinder または LoG により回折スポット中心 (x, y, sigma) を検出します.

    DAOStarFinder は天文学標準（DAOPhot / IRAF）の点光源検出アルゴリズムであり、
    ガウス点拡がり関数 (PSF) との相関および背景ノイズ分散 (sigma_MAD) に基づいて、
    微小ノイズを排除しながら微弱ピークを高感度・高精度に同定します。

    Args:
        image: スポット検出対象の直交画像 (float32 [0, 1] または [0, 255])
        min_sigma: 最小ブロブサイズ (ピクセル)
        max_sigma: 最大ブロブサイズ (ピクセル)
        num_sigma: スケール空間の分割数
        threshold: 検出応答の閾値 (None の場合は背景分散 sigma_MAD に基づき適応的に算出)
        max_spots: 抽出する最大スポット数 (強度順、デフォルト: 150)
        method: 検出アルゴリズム ('daofind': photutils天体標準, 'log': skimage LoG)

    Returns:
        spots: [(x, y, sigma), ...] のリスト
    """
    img_f = image.astype(np.float32)
    if img_f.max() > 1.0:
        img_f /= img_f.max()

    # 適応的閾値の計算 (背景分散 sigma_MAD 基準)
    pos = img_f[img_f > 0.0]
    if len(pos) > 100:
        med = float(np.median(pos))
        mad = float(np.median(np.abs(pos - med)))
        sigma_mad = 1.4826 * mad
        auto_thresh = float(np.clip(3.0 * sigma_mad, 0.008, 0.05))
    else:
        sigma_mad = 0.01
        auto_thresh = 0.03

    eff_thresh = auto_thresh if threshold is None else float(threshold)

    selected: List[Tuple[float, float, float]] = []

    # 1. 天体標準 photutils.detection.DAOStarFinder による高精度点光源同定
    # ラウエスポットは方位角方向にアーク状に延伸しているため、真円前提 (-1.0~1.0) を緩和して検出
    if method == "daofind":
        try:
            expected_fwhm = max(float(min_sigma * 2.355), 3.0)
            daofind = DAOStarFinder(
                fwhm=expected_fwhm,
                threshold=eff_thresh,
                roundness_range=(-1.8, 1.8),
                sharpness_range=(0.05, 2.5),
            )
            sources = daofind(img_f)
            if sources is not None and len(sources) > 0:
                spot_list = []
                for row in sources:
                    xc = float(row["x_centroid"])
                    yc = float(row["y_centroid"])
                    peak_val = float(row["peak"])
                    sig = max(float(expected_fwhm / 2.355), min_sigma)
                    spot_list.append((xc, yc, sig, peak_val))

                spot_list.sort(key=lambda item: item[3], reverse=True)
                selected = [(x, y, sig) for x, y, sig, _ in spot_list[:max_spots]]
                logger.info("photutils DAOStarFinder 検出完了: %d 個のスポットを検出しました (閾値=%.4f)", len(selected), eff_thresh)
        except Exception as e_dao:
            logger.warning("DAOStarFinder 検出中に例外が発生したため LoG にフォールバックします: %s", e_dao)
            selected = []

    # 2. フォールバックまたは明示的 LoG 指定
    if len(selected) == 0:
        blobs = blob_log(
            img_f,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=eff_thresh,
        )

        if len(blobs) == 0:
            logger.warning("スポット検出でスポットが検出されませんでした。閾値を下げてください。")
            return []

        spot_list = []
        h, w = image.shape[:2]
        for b in blobs:
            y, x, sig = float(b[0]), float(b[1]), float(b[2])
            yi, xi = int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))
            val = float(img_f[yi, xi])
            spot_list.append((x, y, sig, val))

        spot_list.sort(key=lambda item: item[3], reverse=True)
        selected = [(x, y, sig) for x, y, sig, _ in spot_list[:max_spots]]
        logger.info("LoGブロブ検出完了: %d 個のスポットを検出しました (閾値=%.4f)", len(selected), eff_thresh)

    return selected


def measure_fwhm_profile(
    profile: np.ndarray,
    peak_idx: int | None = None,
) -> float:
    """1次元輝度プロファイルから半値全幅 (FWHM) を直接サブピクセル線形補間で算出します.

    Args:
        profile: 1次元プロファイル配列
        peak_idx: ピーク位置のインデックス。Noneの場合は argmax。

    Returns:
        fwhm: 半値全幅 (ピクセル単位)。算出不能時は 0.0。
    """
    if len(profile) < 3:
        return 0.0

    if peak_idx is None:
        peak_idx = int(np.argmax(profile))

    peak_val = float(profile[peak_idx])
    base_val = float(np.min(profile))
    half_val = base_val + 0.5 * (peak_val - base_val)

    if peak_val <= base_val:
        return 0.0

    # 左側の交点探索
    left_x = float(peak_idx)
    for i in range(peak_idx - 1, -1, -1):
        if profile[i] <= half_val:
            denom = profile[i + 1] - profile[i]
            frac = (half_val - profile[i]) / denom if abs(denom) > 1e-6 else 0.0
            left_x = i + frac
            break

    # 右側の交点探索
    right_x = float(peak_idx)
    for i in range(peak_idx + 1, len(profile)):
        if profile[i] <= half_val:
            denom = profile[i - 1] - profile[i]
            frac = (half_val - profile[i]) / denom if abs(denom) > 1e-6 else 0.0
            right_x = i - frac
            break

    fwhm = max(0.0, right_x - left_x)
    return float(fwhm)


def measure_spot_fwhm(
    polar_orig: np.ndarray,
    polar_deblur: np.ndarray,
    r_idx: int,
    theta_idx: int,
    window: int = 35,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """特定スポットの復元前後の FWHM とプロファイルカーブを取得します.

    Args:
        polar_orig: 復元前極座標画像
        polar_deblur: 復元後極座標画像
        r_idx: スポットの半径インデックス
        theta_idx: スポットの方位角インデックス
        window: 抽出する方位角方向の窓サイズ

    Returns:
        fwhm_before: 復元前の FWHM
        fwhm_after: 復元後の FWHM
        prof_before: 復元前プロファイル
        prof_after: 復元後プロファイル
    """
    n_theta = polar_orig.shape[1]
    half_w = window // 2
    col_indices = np.mod(np.arange(theta_idx - half_w, theta_idx + half_w + 1), n_theta)

    prof_b = polar_orig[r_idx, col_indices]
    prof_a = polar_deblur[r_idx, col_indices]

    fwhm_b = measure_fwhm_profile(prof_b)
    fwhm_a = measure_fwhm_profile(prof_a)

    return fwhm_b, fwhm_a, prof_b, prof_a



def gaussian_func(x: np.ndarray, a: float, x0: float, sigma: float, c: float) -> np.ndarray:
    """ガウス関数フィッティングモデル."""
    return a * np.exp(-0.5 * ((x - x0) / max(sigma, 1e-4)) ** 2) + c


def fit_gaussian_fwhm(profile: np.ndarray) -> Tuple[float, np.ndarray]:
    """1次元プロファイルにガウス関数をフィッティングして FWHM と予測値を算出します.

    Returns:
        fwhm: フィッティングされた FWHM
        fit_curve: フィッティングカーブ配列
    """
    x = np.arange(len(profile), dtype=np.float64)
    y = profile.astype(np.float64)
    peak_idx = int(np.argmax(y))
    a0 = float(y[peak_idx] - np.min(y))
    c0 = float(np.min(y))
    x0_init = float(peak_idx)
    sig0 = 3.0

    try:
        popt, _ = optimize.curve_fit(
            gaussian_func,
            x,
            y,
            p0=[a0, x0_init, sig0, c0],
            bounds=([0.0, 0.0, 0.5, 0.0], [np.inf, len(profile), len(profile), np.inf]),
            maxfev=1000,
        )
        fitted_sigma = abs(popt[2])
        fwhm = fitted_sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))
        fit_curve = gaussian_func(x, *popt)
        return float(fwhm), fit_curve
    except Exception:
        # フィッティング失敗時は直接法
        fwhm_direct = measure_fwhm_profile(profile, peak_idx)
        return fwhm_direct, profile.copy()


def analyze_diffraction_spots(
    cart_before: np.ndarray,
    cart_after: np.ndarray,
    polar_before: np.ndarray,
    polar_after: np.ndarray,
    center: Tuple[float, float],
    r_range: Tuple[float, float],
    min_sigma: float = 2.0,
    max_sigma: float = 12.0,
    threshold: Optional[float] = None,
    max_spots: int = 150,
    window_theta_pix: int = 35,
) -> Tuple[List[SpotResult], Dict[str, Any]]:
    """検出された全スポットの解析および代表スポットの復元前後プロファイル比較データを抽出します.

    Args:
        cart_before: 幾何規格化後の元直交画像 (背景減算済み)
        cart_after: 復元後の直交画像
        polar_before: 極座標元画像
        polar_after: 復元極座標画像
        center: 直交画像上の中心座標 (xc, yc)
        r_range: 極座標の半径範囲 (r_min, r_max)
        min_sigma: スポット検出最小スケール
        max_sigma: スポット検出最大スケール
        threshold: スポット検出閾値 (None で背景MAD自動)
        max_spots: 最大スポット検出数 (デフォルト: 150)
        window_theta_pix: プロファイル抽出の角度窓幅

    Returns:
        results: 各スポットの SpotResult リスト
        representative_profile: 代表スポットのプロファイル比較辞書
    """
    xc, yc = center
    r_min, r_max = r_range
    n_r, n_theta = polar_before.shape[:2]

    # 復元後直交画像からスポット候補を検出
    raw_spots = detect_spots(
        cart_after,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold,
        max_spots=max_spots,
    )

    results: List[SpotResult] = []
    h_c, w_c = cart_after.shape[:2]

    # 極座標画像の背景ノイズ水準をロバスト推定 (微弱ピークを不当に足切りしない)
    pos_pol = polar_before[polar_before > 0.0]
    if len(pos_pol) > 100:
        med_pol = float(np.median(pos_pol))
        mad_pol = float(np.median(np.abs(pos_pol - med_pol)))
        sigma_pol = 1.4826 * mad_pol
        min_peak_b = max(2.5 * sigma_pol, 0.008)
    else:
        min_peak_b = 0.015

    for idx, (sx, sy, sig) in enumerate(raw_spots):
        dx = sx - xc
        dy = sy - yc
        r = float(np.hypot(dx, dy))
        theta = float(np.arctan2(dy, dx))
        if theta < 0:
            theta += 2.0 * np.pi

        # 極座標インデックスへ変換
        r_idx = int(np.clip(round((r - r_min) / (r_max - r_min) * (n_r - 1)), 0, n_r - 1))
        th_idx = int(np.clip(round((theta / (2.0 * np.pi)) * n_theta), 0, n_theta - 1))

        # theta 軸のプロファイル窓を抽出 (循環境界対応)
        half_w = window_theta_pix // 2
        col_indices = np.mod(np.arange(th_idx - half_w, th_idx + half_w + 1), n_theta)

        prof_before = polar_before[r_idx, col_indices]
        prof_after = polar_after[r_idx, col_indices]

        # FWHM 算出
        fwhm_b = measure_fwhm_profile(prof_before)
        fwhm_a = measure_fwhm_profile(prof_after)

        reduction = (1.0 - (fwhm_a / fwhm_b)) * 100.0 if fwhm_b > 1e-4 else 0.0

        # ピーク強度と局所積分強度
        peak_val = float(cart_after[int(round(sy)), int(round(sx))]) if 0 <= round(sy) < h_c and 0 <= round(sx) < w_c else 0.0

        # 偽スポット (元画像に存在しないノイズ先鋭化アーティファクト) の除外判定
        # 1. 復元前の極座標プロファイルに有意なピークシグナルが存在するか (S/N比 >= 2.5)
        peak_b = float(np.max(prof_before))
        if peak_b < min_peak_b:
            continue

        # 2. 元画像でのFWHMが極端に細い (露光走査ブラーを受けていない孤立ノイズ点)
        if fwhm_b < 2.5:
            continue

        # 3. 復元前後で先鋭化せず、かつ強度が極小のノイズ
        if reduction < 3.0 and peak_val < min_peak_b * 1.5:
            continue

        # 半径 2*sigma 内の積分強度
        rad_i = int(max(round(sig * 2.0), 3))
        y_min = max(0, int(round(sy)) - rad_i)
        y_max = min(h_c, int(round(sy)) + rad_i + 1)
        x_min = max(0, int(round(sx)) - rad_i)
        x_max = min(w_c, int(round(sx)) + rad_i + 1)
        patch = cart_after[y_min:y_max, x_min:x_max]
        integrated_val = float(np.sum(patch))

        res = SpotResult(
            spot_id=len(results) + 1,
            x=sx,
            y=sy,
            r=r,
            theta_rad=theta,
            theta_deg=float(np.degrees(theta)),
            peak_intensity=peak_val,
            integrated_intensity=integrated_val,
            fwhm_before=fwhm_b,
            fwhm_after=fwhm_a,
            fwhm_reduction_ratio=reduction,
            sigma=sig,
        )
        results.append(res)

    # 代表スポットの選定 (隣接スポットの重なりがない、十分に孤立した良質スポットを優先)
    rep_profile = {}
    if len(results) > 0:
        # 各スポットの孤立度 (最も近い他のスポットとの角度距離) を算出
        scored_spots = []
        for i, s1 in enumerate(results):
            min_dth = 360.0
            for j, s2 in enumerate(results):
                if i == j:
                    continue
                dth = abs(s1.theta_deg - s2.theta_deg)
                dth = min(dth, 360.0 - dth)
                if dth < min_dth:
                    min_dth = dth
            scored_spots.append((s1, min_dth))

        # 代表スポットの選定:
        # S/N比が十分に高い主要スポット群（最大強度の 20% 以上、かつ先鋭化率 > 10%）を候補とし、
        # その中で「隣接スポットとの方位角距離が最も離れている（孤立度が高い）」スポットを厳選
        max_p = max(s.peak_intensity for s in results)
        strong_candidates = [
            (s, dth) for s, dth in scored_spots
            if s.peak_intensity >= max_p * 0.20 and s.fwhm_before >= 3.0 and s.fwhm_reduction_ratio > 10.0
        ]
        if strong_candidates:
            # 孤立距離 (隣接スポットとの重なり回避) を最優先し、強度・先鋭化率を乗じた総合スコア
            rep_spot = max(
                strong_candidates,
                key=lambda item: (min(item[1], 30.0) * np.sqrt(item[0].peak_intensity) * max(item[0].fwhm_reduction_ratio, 1.0))
            )[0]
        else:
            rep_spot = max(results, key=lambda s: s.peak_intensity)

        r = rep_spot.r
        theta = rep_spot.theta_rad
        r_idx = int(np.clip(round((r - r_min) / max(r_max - r_min, 1e-4) * (n_r - 1)), 0, n_r - 1))
        th_idx = int(np.clip(round((theta / (2.0 * np.pi)) * n_theta), 0, n_theta - 1))

        # 極座標画像上の真のピーク最大位置をサブサーチして窓中心を正確に一致
        search_w = 4
        sub_col = np.mod(np.arange(th_idx - search_w, th_idx + search_w + 1), n_theta)
        local_max_offset = int(np.argmax(polar_after[r_idx, sub_col])) - search_w
        true_th_idx = (th_idx + local_max_offset) % n_theta

        half_w = window_theta_pix // 2
        col_indices = np.mod(np.arange(true_th_idx - half_w, true_th_idx + half_w + 1), n_theta)
        prof_b = polar_before[r_idx, col_indices]
        prof_a = polar_after[r_idx, col_indices]

        rel_theta = np.linspace(-half_w, half_w, len(col_indices))

        rep_profile = {
            "spot_id": rep_spot.spot_id,
            "r": rep_spot.r,
            "theta_deg": rep_spot.theta_deg,
            "rel_pixels": rel_theta,
            "prof_before": prof_b,
            "prof_after": prof_a,
            "fwhm_before": rep_spot.fwhm_before,
            "fwhm_after": rep_spot.fwhm_after,
        }

    return results, rep_profile
