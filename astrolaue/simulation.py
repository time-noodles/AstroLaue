"""ラウエ回折像物理シミュレーション & 復元精度ベンチマークモジュール.

既知の結晶ゾーン軸・反射スポット・走査ブラー長を持つGround Truth（正解真値）ラウエ像を生成し、
AstroLaue パイプラインによるピーク位置・強度・先鋭化・検出率の再現精度を定量評価します。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import shift
import scipy.special as sp
from scipy.optimize import curve_fit

from astrolaue.deblur import create_psf_1d
from astrolaue.polar import cartesian_to_polar, polar_to_cartesian

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthSpot:
    """シミュレーション生成された真のラウエスポット情報."""

    spot_id: int
    x: float
    y: float
    r: float
    theta_deg: float
    true_peak_intensity: float
    true_integrated_flux: float
    true_fwhm: float


@dataclass
class RecoveryBenchmarkMetrics:
    """復元アルゴリズムのピーク再現性・精度指標."""

    num_ground_truth: int
    num_detected: int
    num_matched: int
    recall: float                  # 再現率 (検出できた真スポット割合: 0.0 - 1.0)
    precision: float               # 適合率 (検出スポット中の真スポット割合: 0.0 - 1.0)
    f1_score: float                # F1スコア (再現率と適合率の調和平均: 0.0 - 1.0)
    mean_position_error_px: float  # 平均重心位置誤差 (ピクセル)
    position_rmse_px: float        # 重心位置RMSE (ピクセル)
    intensity_r2: float            # 真値強度と復元強度の決定係数 R^2
    mean_flux_recovery_ratio: float# 平均フラックス回復比 (復元強度 / 真値強度)
    mean_fwhm_recovery_ratio: float# 固有解像度回復比 (復元FWHM / 真値FWHM, 1.0に近いほど完全復元)
    avg_fwhm_reduction_pct: float  # 平均先鋭化率 (%)
    psnr_db: float                 # 直交復元画像のPSNR (dB)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_asymmetric_shoulder_psf(
    blur_length: float = 14.0,
    shoulder_decay: float = 0.18,
    sigma_rise: float = 1.2,
    sigma_fall: float = 2.0,
) -> np.ndarray:
    """実機測定に対応した、回転方向に肩（ショルダー）を持つ非対称走査PSFカーネルを生成します.

    走査走路上において、立ち上がり側に急峻なピークトップを持ち、
    回転方向（進行方向）に沿って高輝度を保つ肩（ショルダー/プラトー）を引きずり、
    終端で滑らかに背景へ減衰する非対称プロファイルを生成し、
    重心が幾何中心 (x=0) に厳密に一致するようサブピクセル微細補正して返します。
    """
    half_l = blur_length / 2.0
    half_k = int(np.ceil(blur_length * 1.5))
    x = np.arange(-half_k, half_k + 1, dtype=np.float32)
    # 走査範囲 [-half_l, +half_l] 内で立ち上がり側にピークを配置
    x_peak = -half_l + 2.0
    x_end = half_l

    kernel = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < x_peak:
            kernel[i] = np.exp(-0.5 * ((xi - x_peak) / max(sigma_rise, 1e-3)) ** 2)
        elif xi <= x_end:
            frac = (xi - x_peak) / max(x_end - x_peak, 1e-3)
            kernel[i] = 1.0 - shoulder_decay * frac
        else:
            kernel[i] = (1.0 - shoulder_decay) * np.exp(-0.5 * ((xi - x_end) / max(sigma_fall, 1e-3)) ** 2)

    s = np.sum(kernel)
    if s > 0:
        kernel /= s
    else:
        kernel[half_k] = 1.0

    # 重心の幾何中心 (0.0) への精密アライメント (位置バイアス・系統的シフトの完全排除)
    c0 = float(np.sum(x * kernel) / np.sum(kernel))
    if abs(c0) > 1e-4:
        shifted_kernel = shift(kernel, -c0, order=3, mode="constant", cval=0.0)
        shifted_kernel = np.maximum(0.0, shifted_kernel)
        s_shift = float(np.sum(shifted_kernel))
        if s_shift > 0:
            kernel = (shifted_kernel / s_shift).astype(np.float32)

    return kernel.astype(np.float32)


def generate_realistic_laue_simulation(
    size: int = 512,
    center: Tuple[float, float] = (256.0, 256.0),
    r0: float = 28.5,
    r_outer: float = 232.0,
    blur_length_pix: float = 12.5,
    num_zones: int = 12,
    spots_per_zone: int = 9,
    spot_sigma: float = 0.9,
    noise_sigma: float = 0.012,
    random_seed: int = 42,
    psf_mode: str = "asymmetric_shoulder",
) -> Tuple[np.ndarray, np.ndarray, List[GroundTruthSpot]]:
    """実機観測データ (/share/260930-1.png 等) の実測値に厳密に整合させたリアルな模擬ラウエ像を生成します.

    実測された以下の物理特性を忠実に再現します:
    - 中心ビームストッパー径 (r0 ≈ 28.5px) および外周円盤径 (r_outer ≈ 232px)
    - ゾーン軸に沿った約 108 個の回折スポット群
    - 動径方向のシャープな幅 (FWHM_r ≈ 2.1px, sigma_r = 0.9px)
    - 実測走査角 (Δθ ≈ 4.15°, 極座標 L_theta ≈ 12.5px) に基づく回転方向の非対称な肩 (ショルダー)
    - 実機同様のパレート的強度分布 (大半は 0.08〜0.35、主要ピークは 1.2〜2.1、中央値 ≈ 0.13)
    - 実機の背景散乱ベースライン (≈ 0.24 = 62 ADU) および中心ハロー光背

    Returns:
        blurred_image: 走査ブラー・背景散乱・ノイズ・ビームストッパー重畳後の劣化画像 (float32 [0, 1])
        ground_truth_image: ブラーなし・ノイズなしの真の直交スポット像 (float32 [0, 1])
        ground_truth_spots: 各真値スポットのパラメータリスト
    """
    rng = np.random.default_rng(random_seed)
    xc, yc = center

    # 1. ゾーン軸（晶帯軸）に沿ったスポット配置
    gt_spots: List[GroundTruthSpot] = []
    zone_angles = np.linspace(0.0, 360.0, num_zones, endpoint=False) + rng.uniform(-8.0, 8.0, num_zones)

    r_min = r0 * 1.08
    r_max = r_outer * 0.96

    spot_id = 1
    for z_idx, z_th in enumerate(zone_angles):
        # 実機の回折幾何に合わせた動径方向スポット配置 (外周ほど格子面密度が増加)
        u = np.linspace(0.05, 0.95, spots_per_zone)
        radii = r_min + (r_max - r_min) * (0.35 * u + 0.65 * (u ** 0.85))
        curve_factor = rng.uniform(-0.12, 0.12)

        for r_k in radii:
            r_rel = (r_k - r_min) / (r_max - r_min)
            th_deg = (z_th + curve_factor * 25.0 * (r_rel - 0.5) ** 2 + rng.normal(0, 0.9)) % 360.0
            th_rad = np.radians(th_deg)

            # 実機観測 (/share/260930-1) と整合する強度分布:
            # - 中央値 ≈ 0.16, 25%タイル ≈ 0.12, 75%タイル ≈ 0.25
            # - ゾーン軸交点や特定波長・結晶面でのスーパー強ピーク (1.2〜2.2, 確率約 10%)
            is_super_strong = (rng.uniform(0, 1) < 0.10) or (r_rel < 0.25 and rng.uniform(0, 1) < 0.20)
            if is_super_strong:
                intensity = float(rng.uniform(1.2, 2.2))
            else:
                base = 0.16 + 0.10 * float(np.exp(-1.0 * r_rel))
                intensity = float(np.clip(base * rng.lognormal(mean=0.0, sigma=0.35), 0.08, 0.85))

            gx = float(xc + r_k * np.cos(th_rad))
            gy = float(yc + r_k * np.sin(th_rad))

            true_fwhm = float(2.355 * spot_sigma)
            true_flux = float(intensity * 2.0 * np.pi * (spot_sigma ** 2))

            gt_spots.append(GroundTruthSpot(
                spot_id=spot_id,
                x=gx,
                y=gy,
                r=float(r_k),
                theta_deg=float(th_deg),
                true_peak_intensity=intensity,
                true_integrated_flux=true_flux,
                true_fwhm=true_fwhm,
            ))
            spot_id += 1

    # 2. 直交空間での Ground Truth 画像作成
    gt_image = np.zeros((size, size), dtype=np.float32)
    y_coords, x_coords = np.indices((size, size), dtype=np.float32)

    for s in gt_spots:
        d2 = (x_coords - s.x) ** 2 + (y_coords - s.y) ** 2
        patch_mask = d2 <= (4.0 * spot_sigma) ** 2
        gt_image[patch_mask] += s.true_peak_intensity * np.exp(-0.5 * d2[patch_mask] / (spot_sigma ** 2))

    # 3. 極座標空間への展開と方位角モーションブラーの畳み込み
    n_r = 384
    n_theta = 1080
    r_range = (r_min, r_max)

    polar_gt = cartesian_to_polar(
        gt_image,
        center=center,
        r_range=r_range,
        n_r=n_r,
        n_theta=n_theta,
    )

    # 実機対応の非対称ショルダー走査ブラー PSF 生成 (L_theta ≈ 12.5px, 走査角 4.15°)
    if psf_mode == "asymmetric_shoulder":
        psf_1d = create_asymmetric_shoulder_psf(
            blur_length=blur_length_pix,
            shoulder_decay=0.18,
            sigma_rise=1.2,
            sigma_fall=2.0,
        )
    else:
        psf_1d = create_psf_1d(blur_length=blur_length_pix, kernel_type="rect")

    k_len = len(psf_1d)
    k_padded = np.zeros(n_theta, dtype=np.float32)
    half_k = k_len // 2
    k_padded[: k_len - half_k] = psf_1d[half_k:]
    k_padded[-half_k:] = psf_1d[:half_k]

    K = np.fft.rfft(k_padded)
    F_polar = np.fft.rfft(polar_gt, axis=1)
    polar_blurred = np.fft.irfft(F_polar * K, n=n_theta, axis=1)
    polar_blurred = np.maximum(polar_blurred, 0.0)

    # 4. 直交座標系への逆写像
    cart_blurred = polar_to_cartesian(
        polar_blurred,
        output_shape=(size, size),
        center=center,
        r_range=r_range,
    )

    # 5. 実測散乱背景・ダイレクトビーム白丸・ノイズの合成
    dists = np.hypot(x_coords - xc, y_coords - yc)
    disk_mask = dists <= r_outer

    # 不均一散乱背景 (実測 baseline = 0.24 + 中心ハロー 0.06)
    bg_profile = 0.06 * np.exp(-0.5 * (dists / (r_outer * 0.6)) ** 2) + 0.24
    cart_blurred[disk_mask] += bg_profile[disk_mask]

    # 外側背景 (GUIウィンドウ風の薄灰色 0.95)
    cart_blurred[~disk_mask] = 0.95

    # 中心ビームストッパー白丸 (1.0 飽和)
    white_mask = dists <= r0
    cart_blurred[white_mask] = 1.0

    # 検出器ショットノイズ (実測 sigma ≈ 0.012 = 約 3〜4 ADU)
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, cart_blurred.shape).astype(np.float32)
        cart_blurred = np.clip(cart_blurred + noise, 0.0, 1.0)

    logger.info(
        "リアル模擬ラウエ像生成完了: サイズ=%dx%d, スポット数=%d, ブラー長=%.1fpx, ノイズ=%.3f",
        size, size, len(gt_spots), blur_length_pix, noise_sigma,
    )
    return cart_blurred, gt_image, gt_spots


def evaluate_peak_recovery(
    pipeline_result: Any,
    ground_truth_spots: List[GroundTruthSpot],
    ground_truth_image: np.ndarray,
    match_distance_tolerance_px: float = 4.5,
) -> Tuple[RecoveryBenchmarkMetrics, List[Dict[str, Any]]]:
    """パイプライン復元結果とGround Truthを照合し、定量的再現指標を算出します.

    Args:
        pipeline_result: AstroLauePipeline.run() が返した PipelineResult
        ground_truth_spots: 生成時の真値スポットリスト
        ground_truth_image: 真値スポット画像 (未劣化)
        match_distance_tolerance_px: 同一スポットとみなす許容重心距離 (px)

    Returns:
        metrics: 定量評価指標データクラス
        matched_pairs: [(gt_spot, detected_spot, distance_px), ...] の照合詳細
    """
    detected_spots = pipeline_result.spots
    n_gt = len(ground_truth_spots)
    n_det = len(detected_spots)

    if n_det == 0:
        return RecoveryBenchmarkMetrics(
            num_ground_truth=n_gt,
            num_detected=0,
            num_matched=0,
            recall=0.0,
            precision=0.0,
            f1_score=0.0,
            mean_position_error_px=float("nan"),
            position_rmse_px=float("nan"),
            intensity_r2=0.0,
            mean_flux_recovery_ratio=0.0,
            mean_fwhm_recovery_ratio=float("nan"),
            avg_fwhm_reduction_pct=0.0,
            psnr_db=0.0,
        ), []

    # KD-Tree による最近傍マッチング
    det_xy = np.array([[s.x, s.y] for s in detected_spots], dtype=np.float32)
    gt_xy = np.array([[s.x, s.y] for s in ground_truth_spots], dtype=np.float32)

    tree = cKDTree(det_xy)
    distances, indices = tree.query(gt_xy, distance_upper_bound=match_distance_tolerance_px)

    matched_pairs: List[Dict[str, Any]] = []
    used_det_indices = set()

    # 距離の近い順に 1対1 マッチング
    sorted_gt_indices = np.argsort(distances)
    for gt_idx in sorted_gt_indices:
        d = distances[gt_idx]
        if np.isinf(d) or d > match_distance_tolerance_px:
            continue
        det_idx = indices[gt_idx]
        if det_idx in used_det_indices:
            continue
        used_det_indices.add(det_idx)

        gt_s = ground_truth_spots[gt_idx]
        det_s = detected_spots[det_idx]

        matched_pairs.append({
            "gt_spot": gt_s,
            "det_spot": det_s,
            "distance_px": float(d),
            "gt_intensity": gt_s.true_peak_intensity,
            "det_intensity": det_s.peak_intensity,
            "gt_fwhm": gt_s.true_fwhm,
            "det_fwhm": det_s.fwhm_after,
            "fwhm_reduction": det_s.fwhm_reduction_ratio,
        })

    n_matched = len(matched_pairs)
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    precision = n_matched / n_det if n_det > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if n_matched > 0:
        pos_errors = [m["distance_px"] for m in matched_pairs]
        mean_pos_err = float(np.mean(pos_errors))
        rmse_pos_err = float(np.sqrt(np.mean(np.array(pos_errors) ** 2)))

        gt_ints = np.array([m["gt_intensity"] for m in matched_pairs])
        det_ints = np.array([m["det_intensity"] for m in matched_pairs])

        # 強度の相関 R^2
        if len(gt_ints) >= 3 and np.std(gt_ints) > 1e-4 and np.std(det_ints) > 1e-4:
            corr_mat = np.corrcoef(gt_ints, det_ints)
            intensity_r2 = float(corr_mat[0, 1] ** 2)
        else:
            intensity_r2 = 0.0

        # フラックス比
        flux_ratios = [m["det_intensity"] / max(m["gt_intensity"], 1e-4) for m in matched_pairs]
        mean_flux_ratio = float(np.median(flux_ratios))

        # FWHM 比率 (真のFWHM に対する復元後 FWHM の比率)
        fwhm_ratios = [m["det_fwhm"] / max(m["gt_fwhm"], 1e-4) for m in matched_pairs if m["det_fwhm"] > 0]
        mean_fwhm_ratio = float(np.median(fwhm_ratios)) if fwhm_ratios else float("nan")

        reds = [m["fwhm_reduction"] for m in matched_pairs if m["fwhm_reduction"] > 0]
        avg_red = float(np.mean(reds)) if reds else 0.0
    else:
        mean_pos_err = float("nan")
        rmse_pos_err = float("nan")
        intensity_r2 = 0.0
        mean_flux_ratio = 0.0
        mean_fwhm_ratio = float("nan")
        avg_red = 0.0

    # 復元画像と真値画像の PSNR 算出 (有効回折円盤内部)
    rest_img = pipeline_result.restored_cartesian
    h_gt, w_gt = ground_truth_image.shape[:2]
    h_r, w_r = rest_img.shape[:2]
    if (h_r, w_r) != (h_gt, w_gt):
        rest_resized = cv2.resize(rest_img, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)
    else:
        rest_resized = rest_img

    # 規格化
    gt_norm = ground_truth_image / max(ground_truth_image.max(), 1e-4)
    r_norm = rest_resized / max(rest_resized.max(), 1e-4)
    mse = float(np.mean((gt_norm - r_norm) ** 2))
    psnr_db = float(10.0 * np.log10(1.0 / max(mse, 1e-9)))

    metrics = RecoveryBenchmarkMetrics(
        num_ground_truth=n_gt,
        num_detected=n_det,
        num_matched=n_matched,
        recall=recall,
        precision=precision,
        f1_score=f1,
        mean_position_error_px=mean_pos_err,
        position_rmse_px=rmse_pos_err,
        intensity_r2=intensity_r2,
        mean_flux_recovery_ratio=mean_flux_ratio,
        mean_fwhm_recovery_ratio=mean_fwhm_ratio,
        avg_fwhm_reduction_pct=avg_red,
        psnr_db=psnr_db,
    )

    logger.info(
        "ピーク再現性ベンチマーク評価完了: Recall=%.1f%%, Precision=%.1f%%, F1=%.3f, 位置RMSE=%.2fpx, R^2=%.3f, PSNR=%.1fdB",
        recall * 100.0, precision * 100.0, f1, rmse_pos_err, intensity_r2, psnr_db,
    )
    return metrics, matched_pairs


def plot_recovery_benchmark_report(
    ground_truth_image: np.ndarray,
    blurred_image: np.ndarray,
    pipeline_result: Any,
    ground_truth_spots: List[GroundTruthSpot],
    matched_pairs: List[Dict[str, Any]],
    metrics: RecoveryBenchmarkMetrics,
    output_path: str | Path,
) -> Path:
    """ピーク再現性ベンチマークの総合検証レポート図 (4パネル+相関散布図) を生成・保存します."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12), facecolor="#1e1e1e")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.22)

    fig.suptitle(
        f"AstroLaue Synthetic Benchmark Report | Recall: {metrics.recall*100:.1f}% | Position RMSE: {metrics.position_rmse_px:.2f}px | R²: {metrics.intensity_r2:.3f}",
        fontsize=16,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )

    # 1. Ground Truth (真のシャープ像)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#121212")
    ax1.imshow(ground_truth_image, cmap="inferno", origin="upper")
    ax1.set_title(f"1. Ground Truth (N={metrics.num_ground_truth} Spots)", color="#ffffff", fontsize=12)
    ax1.axis("off")

    # 2. Blurred Input (走査ブラー・ノイズ付与像)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#121212")
    ax2.imshow(blurred_image, cmap="gray", origin="upper")
    ax2.set_title("2. Simulated Observation (Rotational Blur + Noise)", color="#ffffff", fontsize=12)
    ax2.axis("off")

    # 3. Restored Cartesian & Matched Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#121212")
    rest = pipeline_result.restored_cartesian
    ax3.imshow(rest, cmap="inferno", origin="upper")
    # 真値スポット (赤十字) と 検出スポット (シアン丸) の重ね合わせ
    for s in ground_truth_spots:
        ax3.plot(s.x, s.y, "+", color="#e74c3c", markersize=6, alpha=0.7)
    for s in pipeline_result.spots:
        ax3.plot(s.x, s.y, "o", markeredgecolor="#00ffff", markerfacecolor="none", markersize=7, alpha=0.8)
    ax3.plot([], [], "+", color="#e74c3c", label=f"Ground Truth ({metrics.num_ground_truth})")
    ax3.plot([], [], "o", markeredgecolor="#00ffff", markerfacecolor="none", label=f"Detected ({metrics.num_detected})")
    ax3.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="none", labelcolor="#ffffff", fontsize=9)
    ax3.set_title(f"3. Restored & Spot Match (Matched: {metrics.num_matched})", color="#ffffff", fontsize=12)
    ax3.axis("off")

    # 4. ピーク強度の相関散布図 (True vs Restored Intensity)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#181818")
    if matched_pairs:
        gt_ints = [m["gt_intensity"] for m in matched_pairs]
        det_ints = [m["det_intensity"] for m in matched_pairs]
        ax4.scatter(gt_ints, det_ints, color="#00ffff", edgecolors="#ffffff", s=45, alpha=0.85)
        # 対角基準線
        max_v = max(max(gt_ints), max(det_ints))
        ax4.plot([0, max_v], [0, max_v], "--", color="#e74c3c", label="Ideal 1:1 Line")
        ax4.set_xlabel("True Peak Intensity", color="#e0e0e0", fontsize=10)
        ax4.set_ylabel("Restored Peak Intensity", color="#e0e0e0", fontsize=10)
        ax4.set_title(f"4. Intensity Linearity (R² = {metrics.intensity_r2:.3f})", color="#ffffff", fontsize=11)
        ax4.grid(True, linestyle="--", alpha=0.3, color="#666666")
        ax4.legend(loc="upper left", facecolor="#2b2b2b", edgecolor="none", labelcolor="#ffffff", fontsize=9)
        ax4.tick_params(colors="#888888")

    # 5. 重心位置誤差ヒストグラム
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#181818")
    if matched_pairs:
        dists = [m["distance_px"] for m in matched_pairs]
        ax5.hist(dists, bins=12, color="#3498db", edgecolor="#ffffff", alpha=0.8)
        ax5.axvline(metrics.mean_position_error_px, color="#f39c12", linestyle="--", linewidth=2.0, label=f"Mean: {metrics.mean_position_error_px:.2f} px")
        ax5.axvline(metrics.position_rmse_px, color="#e74c3c", linestyle=":", linewidth=2.0, label=f"RMSE: {metrics.position_rmse_px:.2f} px")
        ax5.set_xlabel("Position Error [pixels]", color="#e0e0e0", fontsize=10)
        ax5.set_ylabel("Count", color="#e0e0e0", fontsize=10)
        ax5.set_title("5. Centroid Position Error Distribution", color="#ffffff", fontsize=11)
        ax5.grid(True, linestyle="--", alpha=0.3, color="#666666")
        ax5.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="none", labelcolor="#ffffff", fontsize=9)
        ax5.tick_params(colors="#888888")

    # 6. ベンチマーク指標サマリーテーブル
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#181818")
    ax6.axis("off")

    summary_text = (
        "=== Benchmark Summary Metrics ===\n\n"
        f"• Detection Recall (Sensitivity): {metrics.recall*100:.1f} %\n"
        f"• Detection Precision           : {metrics.precision*100:.1f} %\n"
        f"• F1-Score (Harmonic Mean)      : {metrics.f1_score:.3f}\n"
        f"• Matched Spots                 : {metrics.num_matched} / {metrics.num_ground_truth}\n"
        "------------------------------------\n"
        f"• Centroid RMSE (Accuracy)      : {metrics.position_rmse_px:.2f} px\n"
        f"• Mean Position Error           : {metrics.mean_position_error_px:.2f} px\n"
        "------------------------------------\n"
        f"• Intensity Linearity (R²)      : {metrics.intensity_r2:.3f}\n"
        f"• Average FWHM Reduction        : {metrics.avg_fwhm_reduction_pct:.1f} %\n"
        f"• FWHM Recovery to True PSF     : {metrics.mean_fwhm_recovery_ratio:.2f}x\n"
        f"• Image Quality (PSNR)          : {metrics.psnr_db:.1f} dB\n"
    )
    ax6.text(
        0.05, 0.95, summary_text,
        color="#ffffff",
        family="monospace",
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#242424", edgecolor="#444444"),
    )

    fig.savefig(str(out_p), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("シミュレーションベンチマークレポート図を保存しました: %s", out_p)
    return out_p


def plot_distribution_comparison(
    real_pipeline_result: Any,
    sim_pipeline_result: Any,
    output_path: str | Path,
) -> Path:
    """実機観測データ (/share) と模擬シミュレーションのプロファイル・幾何・強度分布比較図を生成・保存します.

    4つの重要指標を直接対比:
    1. 1D スポット回転プロファイル比較 (実機 vs シミュレーションの非対称肩・立ち上がり・テール)
    2. 動径方向スポット強度分布の比較 (r に対するピーク強度散布図)
    3. スポット走査ブラー幅 (FWHM) 分布ヒストグラム比較
    4. 背景散乱光の動径プロファイル比較 (中心ビームストッパーから外周円盤までの平均輝度)

    Args:
        real_pipeline_result: 実機画像 (/share/260930-1.png 等) の PipelineResult
        sim_pipeline_result: 模擬シミュレーション画像の PipelineResult
        output_path: 出力先ファイルパス (例: results_benchmark/distribution_comparison.png)

    Returns:
        output_path: 保存先パス
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12), facecolor="#1e1e1e")
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    fig.suptitle(
        "AstroLaue Profile & Distribution Comparison: Real Data (/share) vs. Simulation",
        fontsize=16,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )

    # 1. 1D スポット回転プロファイル比較 (非対称な肩・立ち上がり・テールの直接比較)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#181818")

    real_prof = real_pipeline_result.representative_profile
    sim_prof = sim_pipeline_result.representative_profile

    if real_prof and "prof_before" in real_prof and sim_prof and "prof_before" in sim_prof:
        # 実機プロファイル
        rx_r = real_prof["rel_pixels"]
        pb_r = real_prof["prof_before"]
        pb_r_base = float(np.median(np.concatenate([pb_r[:3], pb_r[-3:]])))
        pb_r_clean = np.maximum(0.0, pb_r - pb_r_base)
        pb_r_norm = pb_r_clean / max(pb_r_clean.max(), 1e-4)

        # シミュレーションプロファイル
        rx_s = sim_prof["rel_pixels"]
        pb_s = sim_prof["prof_before"]
        pb_s_base = float(np.median(np.concatenate([pb_s[:3], pb_s[-3:]])))
        pb_s_clean = np.maximum(0.0, pb_s - pb_s_base)
        pb_s_norm = pb_s_clean / max(pb_s_clean.max(), 1e-4)

        ax1.plot(rx_r, pb_r_norm, "o-", color="#3498db", linewidth=2.0, markersize=3, label="Real Spot (/share/260930-1)")
        ax1.plot(rx_s, pb_s_norm, "--", color="#e74c3c", linewidth=2.2, label="Simulation (Tuned Asymmetric Shoulder)")

        # モデルA: EMG (Exponentially Modified Gaussian) フィッティング
        def _emg(x, I0, x0, sigma, tau):
            z = (sigma / (np.sqrt(2) * max(tau, 1e-3))) - ((x - x0) / (np.sqrt(2) * max(sigma, 1e-3)))
            arg = np.clip(0.5 * (sigma / max(tau, 1e-3)) ** 2 - (x - x0) / max(tau, 1e-3), -50.0, 50.0)
            return I0 * (sigma / max(tau, 1e-3)) * np.sqrt(np.pi / 2.0) * np.exp(arg) * sp.erfc(z)

        # モデルC: Skew-Normal フィッティング
        def _skew(x, I0, x0, sigma, alpha):
            z = (x - x0) / max(sigma, 1e-3)
            phi = (1.0 / (np.sqrt(2 * np.pi) * max(sigma, 1e-3))) * np.exp(-0.5 * z ** 2)
            Phi = 0.5 * (1.0 + sp.erf(alpha * z / np.sqrt(2.0)))
            return 2.0 * I0 * phi * Phi

        # モデルB: Box-Gaussian (箱型畳み込み)
        def _box(x, I0, x0, sigma, Delta):
            t1 = sp.erf((x - x0) / (np.sqrt(2) * max(sigma, 1e-3)))
            t2 = sp.erf((x - x0 - Delta) / (np.sqrt(2) * max(sigma, 1e-3)))
            return (I0 / (2.0 * max(Delta, 1e-3))) * (t1 - t2)

        x_grid = np.linspace(rx_r.min(), rx_r.max(), 200)

        # モデルA フィッティング
        try:
            p_a, _ = curve_fit(_emg, rx_r, pb_r_norm, p0=[1.9, -5.0, 2.5, 6.0], maxfev=4000)
            y_a = _emg(rx_r, *p_a)
            r2_a = max(0.0, 1.0 - np.sum((pb_r_norm - y_a) ** 2) / max(np.sum((pb_r_norm - pb_r_norm.mean()) ** 2), 1e-6))
            ax1.plot(x_grid, _emg(x_grid, *p_a), ":", color="#2ecc71", linewidth=2.0, label=f"Model A (EMG, R²={r2_a:.3f})")
        except Exception:
            pass

        # モデルC フィッティング
        try:
            p_c, _ = curve_fit(_skew, rx_r, pb_r_norm, p0=[12.0, -6.0, 8.5, 4.0], maxfev=4000)
            y_c = _skew(rx_r, *p_c)
            r2_c = max(0.0, 1.0 - np.sum((pb_r_norm - y_c) ** 2) / max(np.sum((pb_r_norm - pb_r_norm.mean()) ** 2), 1e-6))
            ax1.plot(x_grid, _skew(x_grid, *p_c), "-.", color="#f39c12", linewidth=2.0, label=f"Model C (Skew-Norm, R²={r2_c:.3f})")
        except Exception:
            pass

        # モデルB フィッティング
        try:
            p_b, _ = curve_fit(_box, rx_r, pb_r_norm, p0=[12.0, -1.0, 4.5, 1.0], maxfev=4000)
            y_b = _box(rx_r, *p_b)
            r2_b = max(0.0, 1.0 - np.sum((pb_r_norm - y_b) ** 2) / max(np.sum((pb_r_norm - pb_r_norm.mean()) ** 2), 1e-6))
            ax1.plot(x_grid, _box(x_grid, *p_b), "--", color="#9b59b6", linewidth=1.8, alpha=0.85, label=f"Model B (Box-Gauss, R²={r2_b:.3f})")
        except Exception:
            pass

        ax1.axhline(0.5, color="#888888", linestyle=":", label="Half-Maximum (50%)")

        ax1.set_xlabel("Relative Azimuth Offset Δθ [pixels]", color="#e0e0e0", fontsize=10)
        ax1.set_ylabel("Normalized Intensity", color="#e0e0e0", fontsize=10)
        ax1.set_title("1. Azimuthal Spot Profile vs. Convolution Models (A, B, C)", color="#ffffff", fontsize=12)
        ax1.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="none", labelcolor="#ffffff", fontsize=8.5)
        ax1.grid(True, linestyle="--", alpha=0.3, color="#666666")
        ax1.tick_params(colors="#888888")

    # 2. 動径方向スポット強度分布の比較 (Radial Intensity Distribution)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#181818")

    r_real = [s.r for s in real_pipeline_result.spots]
    p_real = [s.peak_intensity for s in real_pipeline_result.spots]
    r_sim = [s.r for s in sim_pipeline_result.spots]
    p_sim = [s.peak_intensity for s in sim_pipeline_result.spots]

    ax2.scatter(r_real, p_real, color="#3498db", alpha=0.65, s=35, edgecolors="none", label=f"Real Spots (N={len(r_real)})")
    ax2.scatter(r_sim, p_sim, color="#e74c3c", alpha=0.65, s=35, marker="^", edgecolors="none", label=f"Simulated Spots (N={len(r_sim)})")

    ax2.set_xlabel("Radial Distance r [pixels]", color="#e0e0e0", fontsize=10)
    ax2.set_ylabel("Peak Intensity", color="#e0e0e0", fontsize=10)
    ax2.set_title("2. Radial Spot Intensity Distribution (r vs. Peak)", color="#ffffff", fontsize=12)
    ax2.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="none", labelcolor="#ffffff", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3, color="#666666")
    ax2.tick_params(colors="#888888")

    # 3. スポット走査ブラー幅 (FWHM) 分布ヒストグラム比較
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#181818")

    fwhm_real = [s.fwhm_before for s in real_pipeline_result.spots if s.fwhm_before > 0]
    fwhm_sim = [s.fwhm_before for s in sim_pipeline_result.spots if s.fwhm_before > 0]

    bins = np.linspace(2.0, 22.0, 16)
    ax3.hist(fwhm_real, bins=bins, color="#3498db", alpha=0.55, density=True, label=f"Real FWHM (Median={np.median(fwhm_real):.1f}px)")
    ax3.hist(fwhm_sim, bins=bins, color="#e74c3c", alpha=0.55, density=True, label=f"Simulated FWHM (Median={np.median(fwhm_sim):.1f}px)")

    ax3.set_xlabel("Blur FWHM Before Deblur [pixels]", color="#e0e0e0", fontsize=10)
    ax3.set_ylabel("Probability Density", color="#e0e0e0", fontsize=10)
    ax3.set_title("3. Observed Blur Length Distribution (FWHM Histogram)", color="#ffffff", fontsize=12)
    ax3.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="none", labelcolor="#ffffff", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.3, color="#666666")
    ax3.tick_params(colors="#888888")

    # 4. 背景散乱光の動径プロファイル比較
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#181818")

    # 極座標画像の各半径ごとの中央値 (背景レベル)
    real_bg_r = np.median(real_pipeline_result.polar_before, axis=1)
    sim_bg_r = np.median(sim_pipeline_result.polar_before, axis=1)

    r_norm_real = np.linspace(real_pipeline_result.r0, real_pipeline_result.r_outer, len(real_bg_r))
    r_norm_sim = np.linspace(sim_pipeline_result.r0, sim_pipeline_result.r_outer, len(sim_bg_r))

    ax4.plot(r_norm_real, real_bg_r, "-", color="#3498db", linewidth=2.2, label="Real Background Baseline")
    ax4.plot(r_norm_sim, sim_bg_r, "--", color="#e74c3c", linewidth=2.2, label="Simulated Background Baseline")

    ax4.set_xlabel("Radial Distance r [pixels]", color="#e0e0e0", fontsize=10)
    ax4.set_ylabel("Background Level (Polar Median)", color="#e0e0e0", fontsize=10)
    ax4.set_title("4. Radial Background Profile (Air Scatter & Halo)", color="#ffffff", fontsize=12)
    ax4.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="none", labelcolor="#ffffff", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.3, color="#666666")
    ax4.tick_params(colors="#888888")

    fig.savefig(str(out_p), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("実機 vs シミュレーション分布比較グラフを保存しました: %s", out_p)
    return out_p
